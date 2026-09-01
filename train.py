# train.py
import os, math, json, random, argparse, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from tqdm import tqdm
from dataset import FramesControls, IMNET_MEAN, IMNET_STD
from model import StudentPolicy

def loss_fn(pred, target,
            weights=(1.5, 1.0, 1.0),
            pedal_lambda=0.1,
            steer_boost_scale=3.0,
            throttle_boost_scale=2.0,
            brake_boost_scale=2.0):
    """Weighted SmoothL1 + Decoupled Action Boosting + Pedal Exclusivity.

    weights: base importance (steer, throttle, brake)
    pedal_lambda: penalty on simultaneous throttle/brake predictions
    steer_boost_scale: amplify steering loss during turns
    throttle_boost_scale: amplify throttle loss during acceleration (fixes laziness)
    brake_boost_scale: amplify brake loss during braking (ensures safety)
    """
    if pred.shape[-1] != 3:
        raise ValueError('loss_fn expects 3 outputs: steer, throttle, brake')

    w = pred.new_tensor(weights).view(1, 3)
    smooth = F.smooth_l1_loss(pred, target, reduction='none') # [B, 3]

    # 1. Steering Boost (Applied only to steering channel)
    # When target steering is high, we punish steering errors more.
    s_mag = target[:, 0].abs()
    s_boost = 1.0 + (steer_boost_scale - 1.0) * torch.tanh(2.0 * s_mag)
    l_steer = smooth[:, 0] * w[0, 0] * s_boost

    # 2. Throttle Boost (Applied only to throttle channel)
    # When target throttle is high, we punish throttle errors more.
    # This forces the model to commit to acceleration instead of coasting/braking.
    t_mag = target[:, 1]
    t_boost = 1.0 + (throttle_boost_scale - 1.0) * torch.tanh(2.0 * t_mag)
    l_thr = smooth[:, 1] * w[0, 1] * t_boost

    # 3. Brake Boost (Applied only to brake channel)
    # When target brake is high, we punish brake errors more (safety critical).
    b_mag = target[:, 2]
    b_boost = 1.0 + (brake_boost_scale - 1.0) * torch.tanh(2.0 * b_mag)
    l_brk = smooth[:, 2] * w[0, 2] * b_boost

    base = l_steer + l_thr + l_brk

    # Enhanced pedal exclusivity: penalize product of throttle and brake.
    thr = pred[:, 1].clamp(min=0.0)
    brk = pred[:, 2].clamp(min=0.0)
    pedal_penalty = pedal_lambda * (thr * brk)

    return (base + pedal_penalty).mean()

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # number of previous frames to stack (0 => single frame)
    num_prev = max(0, int(args.prev_frames))
    in_ch = 3 * (1 + num_prev)

    # TensorBoard writer
    logdir = args.logdir if getattr(args, 'logdir', None) else os.path.join("runs", f"exp-{time.strftime('%Y%m%d-%H%M%S')}")
    writer = SummaryWriter(log_dir=logdir)
    print(f"TensorBoard logs -> {logdir}")
    global_step = 0

    ds = FramesControls(args.frames, args.labels, split_file=None,
                        train=True, input_size=args.size, use_prev=num_prev,
                        oversample_turns=args.oversample_turns, turn_threshold=0.15)

    # split into train/val (or pass explicit split files for environment-wise split)
    # We use session-based splitting to avoid data leakage (adjacent frames are too similar).
    # If we just used random_split, the model would memorize the track rather than learn to drive.
    val_ratio = 0.2
    
    # Group indices by session
    session_indices = {}
    for idx, item in enumerate(ds.items):
        # item is like "session-0001/frame_000012.jpg"
        # Handle both forward and backward slashes just in case
        item_norm = item.replace('\\', '/')
        sess = item_norm.split('/')[0]
        session_indices.setdefault(sess, []).append(idx)
    
    sessions = sorted(list(session_indices.keys()))
    n_val_sessions = max(1, int(len(sessions) * val_ratio))
    
    # Take the last N sessions for validation (simulating "future" or "unseen" data)
    val_sessions = sessions[-n_val_sessions:]
    train_sessions = sessions[:-n_val_sessions]
    
    print(f"Training sessions: {len(train_sessions)} {train_sessions}")
    print(f"Validation sessions: {len(val_sessions)} {val_sessions}")
    
    train_idx = []
    for s in train_sessions:
        train_idx.extend(session_indices[s])
        
    val_idx = []
    for s in val_sessions:
        val_idx.extend(session_indices[s])
        
    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds = torch.utils.data.Subset(ds, val_idx)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = StudentPolicy(in_ch=in_ch, meta_dim=0, aux_tlight=args.aux_tlight).to(device)
    
    # Count and display model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = GradScaler('cuda')
    
    # Configure learning rate schedule
    if args.schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    elif args.schedule == "step":
        # Reduce LR by 0.1 at 50% and 75% of training
        milestones = [args.epochs // 2, args.epochs * 3 // 4]
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=0.1)
    elif args.schedule == "plateau":
        # verbose is deprecated in newer PyTorch versions, use get_last_lr() to check
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)
    elif args.schedule == "none":
        sched = None
    else:
        raise ValueError(f"Unknown schedule: {args.schedule}")

    ce = nn.CrossEntropyLoss()

    best = math.inf
    for epoch in range(1, args.epochs+1):
        model.train()
        tbar = tqdm(train_loader, desc=f"epoch {epoch} train")
        tr_loss = 0.0
        for x, y, tcls in tbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            
            with autocast('cuda'):
                if args.aux_tlight:
                    pred, tlogits = model(x)
                    l = loss_fn(pred, y)
                    l += 0.2*ce(tlogits, tcls.to(device))
                else:
                    pred = model(x)
                    l = loss_fn(pred, y)
            
            scaler.scale(l).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            
            tr_loss += l.item()*x.size(0)
            current_lr = opt.param_groups[0]["lr"]
            tbar.set_postfix(loss=f"{l.item():.4f}", lr=f"{current_lr:.2e}")

            # log training scalars every N steps
            global_step += 1
            if global_step % args.log_every == 0:
                writer.add_scalar("train/loss", l.item(), global_step)
                writer.add_scalar("train/lr", current_lr, global_step)

        # validate
        model.eval()
        va_loss, n = 0.0, 0
        mae_steer = 0.0
        mae_throttle = 0.0
        mae_brake = 0.0
        mae_steer_turns = 0.0
        n_turns = 0
        correct_signs = 0

        with torch.no_grad():
            for x, y, tcls in tqdm(val_loader, desc="val"):
                x, y = x.to(device), y.to(device)
                if args.aux_tlight:
                    pred, tlogits = model(x)
                    l = loss_fn(pred, y) + 0.2*ce(tlogits, tcls.to(device))
                else:
                    pred = model(x)
                    l = loss_fn(pred, y)
                va_loss += l.item()*x.size(0)
                
                # Metrics
                steer_pred = pred[:, 0]
                steer_gt = y[:, 0]
                thr_pred = pred[:, 1]
                thr_gt = y[:, 1]
                brk_pred = pred[:, 2]
                brk_gt = y[:, 2]
                
                # 1. Overall MAE
                mae_steer += (steer_pred - steer_gt).abs().sum().item()
                mae_throttle += (thr_pred - thr_gt).abs().sum().item()
                mae_brake += (brk_pred - brk_gt).abs().sum().item()
                
                # 2. Turn Analysis (where GT steering > 0.1)
                turn_mask = steer_gt.abs() > 0.1
                if turn_mask.any():
                    n_t = turn_mask.sum().item()
                    n_turns += n_t
                    mae_steer_turns += (steer_pred[turn_mask] - steer_gt[turn_mask]).abs().sum().item()
                    
                    # 3. Direction Accuracy (did we turn the right way?)
                    # Check if signs match (product > 0)
                    same_sign = (steer_pred[turn_mask] * steer_gt[turn_mask]) > 0
                    correct_signs += same_sign.float().sum().item()

                n += x.size(0)

        va_loss /= n
        mae_steer /= n
        mae_throttle /= n
        mae_brake /= n
        mae_turns = mae_steer_turns / max(1, n_turns)
        acc_sign = correct_signs / max(1, n_turns)
        
        print(f"[epoch {epoch}] val loss: {va_loss:.4f} | steer MAE: {mae_steer:.4f} | thr MAE: {mae_throttle:.4f} | brk MAE: {mae_brake:.4f} | turn MAE: {mae_turns:.4f} | turn Sign Acc: {acc_sign:.2%}")

        # log validation scalars
        writer.add_scalar("val/loss", va_loss, epoch)
        writer.add_scalar("val/mae_steer", mae_steer, epoch)
        writer.add_scalar("val/mae_throttle", mae_throttle, epoch)
        writer.add_scalar("val/mae_brake", mae_brake, epoch)
        writer.add_scalar("val/mae_steer_turns", mae_turns, epoch)
        writer.add_scalar("val/acc_sign_turns", acc_sign, epoch)

        if sched is not None:
            if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                sched.step(va_loss)
            else:
                sched.step()

        # optionally log a batch of validation images (current frame) every log_img_freq epochs
        if getattr(args, 'log_img_freq', 0) and (epoch % args.log_img_freq == 0):
            try:
                xb, yb, _ = next(iter(val_loader))
                # always visualize the current frame (last 3 channels)
                xb_vis = xb[:, -3:, :, :]
                mean = torch.tensor(IMNET_MEAN).view(1,3,1,1)
                std  = torch.tensor(IMNET_STD).view(1,3,1,1)
                xb_vis = xb_vis * std + mean
                xb_vis = xb_vis.clamp(0.0, 1.0)
                xb_vis = xb_vis.cpu()
                grid = vutils.make_grid(xb_vis, nrow=min(8, xb_vis.size(0)))
                writer.add_image("val/images", grid, epoch)
            except Exception as e:
                print("Image logging skipped:", e)

        writer.flush()

        if va_loss < best:
            best = va_loss
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({"model": model.state_dict(), "args": vars(args)}, "checkpoints/best.pth")
            print("✓ saved checkpoints/best.pth")

    # close TensorBoard writer
    try:
        writer.close()
    except Exception:
        pass

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="data/frames")
    ap.add_argument("--labels", default="data/labels_teacher")
    ap.add_argument("--size", type=int, default=260)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--prev_frames", type=int, default=2, help="number of previous frames to stack (0,1,2 => e.g. 2 -> [t-2,t-1,t])")
    ap.add_argument("--aux_tlight", action="store_true")
    ap.add_argument("--logdir", type=str, default=None, help="tensorboard log directory (default: runs/exp-<timestamp>)")
    ap.add_argument("--log_every", type=int, default=20, help="log training scalars every N steps")
    ap.add_argument("--log_img_freq", type=int, default=1, help="log validation images every N epochs (0 to disable)")
    ap.add_argument("--workers", type=int, default=8, help="number of dataloader workers")
    ap.add_argument("--wd", type=float, default=1e-2, help="weight decay (default 0.01 for AdamW)")
    ap.add_argument("--schedule", type=str, default="cosine", choices=["cosine", "step", "plateau", "none"],
                   help="LR schedule: cosine (default), step (at 50%/75%), or none (constant)")
    ap.add_argument("--oversample_turns", action="store_true", help="Oversample turn samples to balance dataset")
    args = ap.parse_args()
    train(args)
