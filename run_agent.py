"""run_agent.py

Capture GTA V screen, run the student model, and send controls to the game.
Defaults (you chose):
 - checkpoint: checkpoints/best.pth
 - input size: 224
 - prev_frames: 2 (stack t-2,t-1,t)
 - control: vgamepad (XInput) if available, otherwise keyboard fallback via pynput
 - capture region: centered crop of primary monitor (60% width x 80% height)
 - emergency-stop: ESC toggles engaged state
 - smoothing alpha: 0.6

This script provides a `smoke_test()` function you can run to verify model load + a single forward pass
without sending any controls.

Notes:
 - Install dependencies: pip install mss opencv-python pynput
 - For XInput support install ViGEmBus on Windows and the `vgamepad` Python package
 - Run with: python run_agent.py --ckpt checkpoints/best.pth
"""

import argparse
import time
import os
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# optional deps
try:
    import mss
except Exception:
    mss = None
try:
    import cv2
except Exception:
    cv2 = None

# control backends
try:
    from pynput.keyboard import Controller as KeyController, Key, Listener as KeyListener, KeyCode
except Exception:
    KeyController = None
    Key = None
    KeyListener = None
    KeyCode = None
try:
    from vgamepad import VX360Gamepad
except Exception:
    VX360Gamepad = None
try:
    # XUSB_BUTTON enum provides named gamepad buttons (A/B/X/Y/LB/RB/etc.)
    from vgamepad import XUSB_BUTTON
except Exception:
    XUSB_BUTTON = None

from model import StudentPolicy
from dataset import IMNET_MEAN, IMNET_STD

IMNET_MEAN = np.array(IMNET_MEAN, dtype=np.float32)
IMNET_STD = np.array(IMNET_STD, dtype=np.float32)


def preprocess_pil(img_pil, size):
    img = img_pil.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - IMNET_MEAN) / IMNET_STD
    arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(arr)


class ControlSender:
    def __init__(self, handbrake_key='space', handbrake_button=None, swap_pedals=False, debug=False, brake_mode='handbrake'):
        # Priority: vgamepad (XInput) -> keyboard (pynput)
        self.use_vgamepad = (VX360Gamepad is not None)

        self.gp = None
        if self.use_vgamepad:
            try:
                self.gp = VX360Gamepad()
                print('vgamepad: VX360Gamepad created')
            except Exception as e:
                print('vgamepad init failed:', e)
                self.gp = None
                self.use_vgamepad = False

        if not self.use_vgamepad:
            if KeyController is None:
                raise RuntimeError('No control backend available: install vgamepad or pynput')
            self.kc = KeyController()
            # keep track of pressed keys to avoid repeats
            self._pressed = set()
        # handbrake config (keyboard key name or vgamepad button id)
        self.handbrake_key = handbrake_key
        # handbrake_button is interpreted as a vgamepad name (e.g. 'rb') when provided
        self.handbrake_vg_button = None
        if isinstance(handbrake_button, str) and XUSB_BUTTON is not None:
            hb = handbrake_button.strip().lower()
            if hb in ('rb', 'right_bumper', 'right_shoulder', 'rbumper'):
                try:
                    self.handbrake_vg_button = XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER
                except Exception:
                    self.handbrake_vg_button = None
        # If brake_mode is handbrake and no explicit button provided, default to RB when available
        if (self.handbrake_vg_button is None) and (str(brake_mode or 'handbrake').lower() == 'handbrake') and (XUSB_BUTTON is not None):
            try:
                self.handbrake_vg_button = XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER
            except Exception:
                self.handbrake_vg_button = None
        # optionally swap throttle and brake mapping for devices that appear reversed
        self.swap_pedals = bool(swap_pedals)
        # debug prints raw control values when True
        self.debug = bool(debug)
        # brake mode for vgamepad: 'trigger' (LT only), 'handbrake' (RB only), 'both'
        self.brake_mode = str(brake_mode or 'both').lower()

    def send(self, steer, thr, brk):
        """steer in [-1,1], thr/brk in [0,1]"""
        # allow swapping pedals at the ControlSender level if device mapping is inverted
        if getattr(self, 'swap_pedals', False):
            thr, brk = brk, thr
        # debug print for diagnostics
        # NOTE: keep prints minimal in the hot loop
        # print(f"ControlSender.send steer={steer:.3f} thr={thr:.3f} brk={brk:.3f} use_vjoy={self.use_vjoy}")
        if self.use_vgamepad and self.gp is not None:
            try:
                # map steer [-1,1] -> left joystick X [-32768, 32767]
                lx = int(max(-1.0, min(1.0, steer)) * 32767)
                ly = 0
                # triggers expect 0..255
                # NOTE: swap mapping so left trigger (LT) is BRAKE and right trigger (RT) is THROTTLE
                lt_raw = int(max(0.0, min(1.0, brk)) * 255)
                rt = int(max(0.0, min(1.0, thr)) * 255)
                # apply brake mode
                mode = getattr(self, 'brake_mode', 'both')
                if mode == 'handbrake':
                    lt = 0
                else:
                    lt = lt_raw
                try:
                    # many vgamepad bindings expose methods left_joystick(x_value, y_value)
                    # and left_trigger(value)/right_trigger(value)
                    self.gp.left_joystick(x_value=lx, y_value=ly)
                    self.gp.left_trigger(value=lt)
                    self.gp.right_trigger(value=rt)
                except TypeError:
                    # fallback to alternative arg names
                    self.gp.left_joystick(lx, ly)
                    self.gp.left_trigger(lt)
                    self.gp.right_trigger(rt)
                # handle discrete handbrake button on vgamepad if configured, then update once
                if getattr(self, 'handbrake_vg_button', None) is not None:
                    try:
                        if mode in ('handbrake', 'both') and brk > 0.5:
                            self.gp.press_button(self.handbrake_vg_button)
                        else:
                            # ensure it's released when not in use or in 'trigger' mode
                            self.gp.release_button(self.handbrake_vg_button)
                    except Exception:
                        pass
                self.gp.update()
                if self.debug:
                    print(f'[debug-controls] vgamepad lt={lt} rt={rt} lx={lx} ly={ly}')
            except Exception as e:
                print('vgamepad send failed:', e)
            # handle pyvjoy discrete handbrake above (self.handbrake_button) was handled later
        else:
            # keyboard fallback (discrete). Map steer to A/D, thr->W, brk->S
            # small deadzone for steer so slight jitter doesn't spam keys
            steer_dead = 0.05
            if steer < -steer_dead:
                self._hold_key('a')
                self._release_key('d')
            elif steer > steer_dead:
                self._hold_key('d')
                self._release_key('a')
            else:
                self._release_key('a')
                self._release_key('d')

            if thr > 0.5:
                self._hold_key('w')
            else:
                self._release_key('w')
            # Use SPACE for brake to avoid sending a 'S' key (which can trigger reverse)
            if brk > 0.5:
                # Key.space comes from pynput.keyboard.Key if available.
                kb = self.handbrake_key
                try:
                    if kb == 'space' and Key is not None:
                        self._hold_key(Key.space)
                    else:
                        # single-character keys like 'h' or 'f'
                        self._hold_key(kb)
                except Exception:
                    # fallback to ' ' character
                    self._hold_key(' ')
            else:
                kb = self.handbrake_key
                try:
                    if kb == 'space' and Key is not None:
                        self._release_key(Key.space)
                    else:
                        self._release_key(kb)
                except Exception:
                    self._release_key(' ')

        if self.debug:
            # show which keyboard keys are held (helpful for keyboard fallback)
            held = ','.join(str(x) for x in list(self._pressed)) if hasattr(self, '_pressed') else ''
            print(f'[debug-controls] keyboard held: {held}')

        # vgamepad discrete handbrake is handled above; pyvjoy was removed

    def _hold_key(self, ch):
        if ch in self._pressed:
            return
        self._pressed.add(ch)
        try:
            self.kc.press(ch)
        except Exception:
            pass

    def _release_key(self, ch):
        if ch not in self._pressed:
            return
        try:
            self.kc.release(ch)
        except Exception:
            pass
        self._pressed.discard(ch)

    def release_all(self):
        if self.use_vgamepad and self.gp is not None:
            try:
                # center joystick and zero triggers
                try:
                    self.gp.left_joystick(x_value=0, y_value=0)
                except TypeError:
                    self.gp.left_joystick(0, 0)
                try:
                    self.gp.left_trigger(value=0)
                    self.gp.right_trigger(value=0)
                except TypeError:
                    self.gp.left_trigger(0)
                    self.gp.right_trigger(0)
                # release vgamepad discrete button if used, then update once at the end
                if getattr(self, 'handbrake_vg_button', None) is not None:
                    try:
                        self.gp.release_button(self.handbrake_vg_button)
                    except Exception:
                        pass
                self.gp.update()
            except Exception:
                pass
        
        else:
            for k in list(self._pressed):
                self._release_key(k)


def get_center_region(mon, w_frac=0.6, h_frac=0.8):
    w = mon['width']
    h = mon['height']
    W = int(w * w_frac)
    H = int(h * h_frac)
    x = (w - W) // 2
    y = int(h * 0.08)  # slight top offset to include road
    return {'left': x, 'top': y, 'width': W, 'height': H}


def find_window_rect(title_substr: str):
    """Find a top-level window whose title contains title_substr (case-insensitive).
    Returns a dict compatible with mss region: {'left','top','width','height'}.
    Requires pywin32 (win32gui).
    """
    try:
        import win32gui
    except Exception:
        raise RuntimeError("Window capture requires pywin32 (pip install pywin32)")

    hwnd_found = None

    def _enum(hwnd, _lparam):
        nonlocal hwnd_found
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd) or ""
        if title_substr.lower() in title.lower():
            hwnd_found = hwnd

    win32gui.EnumWindows(_enum, None)
    if not hwnd_found:
        raise RuntimeError(f"Could not find a visible window with title containing: '{title_substr}'")

    left, top, right, bottom = win32gui.GetWindowRect(hwnd_found)
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        raise RuntimeError("Found window has zero area (is it minimized?).")
    return {'left': left, 'top': top, 'width': width, 'height': height}


def smoke_test(ckpt='checkpoints/best.pth', size=224, prev_frames=2, device='cpu'):
    print('Running smoke_test: load model and run one forward on synthetic input')
    in_ch = 3 * (1 + int(prev_frames))
    model = StudentPolicy(in_ch=in_ch, meta_dim=0, aux_tlight=False).to(device)
    if os.path.exists(ckpt):
        ck = torch.load(ckpt, map_location=device)
        model.load_state_dict(ck['model'])
        print('Loaded checkpoint', ckpt)
    else:
        print('Warning: checkpoint not found, using random init')
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, in_ch, size, size, device=device)
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        print('Output:', out)
    print('smoke_test done')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='checkpoints/best.pth')
    ap.add_argument('--size', type=int, default=260)
    ap.add_argument('--prev_frames', type=int, default=2)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    # (vJoy support removed) only vgamepad (XInput) or keyboard fallback are supported
    ap.add_argument('--fps', type=float, default=20.0)
    ap.add_argument('--alpha', type=float, default=1, help='smoothing alpha for EMA (applied when per-channel alphas not provided)')
    ap.add_argument('--alpha-steer', type=float, default=None, help='optional steering-specific smoothing alpha (0-1). Smaller values smooth more.')
    ap.add_argument('--alpha-throttle', type=float, default=None, help='optional throttle smoothing alpha (0-1). Defaults to --alpha if not set.')
    ap.add_argument('--alpha-brake', type=float, default=None, help='optional brake smoothing alpha (0-1). Defaults to --alpha if not set.')
    ap.add_argument('--window', type=str, default=None, help='(Windows) capture specific window by title substring')
    ap.add_argument('--region', type=str, default=None, help='explicit region to capture as left,top,width,height')
    ap.add_argument('--pedal-policy', type=str, default='scale', choices=['zero_smaller','scale','none'],
                    help="how to resolve simultaneous throttle+brake: 'zero_smaller' zeroes the smaller pedal, 'scale' scales throttle by (1-brake), 'none' leaves both")
    ap.add_argument('--steer-multiplier', type=float, default=2.5, help='multiply predicted steer by this value before sending (clamped to [-1,1])')
    ap.add_argument('--max-steer-change', type=float, default=0.25, help='maximum change in steering per frame (0.0-1.0). Lower = smoother but slower reaction.')
    ap.add_argument('--throttle-multiplier', type=float, default=1.25, help='multiply predicted throttle by this value before sending (clamped to [0,1])')
    ap.add_argument('--max-throttle-change', type=float, default=1, help='maximum change in throttle per frame (0.0-1.0). Lower = smoother acceleration.')
    ap.add_argument('--test-controls', action='store_true', help='run a short control output test and exit')
    ap.add_argument('--debug-controls', action='store_true', help='print raw axis/button/key values sent to the control backend')
    ap.add_argument('--swap-pedals', action='store_true', help='swap throttle and brake mapping for devices where they appear reversed')
    ap.add_argument('--handbrake-key', type=str, default='space', help="keyboard key to use for handbrake in keyboard fallback (default 'space')")
    ap.add_argument('--handbrake-button', type=str, default='None', help="vgamepad name like 'rb' to use as a discrete handbrake")
    ap.add_argument('--brake-mode', type=str, default='handbrake', choices=['trigger','handbrake','both'], help="vgamepad brake behavior: default 'handbrake' uses RB only (no reverse). 'trigger' uses LT only (may reverse when stopped). 'both' sends LT and RB.")
    ap.add_argument('--toggle-key', type=str, default='f4', help="keyboard key to toggle engaged state: e.g. 'esc', 'f9', 'space', or a single character like 'g'")
    args = ap.parse_args()

    device = torch.device(args.device)
    in_ch = 3 * (1 + max(0, int(args.prev_frames)))

    model = StudentPolicy(in_ch=in_ch, meta_dim=0, aux_tlight=False).to(device)
    if os.path.exists(args.ckpt):
        ck = torch.load(args.ckpt, map_location=device)
        try:
            model.load_state_dict(ck['model'])
            print('Loaded checkpoint', args.ckpt)
        except RuntimeError as e:
            print(f"Error loading checkpoint: {e}")
            print("This usually means the model architecture has changed (e.g. different input channels or head sizes).")
            print("Please retrain the model or use a compatible checkpoint.")
            return
    else:
        print('Checkpoint not found, running with random weights')
    model.eval()

    sender = ControlSender(
        handbrake_key=getattr(args, 'handbrake_key', 'space'),
        handbrake_button=getattr(args, 'handbrake_button', None),
        swap_pedals=getattr(args, 'swap_pedals', False),
        debug=getattr(args, 'debug_controls', False),
        brake_mode=getattr(args, 'brake_mode', 'both')
    )
    # Ensure devices start in a neutral state (prevent lingering inputs from prior runs)
    try:
        sender.release_all()
        print('Controls released at startup (neutral state).')
    except Exception:
        pass

    # If requested, run a short control test and exit. This helps debug whether
    # controls reach the system/controller or keyboard events are being generated.
    if getattr(args, 'test_controls', False):
        print('Running control output test...')
        try:
            # sequence: center, full throttle, full brake, left, right, center
            sender.release_all()
            time.sleep(0.2)
            print('Full throttle ->')
            sender.send(0.0, 1.0, 0.0)
            time.sleep(1.0)
            print('Full brake ->')
            sender.send(0.0, 0.0, 1.0)
            time.sleep(1.0)
            print('Steer left ->')
            sender.send(-1.0, 0.0, 0.0)
            time.sleep(1.0)
            print('Steer right ->')
            sender.send(1.0, 0.0, 0.0)
            time.sleep(1.0)
            print('Center ->')
            sender.release_all()
            print('Control test finished. Observe controller or game behaviour.')
        except Exception as e:
            print('Control test failed:', e)
        return

    # choose capture region
    if mss is None:
        raise RuntimeError('mss is required for screen capture: pip install mss')
    with mss.mss() as sct:
        mon = sct.monitors[1]
        # choose capture region: explicit --region, --window by title, or default center crop
        if args.region:
            try:
                parts = [int(p) for p in args.region.split(',')]
                if len(parts) != 4:
                    raise ValueError()
                region = {'left': parts[0], 'top': parts[1], 'width': parts[2], 'height': parts[3]}
            except Exception:
                print('Invalid --region format. Expected left,top,width,height')
                region = get_center_region(mon)
        elif args.window:
            try:
                region = find_window_rect(args.window)
            except Exception as e:
                print('Window capture failed:', e)
                print('Falling back to center region')
                region = get_center_region(mon)
        else:
            region = get_center_region(mon)
        print('Capture region:', region)

        # rolling buffer for previous frames
        buffer = deque(maxlen=1 + int(args.prev_frames))

        engaged = False

        # Resolve toggle key configuration
        toggle_name = str(getattr(args, 'toggle_key', 'esc') or 'esc').strip()
        toggle_key_obj = None
        toggle_char = None
        if Key is not None:
            if len(toggle_name) == 1:
                toggle_char = toggle_name.lower()
            else:
                try:
                    toggle_key_obj = getattr(Key, toggle_name.lower())
                except Exception:
                    toggle_key_obj = None

        # keyboard listener for emergency stop (toggle engaged)
        def on_press(key):
            nonlocal engaged
            try:
                if toggle_char is not None:
                    ch = getattr(key, 'char', None)
                    if ch is not None and str(ch).lower() == toggle_char:
                        engaged = not engaged
                        print('Engaged set to', engaged)
                elif toggle_key_obj is not None:
                    if key == toggle_key_obj:
                        engaged = not engaged
                        print('Engaged set to', engaged)
            except Exception:
                pass

        listener = None
        if KeyListener is not None:
            listener = KeyListener(on_press=on_press)
            listener.start()

        ema = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        alpha = float(args.alpha)

        def _resolve_alpha(value, fallback):
            try:
                if value is None:
                    return fallback
                return float(min(1.0, max(0.0, value)))
            except Exception:
                return fallback

        steer_alpha = _resolve_alpha(getattr(args, 'alpha_steer', None), alpha)
        throttle_alpha = _resolve_alpha(getattr(args, 'alpha_throttle', None), alpha)
        brake_alpha = _resolve_alpha(getattr(args, 'alpha_brake', None), alpha)
        alpha_vec = np.array([steer_alpha, throttle_alpha, brake_alpha], dtype=np.float32)
        period = 1.0 / float(args.fps)

        # show which control backend is active
        backend = 'vgamepad' if sender.use_vgamepad and sender.gp is not None else 'keyboard'
        backend_info = backend + (', pedals_swapped' if getattr(sender, 'swap_pedals', False) else '')
        if toggle_char is not None:
            toggle_disp = toggle_char.upper()
        elif toggle_key_obj is not None:
            toggle_disp = toggle_name.upper()
        else:
            toggle_disp = 'ESC'
        print(f'Starting main loop. control backend={backend_info}. Press {toggle_disp} to toggle sending controls. Ctrl+C to quit.')
        prev_engaged = engaged
        first_iter = True
        prev_steer_sent = 0.0
        prev_throttle_sent = 0.0
        
        try:
            while True:
                t0 = time.time()
                img = sct.grab(region)
                # convert to PIL
                img_pil = Image.frombytes('RGB', (img.width, img.height), img.rgb)
                ten = preprocess_pil(img_pil, args.size)
                buffer.append(ten)
                if len(buffer) < buffer.maxlen:
                    # pad with copies of first frame
                    while len(buffer) < buffer.maxlen:
                        buffer.appendleft(ten)
                inp = torch.cat(list(buffer), dim=0).unsqueeze(0).to(device)
                with torch.no_grad():
                    out = model(inp)
                    if isinstance(out, tuple):
                        out = out[0]
                    steer = float(out[0,0].item())
                    thr = float(out[0,1].item())
                    brk = float(out[0,2].item())

                # smoothing (per-channel alpha)
                preds = np.array([steer, thr, brk], dtype=np.float32)
                ema = alpha_vec * preds + (1.0 - alpha_vec) * ema
                s_sm, t_sm, b_sm = float(ema[0]), float(ema[1]), float(ema[2])

                # apply steer multiplier (user-provided), clamp to [-1,1]
                steer_mult = float(getattr(args, 'steer_multiplier', 1.0))
                s_sm = max(-1.0, min(1.0, s_sm * steer_mult))

                # Rate limiting (Slew Rate) to prevent jerky steering
                # Limits the maximum change in steering value per frame
                max_change = float(getattr(args, 'max_steer_change', 1.0))
                delta = s_sm - prev_steer_sent
                # Clip delta to [-max_change, max_change]
                delta = max(-max_change, min(max_change, delta))
                s_sm = prev_steer_sent + delta
                prev_steer_sent = s_sm

                # apply throttle multiplier (user-provided), clamp to [0,1]
                thr_mult = float(getattr(args, 'throttle_multiplier', 1.0))
                t_sm = max(0.0, min(1.0, t_sm * thr_mult))

                # Rate limiting for throttle
                max_thr_change = float(getattr(args, 'max_throttle_change', 1.0))
                delta_t = t_sm - prev_throttle_sent
                delta_t = max(-max_thr_change, min(max_thr_change, delta_t))
                t_sm = prev_throttle_sent + delta_t
                prev_throttle_sent = t_sm

                # small deadzone to avoid jitter
                pedal_dead = 0.02
                if t_sm < pedal_dead:
                    t_sm = 0.0
                if b_sm < pedal_dead:
                    b_sm = 0.0

                # resolve simultaneous throttle+brake based on chosen policy
                policy = getattr(args, 'pedal_policy', 'zero_smaller')
                if policy == 'zero_smaller':
                    # zero the smaller pedal so they are mutually exclusive
                    if t_sm > 0 and b_sm > 0:
                        if t_sm > b_sm:
                            b_sm = 0.0
                        else:
                            t_sm = 0.0
                elif policy == 'scale':
                    # scale throttle by (1 - brake) so heavy braking reduces throttle
                    t_sm = t_sm * (1.0 - b_sm)
                # else 'none' leaves both as predicted

                print(f"pred steer={s_sm:+.3f} thr={t_sm:.3f} brk={b_sm:.3f}  engaged={engaged}")

                # Send or release controls depending on engaged state. When turning OFF,
                # explicitly call release_all() so devices are driven to neutral and the
                # game doesn't keep applying the last command (which can cause reversing).
                if engaged:
                    sender.send(s_sm, t_sm, b_sm)
                else:
                    # release controls on the first loop or on transition from engaged->disengaged
                    if first_iter or prev_engaged:
                        try:
                            sender.release_all()
                        except Exception:
                            pass

                prev_engaged = engaged
                first_iter = False

                # sleep to maintain fps
                dt = time.time() - t0
                to_sleep = max(0.0, period - dt)
                time.sleep(to_sleep)
        except KeyboardInterrupt:
            print('Stopping...')
        finally:
            print('Releasing controls')
            try:
                sender.release_all()
            except Exception:
                pass
            if listener is not None:
                listener.stop()


if __name__ == '__main__':
    main()
