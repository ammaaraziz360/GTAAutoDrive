# run_gradcam.ps1
param(
    [int]$frames_T = 3,
    [string]$ckpt = "../checkpoints/best.pth",
    [string]$out_dir = "gradcam_out"
)
python ..\grad_cam_steer.py --frames_T $frames_T --ckpt $ckpt --out_dir $out_dir
