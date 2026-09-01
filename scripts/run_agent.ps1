# run_agent.ps1
param(
    [string]$window = "Grand Theft Auto V",
    [string]$ckpt = "checkpoints/best.pth",
    [float]$steer_multiplier = 1.0,
    [float]$throttle_multiplier = 1.0,
    [string]$pedal_policy = "zero_smaller"
)
python ..\run_agent.py --window "$window" --ckpt "$ckpt" --steer-multiplier $steer_multiplier --throttle-multiplier $throttle_multiplier --pedal-policy $pedal_policy
