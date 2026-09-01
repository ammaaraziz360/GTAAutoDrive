# run_train.ps1
param(
    [int]$prev_frames = 2,
    [int]$bs = 32,
    [int]$epochs = 50
)
python ..\train.py --prev_frames $prev_frames --bs $bs --epochs $epochs
