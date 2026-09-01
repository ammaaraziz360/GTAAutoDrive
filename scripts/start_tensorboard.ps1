# start_tensorboard.ps1
param(
    [string]$logdir = "runs"
)
Write-Host "Starting TensorBoard with logdir = $logdir"
Start-Process -NoNewWindow -FilePath python -ArgumentList "-m tensorboard.main --logdir $logdir"
Write-Host "TensorBoard started. Open http://localhost:6006 in your browser."