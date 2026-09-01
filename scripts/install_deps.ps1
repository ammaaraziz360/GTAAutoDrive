# install_deps.ps1
# Install required Python packages for the GTAAutoDrive project
python -m pip install --upgrade pip
pip install -r ../requirements.txt

# Note: If you plan to use vgamepad (XInput via ViGEmBus) you must also install the ViGEmBus driver from:
# https://github.com/ViGEm/ViGEmBus/releases
# After installing ViGEmBus, install the Python binding:
# pip install vgamepad

Write-Host "Dependencies installed (you may need to install ViGEmBus driver separately)."