import torch, sys
print('torch.version:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('cuda device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    try:
        print('cuda version:', torch.version.cuda)
    except Exception as e:
        print('cuda version: unknown', e)
    try:
        print('device name:', torch.cuda.get_device_name(0))
    except Exception as e:
        print('device name: error', e)
print('default device for new tensors:', 'cuda' if torch.cuda.is_available() else 'cpu')
