import torch

from mmaction.apis import init_recognizer, inference_recognizer

config_file = 'configs/recognition/tsn/tsn_r50_inference_1x1x3_100e_kinetics400_rgb.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = 'checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth'

# assign the desired device.
device = 'cuda:0' # or 'cpu'
device = torch.device(device)

 # build the model from a config file and a checkpoint file
model = init_recognizer(config_file, checkpoint_file, device=device)

print(model)