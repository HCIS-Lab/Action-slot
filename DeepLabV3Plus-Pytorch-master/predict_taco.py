from torch.utils.data import dataset
from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, cityscapes
from torchvision import transforms as T
from metrics import StreamSegMetrics

import torch
import torch.nn as nn

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from glob import glob

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--input", type=str, default='/media/hankung/ssd/oats/oats_data/images',
                        help="path to a single image or image directory")
    parser.add_argument("--dataset", type=str, default='cityscapes',
                        choices=['voc', 'cityscapes'], help='Name of training set')

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )

    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet101',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--save_val_results_to", default='/media/hankung/ssd/oats/oats_data/',
                        help="save segmentation results to the specified dir")

    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    
    parser.add_argument("--ckpt", default='/media/hankung/ssd/best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar', type=str,
                        help="resume from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    return parser

def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
        decode_fn = VOCSegmentation.decode_target
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
        decode_fn = Cityscapes.decode_target

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    maps = ['ap_Town01', 'ap_Town02', 'ap_Town03', 'ap_Town04', 'ap_Town05', 'ap_Town06', 'ap_Town07','ap_Town10HD',
            'interactive', 'non-interactive',
            'runner_Town03', 'runner_Town05', 'runner_Town10HD'
        ]
    for map in maps:
        map_path = os.path.join(opts.input, map)
        basic_list = [os.path.join(map_path, basic) for basic in os.listdir(map_path)]
        for basic in basic_list:
            variant_list = [os.path.join(basic, 'variant_scenario', variant) for variant in os.listdir(basic, 'variant_scenario')]
            for variant in variant_list:
                video_list = [os.path.join(variant, 'rgb', 'front', img) for img in os.listdir(variant, 'rgb', 'front')]
            print(variant)
        folder_path = os.path.join(opts.input, scenario)
        save_path = os.path.join(variant, scenario+'_segmentation_32x96')
        # Setup dataloader
        image_files = []
        if os.path.isdir(folder_path):
            for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
                files = glob(os.path.join(folder_path, '**/*.%s'%(ext)), recursive=True)
                if len(files)>0:
                    image_files.extend(files)
        # elif os.path.isfile(opts.input):
        #     image_files.append(opts.input)
        
        # Set up model (all models are 'constructed at network.modeling)
        model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
        if opts.separable_conv and 'plus' in opts.model:
            network.convert_to_separable_conv(model.classifier)
        utils.set_bn_momentum(model.backbone, momentum=0.01)
        
        if opts.ckpt is not None and os.path.isfile(opts.ckpt):
            # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
            checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint["model_state"])
            model = nn.DataParallel(model)
            model.to(device)
            print("Resume model from %s" % opts.ckpt)
            del checkpoint
        else:
            print("[!] Retrain")
            model = nn.DataParallel(model)
            model.to(device)


        transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
        with torch.no_grad():
            model = model.eval()
            for img_path in tqdm(image_files):
                ext = os.path.basename(img_path).split('.')[-1]
                img_name = os.path.basename(img_path)[:-len(ext)-1]
                img = Image.open(img_path).convert('RGB')
                img = transform(img).unsqueeze(0) # To tensor of NCHW
                img = img.to(device)
                
                pred = model(img).max(1)[1].cpu().numpy()[0] # HW
                colorized_preds = decode_fn(pred).astype('uint8')
                colorized_preds = Image.fromarray(colorized_preds)
                colorized_preds = colorized_preds.resize((28, 28), Image.NEAREST)
                if opts.save_val_results_to:
                    colorized_preds.save(os.path.join(save_path, img_name+'.png'))

if __name__ == '__main__':
    main()
