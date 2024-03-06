import argparse
import json
import os

import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
from hsluv import hsluv_to_rgb
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import matplotlib.image
from scipy.optimize import linear_sum_assignment



import sys


import taco_fp
from model import generate_model
from utils import *
from parser_eval import get_eval_parser

sys.path.append('../datasets')
sys.path.append('../configs')
sys.path.append('../models')

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

torch.backends.cudnn.benchmark = True


actor_table = ['c:z1-z2', 'c:z1-z3', 'c:z1-z4',
                'c:z2-z1', 'c:z2-z3', 'c:z2-z4',
                'c:z3-z1', 'c:z3-z2', 'c:z3-z4',
                'c:z4-z1', 'c:z4-z2', 'c:z4-z3',
                'b:z1-z2', 'b:z1-z3', 'b:z1-z4',
                'b:z2-z1', 'b:z2-z3', 'b:z2-z4',
                'b:z3-z1', 'b:z3-z2', 'b:z3-z4',
                'b:z4-z1', 'b:z4-z2', 'b:z4-z3',
                'c+:z1-z2', 'c+:z1-z3', 'c+:z1-z4',
                'c+:z2-z1', 'c+:z2-z3', 'c+:z2-z4',
                'c+:z3-z1', 'c+:z3-z2', 'c+:z3-z4',
                'c+:z4-z1', 'c+:z4-z2', 'c+:z4-z3',
                'b+:z1-z2', 'b+:z1-z3', 'b+:z1-z4',
                'b+:z2-z1', 'b+:z2-z3', 'b+:z2-z4',
                'b+:z3-z1', 'b+:z3-z2', 'b+:z3-z4',
                'b+:z4-z1', 'b+:z4-z2', 'b+:z4-z3',
                'p:c1-c2', 'p:c1-c4', 
                'p:c2-c1', 'p:c2-c3', 
                'p:c3-c2', 'p:c3-c4', 
                'p:c4-c1', 'p:c4-c3', 
                'p+:c1-c2', 'p+:c1-c4', 
                'p+:c2-c1', 'p+:c2-c3', 
                'p+:c3-c2', 'p+:c3-c4', 
                'p+:c4-c1', 'p+:c4-c3',
                'bg'] 
                    
def generate_distinct_colors(num_colors):
    colors = []
    for i in range(num_colors):
        hue = (i * 360.0 / num_colors) % 360.0
        saturation = 75.0  # Adjust as needed
        lightness = 65.0   # Adjust as needed
        rgb_color = hsluv_to_rgb((hue, saturation, lightness))
        colors.append(np.array(rgb_color))
    return colors

def plot_slot(attn, model_name, map, id, v, raw, actor, pred_actor, logdir, threshold, mode):


    num_pos = 0
    num_tp = 0
    actor_str = ''
    actor = actor[0]
    pred_actor = pred_actor[0]

    pred_actor = torch.sigmoid(pred_actor)
    pred_actor = pred_actor > 0.5
    if args.allocated_slot:
        for i, a in enumerate(actor):
            if a.data == 1.0:
                num_pos += 1
                actor_str += actor_table[i]
                if pred_actor[i].data == True:
                    actor_str += '  TP'
                    num_tp +=1
                else:
                    actor_str += '          FN'
            
            else:
                if pred_actor[i].data == True:
                    actor_str += actor_table[i] 
                    actor_str += '                  FP'
                else:
                    actor_str += actor_table[i] 
                    actor_str += '                          TN'
            actor_str +='\n'

        path = os.path.join(logdir, 'plot_'+ mode +'_'+str(threshold))
        if not os.path.exists(path):
            os.makedirs(path)
        
        path = os.path.join(path, map+'_'+id + '_' + v)
        if not os.path.exists(path):
            os.makedirs(path)

        with open(os.path.join(path, "label_result.txt"), "w") as text_file:
            text_file.write(actor_str)

    cmap = plt.get_cmap('rainbow')
    colors = [cmap(ii) for ii in np.linspace(0, 1, 20)]


    raw = torch.stack(raw, dim=0)
    raw = torch.permute(raw, (1,2,0,3,4))
    seq_len = 16
    attn = attn.detach()
    m_l, m_n, m_h, m_w = attn.shape[1], attn.shape[2], attn.shape[3], attn.shape[4]
    attn = torch.reshape(attn, (-1, 1, m_h, m_w))
    # masks = F.interpolate(masks, (masks.shape[-3], 128,384))
    attn = F.interpolate(attn, (128,384), mode='bilinear')
    attn = torch.reshape(attn, (1, m_l, m_n, 128, 384))

    # image = image.permute(0, 2, 1, 3, 4)
    raw = raw.permute(0, 2, 1, 3, 4)
    # cur_image = F.interpolate(image, (3, 128,384))
    cur_raw = F.interpolate(raw, (3, 128,384))
    attn = attn[0]
    # cur_image = cur_image[0]
    cur_raw = cur_raw[0]
    
    for j in range(seq_len):
        # image_j = cur_image[j].permute(1,2,0).cpu().numpy()
        raw_j = cur_raw[j].permute(1,2,0).cpu().numpy()
        # image_j = image_j * 0.5 + 0.5
        new_raw_j = raw_j * 0.8 + 0.1
        masks_j = attn[j]

        masks_j = masks_j.cpu().numpy()
        if mode == 'fp':
            alpha_1 = 0.2
            alpha_2 = 0.2
            alpha_3 = 0.2

            color_1 = np.array([1.0, 0.0, 0.0])    # Red
            color_2 = np.array([0.0, 1.0, 0.0])    # Green
            color_3 = np.array([0.0, 0.0, 1.0])    # Blue


            colors = [color_1, color_2, color_3]
            # Overlay the masks on raw_j with opacity
            bool_mask_list = []
            attn_mask_list = []
            for i, a in enumerate(actor):
                if a.data == 0.0 and pred_actor[i].data == True:
                    bool_mask_list.append(masks_j[i] > threshold)
                    attn_mask_list.append((masks_j[i] > threshold).astype('uint8').reshape((128,384)))
            for num_gt in range(len(bool_mask_list)):
                raw_j[bool_mask_list[num_gt], :3] = attn_mask_list[num_gt][bool_mask_list[num_gt]][:, np.newaxis] * colors[0] * alpha_1 + raw_j[bool_mask_list[num_gt], :3] * (1 - alpha_1)


            plt.imshow(raw_j, cmap='gist_rainbow')
            plt.axis('off')

            img_path = os.path.join(path,'frame'+str(j) +'.jpg') 
            plt.savefig(img_path, bbox_inches='tight', pad_inches=0.0)
            plt.close()




torch.cuda.empty_cache()
args, logdir = get_eval_parser()
print(args)

class Engine(object):
    """Engine that runs training and inference.
    Args
        - cur_epoch (int): Current epoch.
        - print_every (int): How frequently (# batches) to print loss.
        - validate_every (int): How frequently (# epochs) to run validation.
        
    """

    def __init__(self, args, cur_epoch=0):
        self.cur_epoch = cur_epoch
        self.args = args

    def validate(self, model, dataloader, epoch):
        model.eval()

        t_confuse_sample, t_confuse_both_sample, t_confuse_pred, t_confuse_both_pred, t_confuse_both_miss, t_confuse_far_both_sample, t_confuse_far_both_miss = 0, 0, 0, 0, 0, 0, 0

        with torch.no_grad():   
            num_batches = 0
            for batch_num, data in enumerate(tqdm(dataloader)):
                map = data['map'][0]
                id = data['id'][0]
                v = data['variants'][0]
                video_in = data['videos']
                raw = data['raw']

                inputs = []

                for i in range(seq_len):
                    inputs.append(video_in[i].to(args.device, dtype=torch.float32))
                batch_size = inputs[0].shape[0]
                ego = data['ego'].to(args.device)
                if ('slot' in args.model_name and not args.allocated_slot) or args.box:
                    actor = data['actor'].to(args.device)
                else:
                    actor = torch.FloatTensor(data['actor']).to(args.device)


                pred_ego, pred_actor, attn = model(inputs)
                if args.plot and args.plot_mode != '':
                    plot_slot(attn, args.model_name, map, id, v, raw, actor, pred_actor, logdir, args.plot_threshold, args.plot_mode)


  
torch.cuda.empty_cache() 
seq_len = args.seq_len
num_ego_class = 4
num_actor_class = 64

# Data
val_set = taco_fp.TACO_FP(args=args, training=False)
dataloader_val = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

model = generate_model(args, num_ego_class, num_actor_class).cuda()
trainer = Engine(args)

model_path = os.path.join(args.cp)
model.load_state_dict(torch.load(model_path))

trainer.validate(model, dataloader_val, None)
