import argparse
import json
import os
import sys

import cv2
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from sklearn.metrics import average_precision_score, precision_score, recall_score, accuracy_score, hamming_loss
from PIL import Image, ImageDraw
from matplotlib.patches import Polygon
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import matplotlib.image


import oats
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
class_map=[
        'C+:Z1-Z3', 'P+:C1-C2', 'P:C2-C1', 'C:Z3-Z1', 'C:Z2-Z1',
        'P:C4-C3', 'C:Z1-Z3', 'C:Z1-Z2', 'C:Z2-Z4', 'C:Z4-Z2',
        'C:Z3-Z4', 'P:C2-C3', 'C:Z4-Z1', 'P:C3-C4', 'C:Z1-Z4',
        'C:Z2-Z3', 'P:C1-C2', 'P+:C2-C3', 'P:C3-C2', 'P+:C3-C4',
        'P+:C4-C3', 'P:C1-C4', 'P+:C3-C2', 'C:Z3-Z2', 'K:Z3-Z1',
        'C:Z4-Z3', 'P+:C1-C4', 'C+:Z4-Z2', 'P+:C4-C1', 'P+:C2-C1',
        'C+:Z3-Z1', 'P:C4-C1', 'K:Z2-Z4', 'C+:Z2-Z4', 'K:Z1-Z3'
        ]
def plot_slot(attn, model_name, scenario, raw, actor, pred_actor, logdir, threshold, mode):
    
    
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
                actor_str += class_map[i]
                if pred_actor[i].data == True:
                    actor_str += '  TP'
                    num_tp +=1
                else:
                    actor_str += '          FN'
            
            else:
                if pred_actor[i].data == True:
                    actor_str += class_map[i] 
                    actor_str += '                  FP'
                else:
                    actor_str += class_map[i] 
                    actor_str += '                          TN'
            actor_str +='\n'
        if num_pos != num_tp and model_name =='action_slot':
            return
        path = os.path.join(logdir, 'plot_'+ mode +'_'+str(threshold))
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, str(scenario))
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, "label_result.txt"), "w") as text_file:
            text_file.write(actor_str)


    raw = torch.stack(raw, dim=0)
    # 16, 1, 3, h, w
    # raw = torch.permute(raw, (1,2,0,3,4))
    seq_len = 16
    attn = attn.detach()
    m_l, m_n, m_h, m_w = attn.shape[1], attn.shape[2], attn.shape[3], attn.shape[4]
    attn = torch.reshape(attn, (-1, 1, m_h, m_w))
    attn = F.interpolate(attn, (1200,1920), mode='bilinear')
    attn = torch.reshape(attn, (1, m_l, m_n, 1200, 1920))
    attn = attn.permute(0, 1, 2, 4, 3)

    raw = raw.permute(1, 0, 2, 3, 4)
    cur_raw = F.interpolate(raw, (3, 1200,1920))

    attn = attn[0]
    # cur_image = cur_image[0]
    cur_raw = cur_raw[0]
    
    for j in range(seq_len):
        # image_j = cur_image[j].permute(1,2,0).cpu().numpy()

        raw_j = cur_raw[j].permute(2,1,0).cpu().numpy()
        # image_j = image_j * 0.5 + 0.5
        masks_j = attn[j]
        tk = args.num_slots
        if args.bg_slot:
            tk += 1
        masks_j = masks_j.cpu().numpy()
        
        if mode == 'multi':
            alpha_1 = 0.2
            alpha_2 = 0.2
            alpha_3 = 0.2

            color_1 = np.array([1.0, 0.0, 0.0])    # Red
            color_2 = np.array([0.0, 1.0, 0.0])    # Green
            color_3 = np.array([0.0, 0.0, 1.0])    # Blue
            color_4 = np.array([1.0, 1.0, 0.0])    # Yellow
            color_5 = np.array([1.0, 0.0, 1.0])    # Magenta
            color_6 = np.array([0.5, 0.5, 0.0])   # Olive
            color_7 = np.array([0.0, 1.0, 1.0])    # Cyan
            color_8 = np.array([1.0, 0.5, 0.0])   # Orange

            color_9 = np.array([0.2, 0.5, 1.0])   # Steel Blue
            color_10 = np.array([0.5, 0.0, 0.5])   # Purple
            colors = [color_1, color_2, color_3, color_4, color_5, color_6, color_7, color_8, color_9, color_10]
            # Overlay the masks on raw_j with opacity
            bool_mask_list = []
            attn_mask_list = []
            for i, a in enumerate(actor):
                if a.data == 1.0:
                    bool_mask_list.append(masks_j[i] > threshold)
                    attn_mask_list.append((masks_j[i] > threshold).astype('uint8'))

            for num_gt in range(len(bool_mask_list)):
                raw_j[bool_mask_list[num_gt], :3] = attn_mask_list[num_gt][bool_mask_list[num_gt]][:, np.newaxis] * colors[num_gt] * alpha_1 + raw_j[bool_mask_list[num_gt], :3] * (1 - alpha_1)
            raw_j = np.transpose(raw_j, (1, 0, 2))

            plt.imshow(raw_j, cmap='gist_rainbow')
            plt.axis('off')

            img_path = os.path.join(path,'frame'+str(j) +'.png') 
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
            total_actor = 0

            correct_actor = 0
            label_actor_list = []
            map_pred_actor_list = []

            for batch_num, data in enumerate(tqdm(dataloader)):
                scenario = data['scenario'][0]
                video_in = data['videos']
                raw = data['raw']

                if args.box:
                    box_in = data['box']
                inputs = []

                for i in range(seq_len):
                    inputs.append(video_in[i].to(args.device, dtype=torch.float32))
     
                if args.box:
                    if isinstance(box_in,np.ndarray):
                        boxes = torch.from_numpy(box_in).to(args.device, dtype=torch.float32)
                    else:
                        boxes = box_in.to(args.device, dtype=torch.float32)
                
                batch_size = inputs[0].shape[0]
                if ('slot' in args.model_name and not args.allocated_slot) or args.box:
                    actor = data['actor'].to(args.device)
                else:
                    actor = torch.FloatTensor(data['actor']).to(args.device)


                if ('slot' in args.model_name) or args.box or 'mvit' in args.model_name:
                    if args.box:
                        _, pred_actor = model(inputs, boxes)
                    else:
                        pred_actor, attn = model(inputs)
                        if args.plot:
                            channel_idx = [-1]
                            if ('mvit' in args.model_name):
                                for j,(attn,thw) in enumerate(attn):
                                    for c_idx in channel_idx:
                                        plot_mvit(attn[0], c_idx, raw, logdir , id, v, j, grid_size=(thw[1],thw[2]))
                            else:
                                plot_slot(attn, args.model_name, scenario, raw, actor, pred_actor, logdir, args.plot_threshold, args.plot_mode)

                else:
                    pred_actor = model(inputs)

                num_batches += 1
                if ('slot' in args.model_name and not args.allocated_slot) or args.box:
                    pred_actor = torch.nn.functional.softmax(pred_actor, dim=-1)
                    _, pred_actor_idx = torch.max(pred_actor.data, -1)
                    pred_actor_idx = pred_actor_idx.detach().cpu().numpy().astype(int)
                    map_batch_new_pred_actor = []
                    for i, b in enumerate(pred_actor_idx):
                        map_new_pred = np.zeros(num_actor_class, dtype=np.float32)+1e-5

                        for j, pred in enumerate(b):
                            if pred != num_actor_class:
                                if pred_actor[i, j, pred] > map_new_pred[pred]:
                                    map_new_pred[pred] = pred_actor[i, j, pred]
                        map_batch_new_pred_actor.append(map_new_pred)
                    map_batch_new_pred_actor = np.array(map_batch_new_pred_actor)
                    map_pred_actor_list.append(map_batch_new_pred_actor)
                    label_actor_list.append(data['slot_eval_gt'])
                else:
                    pred_actor = torch.sigmoid(pred_actor)
                    map_pred_actor_list.append(pred_actor.detach().cpu().numpy())
                    label_actor_list.append(actor.detach().cpu().numpy())

            map_pred_actor_list = np.stack(map_pred_actor_list, axis=0)
            label_actor_list = np.stack(label_actor_list, axis=0)
            
            map_pred_actor_list = map_pred_actor_list.reshape((map_pred_actor_list.shape[0], num_actor_class))
            label_actor_list = label_actor_list.reshape((label_actor_list.shape[0], num_actor_class))
            map_pred_actor_list = np.array(map_pred_actor_list)
            label_actor_list = np.array(label_actor_list)
            
            # index
            mAP = average_precision_score(
                    label_actor_list,
                    map_pred_actor_list.astype(np.float32),
                    )


            print(f'(val) mAP of the actor: {mAP}')


            
torch.cuda.empty_cache() 
seq_len = args.seq_len
num_ego_class = 0
num_actor_class = 35

# Data
val_set = oats.OATS(args=args, training=False)
dataloader_val = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

model = generate_model(args, num_ego_class, num_actor_class).cuda()
trainer = Engine(args)
# model.load_state_dict(torch.load(os.path.join(args.logdir, 'model_100.pth')))

model_path = os.path.join(args.cp)
model.load_state_dict(torch.load(model_path))

trainer.validate(model, dataloader_val, None)
