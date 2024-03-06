import argparse
import json
import os
import sys


import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import cv2
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from tqdm import tqdm
from sklearn.metrics import average_precision_score, precision_score, recall_score, accuracy_score, hamming_loss
from torchvision import models
import matplotlib.image
from scipy.optimize import linear_sum_assignment
from PIL import Image, ImageDraw

import nuscenes
from model import generate_model
from utils import *
from parser_eval import get_eval_parser

torch.backends.cudnn.benchmark = True

sys.path.append('../datasets')
sys.path.append('../configs')
sys.path.append('../models')

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

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
                      
actor_table = [ 'c12', 'c13', 'c14',
                'c21', 'c23', 'c24',
                'c31', 'c32', 'c34',
                'c41', 'c42', 'c43',
                'c+12', 'c+13', 'c+14',
                'c+21', 'c+23', 'c+24',
                'c+31', 'c+32', 'c+34',
                'c+41', 'c+42', 'c+43',
                'k12', 'k13', 'k14',
                'k21', 'k23', 'k24',
                'k31', 'k32', 'k34',
                'k41', 'k42', 'k43',
                'k+12', 'k+13', 'k+14',
                'k+21', 'k+23', 'k+24',
                'k+31', 'k+32', 'k+34',
                'k+41', 'k+42', 'k+43',
                'p12', 'p14', 
                'p21', 'p23', 
                'p32', 'p34', 
                'p41', 'p43',
                'p+12', 'p+14', 
                'p+21', 'p+23', 
                'p+32', 'p+34',
                'p+41', 'p+43'
                ]
def plot_slot(attn, model_name, city, scenario, raw, actor, pred_actor, logdir, threshold, mode):


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
        # if num_pos > num_tp*2 and model_name == 'action_slot':
        #     return

        path = os.path.join(logdir, 'plot_'+ mode +'_'+str(threshold))
        if not os.path.exists(path):
            os.makedirs(path)
        
        path = os.path.join(path, city+'_'+scenario)
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
    attn = F.interpolate(attn, (900,1600), mode='bilinear')
    attn = torch.reshape(attn, (1, m_l, m_n, 900, 1600))

    raw = raw.permute(0, 2, 1, 3, 4)
    cur_raw = F.interpolate(raw, (3, 900,1600))
    attn = attn[0]
    cur_raw = cur_raw[0]
    
    for j in range(seq_len):
        raw_j = cur_raw[j].permute(1,2,0).cpu().numpy()
        new_raw_j = raw_j * 0.8 + 0.1
        masks_j = attn[j]
        tk = args.num_slots
        if args.bg_slot:
            tk += 1
        masks_j = masks_j.cpu().numpy()
        
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
                attn_mask_list.append((masks_j[i] > threshold).astype('uint8').reshape((900,1600)))

        for num_gt in range(len(bool_mask_list)):
            raw_j[bool_mask_list[num_gt], :3] = attn_mask_list[num_gt][bool_mask_list[num_gt]][:, np.newaxis] * colors[num_gt] * alpha_1 + raw_j[bool_mask_list[num_gt], :3] * (1 - alpha_1)


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
            total_ego = 0
            total_actor = 0

            correct_ego = 0
            correct_actor = 0
            label_actor_list = []
            map_pred_actor_list = []

            num_selected_sample = 0
            for batch_num, data in enumerate(tqdm(dataloader)):

                city = data['city'][0].split('_')[1]
                scenario = data['scenario'][0]
                video_in = data['videos']
                raw = data['raw']
                scenario = city + '_'+scenario

                # print(city)
                if args.nuscenes_test_split == 'boston' and city != 'boston':
                    continue
                elif args.nuscenes_test_split == 'singapore' and city != 'singapore':
                    continue

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
                ego = data['ego'].to(args.device)
                if ('slot' in args.model_name and not args.allocated_slot) or args.box:
                    actor = data['actor'].to(args.device)
                else:
                    actor = torch.FloatTensor(data['actor']).to(args.device)


                if ('slot' in args.model_name) or args.box or 'mvit' in args.model_name:
                    if args.box:
                        pred_ego, pred_actor = model(inputs, boxes)
                    else:
                        pred_ego, pred_actor, attn = model(inputs)

                        # if args.plot_mode == 'mask':
                        #     plot_mask(seg_front, args.id, id, v, logdir)
                        # elif args.plot:

                        if args.plot and args.plot_mode != '':
                            channel_idx = [-1]
                            if ('mvit' in args.model_name):
                                for j,(attn,thw) in enumerate(attn):
                                    # attn = attn[0].mean(0)
                                    # print(attn.shape)
                                    # print(thw)
                                    for c_idx in channel_idx:
                                        plot_mvit(attn[0], c_idx, raw, logdir , id, v, j, grid_size=(thw[1],thw[2]))
                                # raise BaseException
                            else:
                                plot_slot(attn, args.model_name, city, scenario, raw, actor, pred_actor, logdir, args.plot_threshold, args.plot_mode)

                else:

                    pred_ego, pred_actor = model(inputs)


                num_batches += 1
                pred_ego = torch.nn.functional.softmax(pred_ego, dim=1)
                _, pred_ego = torch.max(pred_ego.data, 1)

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
                    if args.dataset == 'nuscenes' and args.pretrain == 'oats' and not 'nuscenes'in args.cp:
                        dummy_classes = np.zeros((1,29), dtype=float)
                        map_pred_actor_list.append(
                            np.concatenate(
                                (pred_actor.detach().cpu().numpy(), dummy_classes), axis=1)
                            )
                    else:
                        map_pred_actor_list.append(pred_actor.detach().cpu().numpy())
                    label_actor_list.append(actor.detach().cpu().numpy())

                if args.val_confusion:
                    confuse_sample, confuse_both_sample, confuse_pred, confuse_both_pred, confuse_both_miss, confuse_far_both_sample, confuse_far_both_miss= calculate_confusion(confusion_label, f1_pred_actor)
                    t_confuse_sample = t_confuse_sample + confuse_sample
                    t_confuse_both_sample = t_confuse_both_sample + confuse_both_sample
                    t_confuse_pred = t_confuse_pred + confuse_pred
                    t_confuse_both_pred = t_confuse_both_pred + confuse_both_pred
                    t_confuse_both_miss = t_confuse_both_miss + confuse_both_miss
                    t_confuse_far_both_sample = t_confuse_far_both_sample + confuse_far_both_sample
                    t_confuse_far_both_miss = t_confuse_far_both_miss + confuse_far_both_miss
                total_ego += ego.size(0)
                correct_ego += (pred_ego == ego).sum().item()



            map_pred_actor_list = np.stack(map_pred_actor_list, axis=0)
            label_actor_list = np.stack(label_actor_list, axis=0)
            
            map_pred_actor_list = map_pred_actor_list.reshape((map_pred_actor_list.shape[0], num_actor_class))
            label_actor_list = label_actor_list.reshape((label_actor_list.shape[0], num_actor_class))
            map_pred_actor_list = np.array(map_pred_actor_list)
            label_actor_list = np.array(label_actor_list)
            

            mAP = average_precision_score(
                    np.concatenate([label_actor_list[:, :36], label_actor_list[:, -16:]], axis=1),
                    np.concatenate([map_pred_actor_list[:, :36], map_pred_actor_list[:, -16:]], axis=1).astype(np.float32),
                    )
            # c_mAP = average_precision_score(
            #         label_actor_list[:, :12],
            #         map_pred_actor_list[:, :12].astype(np.float32)
            #         )
            # b_mAP = average_precision_score(
            #         label_actor_list[:, 12:24],
            #         map_pred_actor_list[:, 12:24].astype(np.float32)
            #         )
            # p_mAP = average_precision_score(
            #         label_actor_list[:, 48:56],
            #         map_pred_actor_list[:, 48:56].astype(np.float32),
            #         )
            # group_c_mAP = average_precision_score(
            #         label_actor_list[:, 24:36],
            #         map_pred_actor_list[:, 24:36].astype(np.float32)
            #         )
            # group_b_mAP = average_precision_score(
            #         label_actor_list[:, 36:48],
            #         map_pred_actor_list[:, 36:48].astype(np.float32)
            #         )
            # group_p_mAP = average_precision_score(
            #         label_actor_list[:, 56:64],
            #         map_pred_actor_list[:, 56:64].astype(np.float32),
            #         )
            mAP_per_class = average_precision_score(
                    label_actor_list,
                    map_pred_actor_list.astype(np.float32), 
                    average=None)


            print(f'(val) mAP of the actor: {mAP}')
            # print(f'(val) mAP of the c: {c_mAP}')
            # print(f'(val) mAP of the b: {b_mAP}')
            # print(f'(val) mAP of the p: {p_mAP}')
            # print(f'(val) mAP of the c+: {group_c_mAP}')
            # print(f'(val) mAP of the b+: {group_b_mAP}')
            # print(f'(val) mAP of the p+: {group_p_mAP}')

            # print(f'acc of the ego: {correct_ego/total_ego}')
            # print('**********************')
            # print(num_selected_sample)

            
torch.cuda.empty_cache() 
seq_len = args.seq_len
num_ego_class = 4
num_actor_class = 64


# Data
val_set = nuscenes.NUSCENES(args=args, training=False)
# label_stat = []
# for i in range(7):
#     label_stat.append({})
#     for k in train_set.label_stat[i].keys():
#         label_stat[i][k] = train_set.label_stat[i][k] + val_set.label_stat[i][k]

# Data
val_set = nuscenes.NUSCENES(args=args, training=False)
dataloader_val = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

model = generate_model(args, num_ego_class, num_actor_class).cuda()
trainer = Engine(args)
# model.load_state_dict(torch.load(os.path.join(args.logdir, 'model_100.pth')))

model_path = os.path.join(args.cp)

if not 'nuscenes' in args.cp and args.pretrain == 'oats':
    checkpoint = torch.load(model_path)
    checkpoint = {k: v for k, v in checkpoint.items() if (k in checkpoint and 'ego' not in k)}
    model.load_state_dict(checkpoint, strict=False)
    if 'slot' in args.model_name:
        model.slot_attention.extend_slots()
else:
    model.load_state_dict(torch.load(model_path))



trainer.validate(model, dataloader_val, None)
