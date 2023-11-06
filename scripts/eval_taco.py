import torch.nn as nn
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import cv2
import torch.nn.functional as F
import os


import argparse
import json
import os
from tqdm import tqdm

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn

torch.backends.cudnn.benchmark = True

import sys
sys.path.append('/media/hankung/ssd/Action-Slot/datasets')
sys.path.append('/media/hankung/ssd/Action-Slot/config')
sys.path.append('/media/hankung/ssd/Action-Slot/models')

sys.path.append('/work/u8526971/retrieval/datasets')
sys.path.append('/work/u8526971/retrieval/config')
sys.path.append('/work/u8526971/retrieval/models')


sys.path.append('/media/hcis-s19/DATA/Action-Slot/datasets')
sys.path.append('/media/hcis-s19/DATA/Action-Slot/config')
sys.path.append('/media/hcis-s19/DATA/Action-Slot/models')

sys.path.append('/media/hcis-s20/SRL/Action-Slot/datasets')
sys.path.append('/media/hcis-s20/SRL/Action-Slot/configs')
sys.path.append('/media/hcis-s20/SRL/Action-Slot/models')

sys.path.append('/media/hcis-s16/hank/Action-Slot/datasets')
sys.path.append('/media/hcis-s16/hank/Action-Slot/configs')
sys.path.append('/media/hcis-s16/hank/Action-Slot/models')

sys.path.append('/media/user/data/Action-Slot/datasets')
sys.path.append('/media/user/data/Action-Slot/configs')
sys.path.append('/media/user/data/Action-Slot/models')

# from .configs.config import GlobalConfig
import taco

from sklearn.metrics import average_precision_score, precision_score, recall_score, accuracy_score, hamming_loss

from PIL import Image, ImageDraw

from model import generate_model
from utils import *
from torchvision import models
import matplotlib.image
from scipy.optimize import linear_sum_assignment
from parser import get_parser
# matplotlib.use('TkAgg')


# actor_table = ['z1-z2', 'z1-z3', 'z1-z4',
#                 'z2-z1', 'z2-z3', 'z2-z4',
#                 'z3-z1', 'z3-z2', 'z3-z4',
#                 'z4-z1', 'z4-z2', 'z4-z3',
#                 'c1-c2', 'c1-c4', 
#                 'c2-c1', 'c2-c3', 
#                 'c3-c2', 'c3-c4', 
#                 'c4-c1', 'c4-c3', 'bg']
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
                        
def plot_slot(attn, model_name, id, v, raw, actor, pred_actor, logdir, threshold, mode):
    path = os.path.join(logdir, 'plot_'+ mode +'_'+str(threshold))
    if not os.path.exists(path):
        os.makedirs(path)
    
    path = os.path.join(path, id + '_' + v)
    if not os.path.exists(path):
        os.makedirs(path)

    actor_str = ''
    actor = actor[0]
    pred_actor = pred_actor[0]

    pred_actor = torch.sigmoid(pred_actor)
    pred_actor = pred_actor > 0.5
    if args.allocated_slot:
        for i, a in enumerate(actor):
            if a.data == 1.0:
                actor_str += actor_table[i]
                if pred_actor[i].data == True:
                    actor_str += '  TP'
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
        with open(os.path.join(path, "label_result.txt"), "w") as text_file:
            text_file.write(actor_str)

    cmap = plt.get_cmap('rainbow')
    colors = [cmap(ii) for ii in np.linspace(0, 1, 20)]

    # image = torch.stack(image, dim=0) #[v, b, 2048, h, w]
    # l, b, c, h, w
    # image = torch.permute(image, (1,2,0,3,4)) #[b, v, 2048, h, w]
    # raw = [raw[i] for i in range(0, 13, 2)]
    raw = torch.stack(raw, dim=0)
    raw = torch.permute(raw, (1,2,0,3,4))
    seq_len = 16
    attn = attn.detach()
    m_l, m_n, m_h, m_w = attn.shape[1], attn.shape[2], attn.shape[3], attn.shape[4]
    attn = torch.reshape(attn, (-1, 1, m_h, m_w))
    # masks = F.interpolate(masks, (masks.shape[-3], 128,384))
    attn = F.interpolate(attn, (128,384), mode='bilinear')
    attn = torch.reshape(attn, (1, m_l, m_n, 128, 384))
    # index_mask = masks.argmax(dim = 2)
    # index_mask = F.one_hot(index_mask,num_classes = 20)
    # index_mask = index_mask.permute(0,1,4,2,3)
    # masks = masks * index_mask

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
        tk = args.num_slots
        if args.bg_slot:
            tk += 1
        masks_j = masks_j.cpu().numpy()
        # if mode != 'multi':
        #     for slot_idx in range(tk):
        #         if mode == 'both':
        #             masks_j[slot_idx] = masks_j[slot_idx] * (masks_j[slot_idx] > threshold).astype('uint8')
        #             plt.figure(figsize=(15, 5))
        #             plt.imshow(new_raw_j)
        #             plt.imshow(masks_j[slot_idx], alpha=0.8)
        #             plt.axis('off')
        #             plt.show()

        #             seg_path = os.path.join(path, actor_table[slot_idx]+ '_frame'+str(j)+'.jpg')
        #             # fig.savefig(path)
        #             plt.savefig(seg_path)
        #             plt.close()

        #             plt.figure(figsize=(15, 5))
        #             plt.imshow(raw_j)
        #             plt.axis('off')
        #             plt.show()
        #             img_path = os.path.join(path, actor_table[slot_idx] + '_frame'+str(j)+ '_img' +'.jpg') 
        #             plt.savefig(img_path)
        #             plt.close()

        #             im1 = Image.open(img_path)
        #             im2 = Image.open(seg_path)
        #             dst = Image.new('RGB', (im1.width, im1.height + im2.height))
        #             dst.paste(im1, (0, 0))
        #             dst.paste(im2, (0, im1.height))
        #             dst.save(seg_path)
        #             os.remove(img_path)


        #         elif mode == 'attn':
        #             masks_j[slot_idx] = masks_j[slot_idx] * (masks_j[slot_idx] > threshold).astype('uint8')
        #             # plt.figure(figsize=(15, 5))
        #             # plt.imshow(new_raw_j)
        #             # plt.imshow(masks_j[seg], alpha=0.5)
        #             # plt.axis('off')
        #             # plt.show()

        #             # seg_path = os.path.join(path, actor_table[seg]+ '_frame'+str(j)+'.png')
        #             # # fig.savefig(path)
        #             # plt.savefig(seg_path)
        #             # plt.close()

        #             plt.imshow(new_raw_j)
        #             plt.axis('off')
        #             # plt.imshow()
        #             plt.imshow(masks_j[slot_idx], alpha=0.5)
        #             # plt.show()
        #             seg_path = os.path.join(path, actor_table[slot_idx]+ '_frame'+str(j)+'.jpg')
        #             plt.savefig(seg_path, bbox_inches='tight', pad_inches=0.0)
        #             plt.close()

        #     if mode == 'rgb':
        #         # plt.figure(figsize=(15, 5))
        #         plt.imshow(raw_j)
        #         plt.axis('off')
        #         # plt.show()
        #         img_path = os.path.join(path,'frame'+str(j)+ '_img' +'.jpg') 
        #         plt.savefig(img_path, bbox_inches='tight', pad_inches=0.0)
        #         plt.close()

        if mode == 'bg':
            t_mask_bg = (masks_j[-1] > threshold)
            masks_bg = (masks_j[-1] > threshold).astype('uint8').reshape((128,384))
            raw_j[t_mask_bg, 0] = masks_bg[t_mask_bg]
            plt.imshow(raw_j, cmap='gist_rainbow')
            plt.axis('off')
            img_path = os.path.join(path,'frame'+str(j)+ '_bg' +'.jpg') 
            plt.savefig(img_path, bbox_inches='tight', pad_inches=0.0)
            plt.close()
        else:
            # plt.imshow(raw_j)
            # plt.axis('off')
            t_masks_3 = (masks_j[0] > threshold)
            t_masks_6 = (masks_j[18] > threshold)
            # t_masks_b = (masks_j[-1] > threshold)
            # t = t_masks_3+t_masks_6+t_masks_b
            t = t_masks_3+t_masks_6

            masks_3 = (masks_j[0] > threshold).astype('uint8').reshape((128,384))
            masks_6 = (masks_j[18] > threshold).astype('uint8').reshape((128,384))
            # masks_b = (masks_j[-1] > threshold).astype('uint8').reshape((128,384))
            # print(masks_3.shape)
            # print(t_masks_3.shape)
            # print(raw_j.shape)
            raw_j[t_masks_3, 0] = masks_3[t_masks_3]
            raw_j[t_masks_6, 1] = masks_6[t_masks_6]
            # raw_j[t_masks_b, 2] = masks_b[t_masks_b]
            plt.imshow(raw_j, cmap='gist_rainbow')
            plt.axis('off')
            # plt.imshow(masks_j[3], alpha=0.5, cmap='Oranges')

            # masks_j[6] = (masks_j[6] > threshold).astype('uint8')
            # plt.imshow(masks_j[6], alpha=0.5, cmap='Blues')

            # masks_j[-1] = (masks_j[-1] > threshold).astype('uint8')
            # plt.imshow(masks_j[-1], alpha=0.5, cmap='Purples')

            img_path = os.path.join(path,'frame'+str(j)+ '_multi' +'.jpg') 
            plt.savefig(img_path, bbox_inches='tight', pad_inches=0.0)
            plt.close()

def plot_mvit(att_map, grid_index, raw,logdir,id,v, head_idx,grid_size=14, alpha=0.6,threshold=0.5):
    path = os.path.join(logdir, id + '_' + v)
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path,str(head_idx))
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path,str(grid_index))
    if not os.path.exists(path):
        os.makedirs(path)

    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
    # att_map = att_map.reshape(8,grid_size[0]*grid_size[1],-1)
    att_map = att_map.reshape(8,grid_size[0],grid_size[1],-1)[None]
    att_map = att_map.permute(0,4,1,2,3)
    # interpolate
    att_map = F.interpolate(att_map,(16,grid_size[0],grid_size[1]),mode='trilinear')[0]
    att_map = att_map.permute(1,2,3,0)
    
    raw = torch.stack(raw, dim=0)
    raw = raw.permute(1,0,2,3,4)[0]
    for t in range(16):
        att_map_j = att_map[t]
        raw_j = raw[t].permute(1,2,0).cpu().numpy()
        # image_j = image_j * 0.5 + 0.5
        new_raw_j = raw_j * 0.8 + 0.1
        image = Image.fromarray((new_raw_j * 255).astype(np.uint8))
        
        # H,W = att_map_j.shape
        H,W,_ = att_map_j.shape
        with_cls_token = False
        # grid_image = highlight_grid(image, [grid_index], grid_size)
        grid_image = image
        # mask = att_map_j[grid_index].reshape(8,grid_size[0], grid_size[1])[t//2].cpu().numpy()
        # mask = Image.fromarray(mask).resize((image.size))
        if grid_index == -1:
            mask = att_map_j[:,:].mean(-1).sigmoid().cpu().numpy()
        else:
            mask = att_map_j[:,:,grid_index].sigmoid().cpu().numpy()
        mask = mask/np.max(mask)
        mask = mask * (mask>threshold)
        mask = Image.fromarray(mask).resize((image.size))
        # mask = mask/np.max(mask)
        # mask = mask * (mask>threshold)
        fig, ax = plt.subplots(1, 2, figsize=(10,7))
        fig.tight_layout()
        
        ax[0].imshow(grid_image)
        ax[0].axis('off')
        
        ax[1].imshow(grid_image)
        ax[1].imshow(mask/np.max(mask), alpha=alpha, cmap='gist_rainbow')
        ax[1].axis('off')
        img_path = os.path.join(path,'frame'+str(t+1) +'.png') 
        plt.savefig(img_path, bbox_inches='tight', pad_inches=0.0)
        plt.close()
    
def highlight_grid(image, grid_indexes, grid_size=14):
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)
    W, H = image.size
    h = H / grid_size[0]
    w = W / grid_size[1]
    image = image.copy()
    for grid_index in grid_indexes:
        x, y = np.unravel_index(grid_index, (grid_size[0], grid_size[1]))
        a= ImageDraw.ImageDraw(image)
        a.rectangle([(y*w,x*h),(y*w+w,x*h+h)],fill =None,outline ='red',width =2)
    return image

def plot_mask(masks, model_name, id, v, logdir):
    path = os.path.join(logdir, 'plot_mask')
    if not os.path.exists(path):
        os.makedirs(path)
    
    path = os.path.join(path, id + '_' + v)
    if not os.path.exists(path):
        os.makedirs(path)

    seq_len = 16
    masks = masks.detach()
    bl, m_h, m_w = masks.shape[0], masks.shape[2], masks.shape[3]

    # masks = torch.reshape(masks, (-1, 1, m_h, m_w))
    # masks = F.interpolate(masks, (masks.shape[-3], 128,384))
    masks = F.interpolate(masks, (128,384), mode='bilinear')
    masks = torch.reshape(masks, (1, 16, 128, 384))
    
    
    masks = masks[0]
    for j in range(seq_len):
        masks_j = masks[j]
        masks_j = masks_j.cpu().numpy()

        plt.imshow(masks_j,cmap='binary')
        plt.axis('off')
        plt.show()
        seg_path = os.path.join(path, '_frame'+str(j)+'.png')
        plt.savefig(seg_path, bbox_inches='tight', pad_inches=0.0)
        plt.close()

def calculate_confusion(confusion_label, pred):

    actor_table = { 'z1-z2': 0, 'z1-z3':1, 'z1-z4':2,
                                'z2-z1': 3, 'z2-z3': 4, 'z2-z4': 5,
                                'z3-z1': 6, 'z3-z2': 7, 'z3-z4': 8,
                                'z4-z1': 9, 'z4-z2': 10, 'z4-z3': 11,

                                'c1-c2': 12, 'c1-c4': 13, 
                                'c2-c1': 14, 'c2-c3': 15, 
                                'c3-c2': 16, 'c3-c4': 17, 
                                'c4-c1': 18, 'c4-c3': 19 }
    confuse_sample = 0
    confuse_both_sample = 0
    confuse_far_both_sample = 0
    confuse_pred = 0
    confuse_both_pred = 0
    confuse_both_miss = 0
    confuse_far_both_miss = 0
    pred = pred[0]
    if confusion_label['c1-c2']==0:
        confuse_sample +=1
        if pred[12]==0. and pred[14]==1.:
            confuse_pred+=1
        elif pred[12]==1. and pred[14]==1.:
            confuse_both_pred +=1
    elif confusion_label['c1-c2']==1:
        confuse_sample +=1
        if pred[14]==0. and pred[12]==1.:
            confuse_pred+=1
        elif pred[14]==1. and pred[12]==1.:
            confuse_both_pred +=1
    elif confusion_label['c1-c2']==2:
        confuse_both_sample +=1
        if not (pred[14]==1. and pred[12]==1.):
            confuse_both_miss+=1
    # -----
    if confusion_label['c2-c3']==0:
        confuse_sample +=1
        if pred[15]==0. and pred[16]==1.:
            confuse_pred+=1
        elif pred[15]==1. and pred[16]==1.:
            confuse_both_pred +=1
    elif confusion_label['c2-c3']==1:
        confuse_sample +=1
        if pred[16]==0. and pred[15]==1.:
            confuse_pred+=1
        elif pred[16]==1. and pred[15]==1.:
            confuse_both_pred +=1
    elif confusion_label['c2-c3']==2:
        confuse_both_sample +=1
        confuse_far_both_sample +=1
        if not (pred[15]==1. and pred[16]==1.):
            confuse_both_miss+=1
            confuse_far_both_miss +=1
    # -----
    if confusion_label['c3-c4']==0:
        confuse_sample +=1
        if pred[17]==0. and pred[19]==1.:
            confuse_pred+=1
        elif pred[17]==1. and pred[19]==1.:
            confuse_both_pred +=1
    elif confusion_label['c3-c4']==1:
        confuse_sample +=1
        if pred[19]==0. and pred[17]==1.:
            confuse_pred+=1
        elif pred[19]==1. and pred[17]==1.:
            confuse_both_pred +=1
    elif confusion_label['c3-c4']==2:
        confuse_both_sample +=1
        if not (pred[17]==1. and pred[19]==1.):
            confuse_both_miss+=1
    # -----
    if confusion_label['c4-c1']==0:
        confuse_sample +=1
        if pred[18]==0. and pred[13]==1.:
            confuse_pred+=1
        elif pred[18]==1. and pred[13]==1.:
            confuse_both_pred +=1
    elif confusion_label['c4-c1']==1:
        confuse_sample +=1
        if pred[13]==0. and pred[18]==1.:
            confuse_pred+=1
        elif pred[13]==1. and pred[18]==1.:
            confuse_both_pred +=1
    elif confusion_label['c4-c1']==2:
        confuse_both_sample +=1
        if not (pred[13]==1. and pred[18]==1.):
            confuse_both_miss+=1

    return confuse_sample, confuse_both_sample, confuse_pred, confuse_both_pred, confuse_both_miss, confuse_far_both_sample, confuse_far_both_miss



torch.cuda.empty_cache()
args, logdir = get_parser()
print(args)
writer = SummaryWriter(log_dir=logdir)

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
        ego_ce = nn.CrossEntropyLoss(reduction='mean').cuda()
        mask_bce = nn.BCELoss()

        t_confuse_sample, t_confuse_both_sample, t_confuse_pred, t_confuse_both_pred, t_confuse_both_miss, t_confuse_far_both_sample, t_confuse_far_both_miss = 0, 0, 0, 0, 0, 0, 0

        if ('slot' in args.model_name and not args.allocated_slot) or args.box:
            ce_weights = torch.ones(num_actor_class+1)*args.ce_pos_weight
            ce_weights[-1] = self.args.ce_neg_weight
            ce = nn.CrossEntropyLoss(reduction='mean', weight=ce_weights).cuda()
        elif 'slot' in args.model_name and args.allocated_slot:
            bce_weights = torch.ones([num_actor_class])*args.bce_pos_weight
            bce = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=bce_weights).cuda()
        else:
            bce_weights = torch.ones([num_actor_class])*args.bce_pos_weight
            bce = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=bce_weight).cuda()
        with torch.no_grad():   
            num_batches = 0
            total_loss = 0.
            loss = 0.
            attn_loss_epoch= 0.
            action_attn_loss_epoch = 0.
            bg_attn_loss_epoch = 0.
            total_ego = 0
            total_actor = 0

            correct_ego = 0
            correct_actor = 0
            label_actor_list = []
            map_pred_actor_list = []

            action_inter = AverageMeter()
            action_union = AverageMeter()
            bg_inter = AverageMeter()
            bg_union = AverageMeter()

            for batch_num, data in enumerate(tqdm(dataloader)):
                id = data['id'][0]
                v = data['variants'][0]
                video_in = data['videos']
                raw = data['raw']
                if args.val_confusion:
                    confusion_label = data['confusion_label']
                idv = id + '_' + v
                # plot_list = ['i1_901', '10_t1-7_1_p_c_r_1_0_1']
                # plot_list = ['t2_221', 't2_251']
                # plot_list = ['t2_157']
                # plot_list = ['t2_102', 't2_82', 't2_61', 't2_31']
                # plot_list = ['t6_616', 't7_172', 'i1_105', 'i1_600',
                #  'i1_391', 't1_1213', 't6_619', 't7_638', 'i1_42',
                #   't1_1192', 't1_1223', 'i1_134', 't6_414', 't7_467',
                  # '10_i-1_1_c_f_f_1_rl_6', '10_i-1_1_c_f_f_1_rl_8', 'i1_19', 'i1_151', 'i1_155','i1_901']


                if args.bg_mask:
                    bg_seg_in = data['bg_seg']
                if args.box:
                    box_in = data['box']
                inputs = []
                bg_seg = []

                for i in range(seq_len):
                    inputs.append(video_in[i].to(args.device, dtype=torch.float32))
     
                if args.box:
                    if isinstance(box_in,np.ndarray):
                        boxes = torch.from_numpy(box_in).to(args.device, dtype=torch.float32)
                    else:
                        boxes = box_in.to(args.device, dtype=torch.float32)
                
                if args.action_attn_weight>0 or args.bg_attn_weight>0:
                    ds_size = (model.resolution[0]*args.bg_upsample, model.resolution[1]*args.bg_upsample)
                if args.bg_mask:
                    for i in range(args.seq_len//args.mask_every_frame):
                        bg_seg.append(bg_seg_in[i].to(args.device, dtype=torch.float32))

                    h, w = bg_seg[0].shape[-2], bg_seg[0].shape[-1]
                    bg_seg = torch.stack(bg_seg, 0)
                    bg_seg = torch.permute(bg_seg, (1, 0, 2, 3)) #[batch, len, h, w]
                    b, l, h, w = bg_seg.shape
                    bg_seg = torch.reshape(bg_seg, (b*l, 1, h, w))
                    bg_seg = F.interpolate(bg_seg, size=ds_size)
                    bg_seg = torch.reshape(bg_seg, (b, l, ds_size[0], ds_size[1]))

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
                        if args.plot:
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
                                plot_slot(attn, args.model_name, id, v, raw, actor, pred_actor, logdir, args.plot_threshold, args.plot_mode)

                else:
                    pred_ego, pred_actor = model(inputs)

                ego_loss = ego_ce(pred_ego, ego)

                if ('slot' in args.model_name and not args.allocated_slot) or args.box:
                    bs, num_queries = pred_actor.shape[:2]
                    out_prob = pred_actor.clone().detach().flatten(0, 1).softmax(-1)
                    actor_gt_np = actor.clone().detach()
                    tgt_ids = torch.cat([v for v in actor_gt_np.detach()])
                    C = -out_prob[:, tgt_ids].clone().detach()
                    C = C.view(bs, num_queries, -1).cpu()
                    sizes = [len(v) for v in actor_gt_np]
                    indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
                    indx = [(torch.as_tensor(i, dtype=torch.int64).detach(), torch.as_tensor(j, dtype=torch.int64).detach()) for i, j in indices]
                    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indx)]).detach()
                    src_idx = torch.cat([src for (src, _) in indx]).detach()
                    idx = (batch_idx, src_idx)
                    target_classes_o = torch.cat([t[J] for t, (_, J) in zip(actor, indx)]).cuda()
                    target_classes = torch.full(pred_actor.shape[:2], num_actor_class,
                                        dtype=torch.int64, device=out_prob.device)

                    target_classes[idx] = target_classes_o
                    actor_loss = instance_ce(pred_actor.transpose(1, 2), target_classes)

                elif 'slot' in args.model_name and args.allocated_slot:
                    actor_loss = bce(pred_actor, actor)

                    # w/o L_bg, w/ L_neg
                    if not args.bg_attn_weight >0 and args.action_attn_weight >0:
                        b, l, n, h, w = attn.shape
                        if args.bg_upsample != 1:
                            attn = attn.reshape(-1, 1, h, w)
                            attn = F.interpolate(attn, size=ds_size, mode='bilinear')
                            _, _, h, w = attn.shape
                            attn = attn.reshape(b, l, n, h, w)
                        action_attn = attn[:, :, :num_actor_class, :, :]
                        class_idx = actor == 0.0
                        class_idx = class_idx.view(b, num_actor_class, 1, 1, 1).repeat(1, 1, l, h, w)
                        class_idx = torch.permute(class_idx, (0, 2, 1, 3, 4))

                        attn_gt = torch.zeros([b, l, num_actor_class, h, w], dtype=torch.float32).cuda()
                        action_attn_loss = mask_bce(action_attn[class_idx], attn_gt[class_idx])
                        attn_loss = args.action_attn_weight*action_attn_loss

                        action_attn_pred = action_attn[class_idx] > 0.5
                        inter, union = inter_and_union(action_attn_pred.reshape(-1, h, w), attn_gt[class_idx].reshape(-1, h, w), 1, 0)
                        action_inter.update(inter)
                        action_union.update(union)

                        attn_loss_epoch += float(attn_loss.item())
                        action_attn_loss_epoch += float(action_attn_loss.item())

                    # w/ L_bg, w/ L_neg
                    elif args.action_attn_weight >0. and args.bg_attn_weight>0.:
                        b, l, n, h, w = attn.shape
                        if args.bg_upsample != 1:
                            attn = attn.reshape(-1, 1, h, w)
                            attn = F.interpolate(attn, size=ds_size, mode='bilinear')
                            _, _, h, w = attn.shape
                            attn = attn.reshape(b, l, n, h, w)

                        action_attn = attn[:, :, :num_actor_class, :, :]
                        bg_attn = attn[:, ::args.mask_every_frame, -1, :, :].reshape(b, -1, h, w)

                        class_idx = actor == 0.0
                        class_idx = class_idx.view(b, num_actor_class, 1, 1, 1).repeat(1, 1, l, h, w)
                        class_idx = torch.permute(class_idx, (0, 2, 1, 3, 4))

                        attn_gt = torch.zeros([b, l, num_actor_class, h, w], dtype=torch.float32).cuda()
                        action_attn_loss = mask_bce(action_attn[class_idx], attn_gt[class_idx])
                        bg_attn_loss = mask_bce(bg_attn, bg_seg)
                        attn_loss = args.action_attn_weight*action_attn_loss + args.bg_attn_weight*bg_attn_loss

                        action_attn_pred = action_attn[class_idx] > 0.5
                        inter, union = inter_and_union(action_attn_pred.reshape(-1, h, w), attn_gt[class_idx].reshape(-1, h, w), 1, 0)
                        action_inter.update(inter)
                        action_union.update(union)

                        bg_attn_pred = bg_attn > 0.5
                        inter, union = inter_and_union(bg_attn_pred, bg_seg, 1, 1)
                        bg_inter.update(inter)
                        bg_union.update(union)

                        attn_loss_epoch += float(attn_loss.item())
                        action_attn_loss_epoch += float(action_attn_loss.item())
                        bg_attn_loss_epoch += float(bg_attn_loss.item())

                    # w/ L_bg, w/o L_neg
                    elif args.bg_attn_weight>0. and not args.action_attn_weight >0:
                        b, l, n, h, w = attn.shape

                        if args.bg_upsample != 1:
                            attn = attn.reshape(-1, 1, h, w)
                            attn = F.interpolate(attn, size=ds_size, mode='bilinear')
                            _, _, h, w = attn.shape
                            attn = attn.reshape(b, l, n, h, w)

                        bg_attn = attn[:, ::2, -1, :, :].reshape(b, l//2, h, w)

                        # bg_idx = torch.ones(b, dtype=torch.bool).cuda()
                        # bg_idx = torch.reshape(bg_idx, (b, 1))

                        bg_attn_loss = mask_bce(bg_attn, bg_seg)
                        attn_loss = args.bg_attn_weight*bg_attn_loss

                        bg_attn_pred = bg_attn > 0.5
                        inter, union = inter_and_union(bg_attn_pred.reshape(-1, h, w), bg_seg.reshape(-1, h, w), 1, 1)
                        bg_inter.update(inter)
                        bg_union.update(union)

                        attn_loss_epoch += float(attn_loss.item())
                        bg_attn_loss_epoch += float(bg_attn_loss.item())

                else:
                    actor_loss = bce(pred_actor, actor)
                
                if args.action_attn_weight>0.or args.bg_attn_weight>0.:
                    loss = actor_loss + args.ego_loss_weight*ego_loss + attn_loss
                else:
                    loss = actor_loss + args.ego_loss_weight*ego_loss

                num_batches += 1
                total_loss += float(loss.item())
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

            if args.action_attn_weight>0. or args.bg_attn_weight>0.:
                attn_loss_epoch = attn_loss_epoch / num_batches
                print('attn loss:')
                print(attn_loss_epoch)
                if args.action_attn_weight >0:
                    action_attn_loss_epoch = action_attn_loss_epoch /num_batches
                    print('action_attn_loss')
                    print(action_attn_loss_epoch)
                if args.bg_attn_weight > 0:
                    bg_attn_loss_epoch = bg_attn_loss_epoch / num_batches
                    print('bg_attn_loss_epoch')
                    print(bg_attn_loss_epoch)
                
            if args.action_attn_weight>0:
                iou = action_inter.sum / (action_union.sum + 1e-10)
                for i, val in enumerate(iou):
                    print('Action IoU {0}: {1:.2f}'.format(i, val * 100))
            if args.bg_attn_weight > 0:
                iou = bg_inter.sum / (bg_union.sum + 1e-10)
                for i, val in enumerate(iou):
                    print('BG IoU {0}: {1:.2f}'.format(i, val * 100))

            map_pred_actor_list = np.stack(map_pred_actor_list, axis=0)
            label_actor_list = np.stack(label_actor_list, axis=0)
            
            map_pred_actor_list = map_pred_actor_list.reshape((map_pred_actor_list.shape[0], num_actor_class))
            label_actor_list = label_actor_list.reshape((label_actor_list.shape[0], num_actor_class))
            map_pred_actor_list = np.array(map_pred_actor_list)
            label_actor_list = np.array(label_actor_list)
            

            mAP = average_precision_score(
                    label_actor_list,
                    map_pred_actor_list.astype(np.float32),
                    )
            c_mAP = average_precision_score(
                    label_actor_list[:, :12],
                    map_pred_actor_list[:, :12].astype(np.float32)
                    )
            b_mAP = average_precision_score(
                    label_actor_list[:, 12:24],
                    map_pred_actor_list[:, 12:24].astype(np.float32)
                    )
            p_mAP = average_precision_score(
                    label_actor_list[:, 48:56],
                    map_pred_actor_list[:, 48:56].astype(np.float32),
                    )
            group_c_mAP = average_precision_score(
                    label_actor_list[:, 24:36],
                    map_pred_actor_list[:, 24:36].astype(np.float32)
                    )
            group_b_mAP = average_precision_score(
                    label_actor_list[:, 36:48],
                    map_pred_actor_list[:, 36:48].astype(np.float32)
                    )
            group_p_mAP = average_precision_score(
                    label_actor_list[:, 56:64],
                    map_pred_actor_list[:, 56:64].astype(np.float32),
                    )
            mAP_per_class = average_precision_score(
                    label_actor_list,
                    map_pred_actor_list.astype(np.float32), 
                    average=None)


            print(f'(val) mAP of the actor: {mAP}')
            print(f'(val) mAP of the c: {c_mAP}')
            print(f'(val) mAP of the b: {b_mAP}')
            print(f'(val) mAP of the p: {p_mAP}')
            print(f'(val) mAP of the c+: {group_c_mAP}')
            print(f'(val) mAP of the b+: {group_b_mAP}')
            print(f'(val) mAP of the p+: {group_p_mAP}')

            print(f'acc of the ego: {correct_ego/total_ego}')
            writer.add_scalar('ego', correct_ego/total_ego, epoch)


            total_loss = total_loss / float(num_batches)
            tqdm.write(f'Epoch {self.cur_epoch:03d}, Batch {batch_num:03d}:' + f' Loss: {total_loss:3.3f}')

            
torch.cuda.empty_cache() 
seq_len = args.seq_len
num_ego_class = 4
num_actor_class = 64

# Data
val_set = taco.TACO(args=args, training=False)
dataloader_val = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

model = generate_model(args, num_ego_class, num_actor_class).cuda()
trainer = Engine(args)
# model.load_state_dict(torch.load(os.path.join(args.logdir, 'model_100.pth')))

model_path = os.path.join(logdir, args.cp)
model.load_state_dict(torch.load(model_path))

trainer.validate(model, dataloader_val, None)
