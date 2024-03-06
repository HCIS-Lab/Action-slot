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
sys.path.append('/media/hankung/ssd/retrieval/datasets')
sys.path.append('/media/hankung/ssd/retrieval/config')
sys.path.append('/media/hankung/ssd/retrieval/models')

import road_dataset

from sklearn.metrics import average_precision_score, precision_score, f1_score, recall_score, accuracy_score, hamming_loss
from PIL import Image, ImageDraw

from model import generate_model
from utils import *
from torchvision import models
import matplotlib.image
from scipy.optimize import linear_sum_assignment
from parser import get_parser

actor_table = ['z1-z2', 'z1-z3', 'z1-z4',
                'z2-z1', 'z2-z3', 'z2-z4',
                'z3-z1', 'z3-z2', 'z3-z4',
                'z4-z1', 'z4-z2', 'z4-z3',
                'c1-c2', 'c1-c4', 
                'c2-c1', 'c2-c3', 
                'c3-c2', 'c3-c4', 
                'c4-c1', 'c4-c3', 'bg']


torch.cuda.empty_cache()
args = get_parser()
print(args)
writer = SummaryWriter(log_dir=args.logdir)

class Engine(object):
    """Engine that runs training and inference.
    Args
        - cur_epoch (int): Current epoch.
        - print_every (int): How frequently (# batches) to print loss.
        - validate_every (int): How frequently (# epochs) to run validation.
        
    """

    def __init__(self, args, cur_epoch=0, cur_iter=0, bce_weight=1, ego_weight=1):
        self.cur_epoch = cur_epoch
        self.cur_iter = cur_iter
        self.bestval_epoch = cur_epoch
        self.train_loss = []
        self.val_loss = []
        self.bestval = 1e10
        self.best_f1 = 1e-5
        self.bce_weight = bce_weight
        self.ego_weight = ego_weight
        self.args = args

    def validate(self, model, dataloader, epoch, cam=False, model_name='', ce_weight=5):
        model.eval()
        ego_ce = nn.CrossEntropyLoss(reduction='mean').cuda()
        seg_ce = nn.CrossEntropyLoss(reduction='mean').cuda()


        t_confuse_sample, t_confuse_both_sample, t_confuse_pred, t_confuse_both_pred, t_confuse_both_miss, t_confuse_far_both_sample, t_confuse_far_both_miss = 0, 0, 0, 0, 0, 0, 0

        if ('slot' in model_name and not args.fix_slot) or args.box:
            empty_weight = torch.ones(num_actor_class+1)*ce_weight
            empty_weight[-1] = self.args.empty_weight
            slot_ce = nn.CrossEntropyLoss(reduction='mean', weight=empty_weight).cuda()
        elif 'slot' in model_name and args.fix_slot:
            pos_weight = torch.ones([num_actor_class])*args.weight
            slot_ce = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight).cuda()
        else:
            pos_weight = torch.ones([num_actor_class])*args.weight
            bce = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight).cuda()
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
            mean_f1 = 0
            label_actor_list = []
            f1_pred_actor_list = []
            map_pred_actor_list = []

            action_inter = AverageMeter()
            action_union = AverageMeter()
            bg_inter = AverageMeter()
            bg_union = AverageMeter()

            for batch_num, data in enumerate(tqdm(dataloader)):
                fronts_in = data['fronts']
                inputs = []
                for i in range(seq_len):
                    inputs.append(fronts_in[i].to(args.device, dtype=torch.float32))
     
                batch_size = inputs[0].shape[0]
                ego = data['ego'].to(args.device)
                actor = torch.FloatTensor(data['actor']).to(args.device)

                # if args.seg:
                #     h, w = seg_front[0].shape[-2], seg_front[0].shape[-1]
                #     seg_front = torch.stack(seg_front, 0)
                #     seg_front = torch.permute(seg_front, (1, 0, 2, 3)) #[batch, len, h, w]
                #     b, l, h, w = seg_front.shape
                #     ds_size = (model.resolution[0]*args.upsample, model.resolution[1]*args.upsample)
                #     seg_front = torch.reshape(seg_front, (b*l, 1, h, w))
                #     seg_front = F.interpolate(seg_front, size=ds_size)
                #     seg_front = torch.reshape(seg_front, (b, l, ds_size[0], ds_size[1]))
                if ('slot' in model_name) or args.box or ('mvit' in model_name):
                    pred_ego, pred_actor, attn = model(inputs)
                    return
                else:
                    pred_ego, pred_actor = model(inputs)

                ego_loss = ego_ce(pred_ego, ego)

                # if ('slot' in model_name and not args.fix_slot) or args.box:
                #     bs, num_queries = pred_actor.shape[:2]
                #     out_prob = pred_actor.clone().detach().flatten(0, 1).softmax(-1)
                #     actor_gt_np = actor.clone().detach()
                #     tgt_ids = torch.cat([v for v in actor_gt_np.detach()])
                #     C = -out_prob[:, tgt_ids].clone().detach()
                #     C = C.view(bs, num_queries, -1).cpu()
                #     sizes = [len(v) for v in actor_gt_np]
                #     indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
                #     indx = [(torch.as_tensor(i, dtype=torch.int64).detach(), torch.as_tensor(j, dtype=torch.int64).detach()) for i, j in indices]
                #     batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indx)]).detach()
                #     src_idx = torch.cat([src for (src, _) in indx]).detach()
                #     idx = (batch_idx, src_idx)
                #     target_classes_o = torch.cat([t[J] for t, (_, J) in zip(actor, indx)]).cuda()
                #     target_classes = torch.full(pred_actor.shape[:2], num_actor_class,
                #                         dtype=torch.int64, device=out_prob.device)

                #     target_classes[idx] = target_classes_o
                #     actor_loss = slot_ce(pred_actor.transpose(1, 2), target_classes)

                # elif 'slot' in model_name and args.fix_slot:
                #     actor_loss = slot_ce(pred_actor, actor)
                #     if args.mask and args.action_attn_weight >0 and args.bg_attn_weight==0.:
                #         b, l, n, h, w = attn.shape
                #         if args.upsample != 1.0:
                #             attn = attn.reshape(-1, 1, h, w)
                #             attn = F.interpolate(attn, size=ds_size, mode='bilinear')
                #             _, _, h, w = attn.shape
                #             attn = attn.reshape(b, l, n, h, w)
                #         action_attn = attn[:, :, :num_actor_class, :, :]
                #         class_idx = actor == 0.0
                #         class_idx = class_idx.view(b, num_actor_class, 1, 1, 1).repeat(1, 1, l, h, w)
                #         class_idx = torch.permute(class_idx, (0, 2, 1, 3, 4))

                #         attn_gt = torch.zeros([b, l, num_actor_class, h, w], dtype=torch.float32).cuda()
                #         bcecriterion = nn.BCELoss()
                #         action_attn_loss = bcecriterion(action_attn[class_idx], attn_gt[class_idx])
                #         attn_loss = args.action_attn_weight*action_attn_loss

                #         action_attn_pred = action_attn[class_idx] > 0.5
                #         inter, union = inter_and_union(action_attn_pred.reshape(-1, h, w), attn_gt[class_idx].reshape(-1, h, w), 1, 0)
                #         action_inter.update(inter)
                #         action_union.update(union)

                #         attn_loss_epoch += float(attn_loss.item())
                #         action_attn_loss_epoch += float(action_attn_loss.item())

                #     elif args.mask and args.seg and args.action_attn_weight >0. and args.bg_attn_weight>0.:
                #         b, l, n, h, w = attn.shape

                #         if args.upsample != 1.0:
                #             attn = attn.reshape(-1, 1, h, w)
                #             attn = F.interpolate(attn, size=ds_size, mode='bilinear')
                #             _, _, h, w = attn.shape
                #             attn = attn.reshape(b, l, n, h, w)

                #         action_attn = attn[:, :, :num_actor_class, :, :]
                #         bg_attn = attn[:, :, -1, :, :].reshape(b, l, h, w)

                #         class_idx = actor == 0.0
                #         bg_idx = torch.ones(b, dtype=torch.bool).cuda()
                #         bg_idx = torch.reshape(bg_idx, (b, 1))
                #         # class_idx = torch.cat((class_idx, bg_idx), -1)
                #         class_idx = class_idx.view(b, num_actor_class, 1, 1, 1).repeat(1, 1, l, h, w)
                #         class_idx = torch.permute(class_idx, (0, 2, 1, 3, 4))

                #         attn_gt = torch.zeros([b, l, num_actor_class, h, w], dtype=torch.float32).cuda()
                #         # seg_front = torch.reshape(seg_front, (b, l, 1, h, w))
                #         # attn_gt = torch.cat((attn_gt, seg_front), 2)
                #         bcecriterion = nn.BCELoss()
                #         action_attn_loss = bcecriterion(action_attn[class_idx], attn_gt[class_idx])
                #         bg_attn_loss = bcecriterion(bg_attn, seg_front)
                #         attn_loss = args.action_attn_weight*action_attn_loss + args.bg_attn_weight*bg_attn_loss

                #         action_attn_pred = action_attn[class_idx] > 0.5
                #         inter, union = inter_and_union(action_attn_pred.reshape(-1, h, w), attn_gt[class_idx].reshape(-1, h, w), 1, 0)
                #         action_inter.update(inter)
                #         action_union.update(union)

                #         bg_attn_pred = bg_attn > 0.5
                #         inter, union = inter_and_union(bg_attn_pred, seg_front, 1, 1)
                #         bg_inter.update(inter)
                #         bg_union.update(union)

                #         attn_loss_epoch += float(attn_loss.item())
                #         action_attn_loss_epoch += float(action_attn_loss.item())
                #         bg_attn_loss_epoch += float(bg_attn_loss.item())
                #     elif not args.mask and args.seg and args.bg_attn_weight>0 and args.action_attn_weight ==0.:
                #         b, l, n, h, w = attn.shape

                #         if args.upsample != 1.0:
                #             attn = attn.reshape(-1, 1, h, w)
                #             attn = F.interpolate(attn, size=ds_size, mode='bilinear')
                #             _, _, h, w = attn.shape
                #             attn = attn.reshape(b, l, n, h, w)

                #         bg_attn = attn[:, :, -1, :, :].reshape(b, l, h, w)

                #         bg_idx = torch.ones(b, dtype=torch.bool).cuda()
                #         bg_idx = torch.reshape(bg_idx, (b, 1))

                #         bcecriterion = nn.BCELoss()
                #         bg_attn_loss = bcecriterion(bg_attn, seg_front)
                #         attn_loss = args.bg_attn_weight*bg_attn_loss

                #         bg_attn_pred = bg_attn > 0.5
                #         inter, union = inter_and_union(bg_attn_pred.reshape(-1, h, w), seg_front.reshape(-1, h, w), 1, 1)
                #         bg_inter.update(inter)
                #         bg_union.update(union)

                #         attn_loss_epoch += float(attn_loss.item())
                #         bg_attn_loss_epoch += float(bg_attn_loss.item())

                # else:
                #     bce = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight).cuda()
                #     actor_loss = bce(pred_actor, actor)
                
                # if (args.mask and args.action_attn_weight>0.) or (args.seg and args.bg_attn_weight>0.):
                #     loss = actor_loss + args.ego_weight*ego_loss + attn_loss
                # else:
                #     loss = actor_loss + args.ego_weight*ego_loss
                num_batches += 1
                # total_loss += float(loss.item())
                pred_ego = torch.nn.functional.softmax(pred_ego, dim=1)
                _, pred_ego = torch.max(pred_ego.data, 1)

                if ('slot' in model_name and not args.fix_slot) or args.box:
                    pred_actor = torch.nn.functional.softmax(pred_actor, dim=-1)
                    _, pred_actor_idx = torch.max(pred_actor.data, -1)
                    pred_actor_idx = pred_actor_idx.detach().cpu().numpy().astype(int)
                    f1_pred_actor= []
                    map_batch_new_pred_actor = []
                    for i, b in enumerate(pred_actor_idx):
                        f1_new_pred = np.zeros(num_actor_class, dtype=int)
                        map_new_pred = np.zeros(num_actor_class, dtype=np.float32)+1e-5

                        for j, pred in enumerate(b):
                            if pred != num_actor_class:
                                f1_new_pred[pred] = 1.
                                if pred_actor[i, j, pred] > map_new_pred[pred]:
                                    map_new_pred[pred] = pred_actor[i, j, pred]
                        f1_pred_actor.append(f1_new_pred)
                        map_batch_new_pred_actor.append(map_new_pred)
                    f1_pred_actor = np.array(f1_pred_actor)
                    pred_actor = np.array(map_batch_new_pred_actor)
                    f1_pred_actor_list.append(f1_pred_actor)
                    map_pred_actor_list.append(pred_actor)
                    label_actor_list.append(data['slot_eval_gt'])
                else:
                    pred_actor = torch.sigmoid(pred_actor)
                    f1_pred_actor = pred_actor > 0.5
                    f1_pred_actor = f1_pred_actor.float()
                    map_pred_actor_list.append(pred_actor.detach().cpu().numpy())
                    f1_pred_actor_list.append(f1_pred_actor.detach().cpu().numpy())
                    label_actor_list.append(actor.detach().cpu().numpy())

                total_ego += ego.size(0)
                correct_ego += (pred_ego == ego).sum().item()

            # if (args.mask and args.action_attn_weight>0.) or (args.seg and args.bg_attn_weight>0.):
            #     attn_loss_epoch = attn_loss_epoch / num_batches
            #     print('attn loss:')
            #     print(attn_loss_epoch)
            #     if args.mask:
            #         action_attn_loss_epoch = action_attn_loss_epoch /num_batches
            #         print('action_attn_loss')
            #         print(action_attn_loss_epoch)
            #     if args.seg:
            #         bg_attn_loss_epoch = bg_attn_loss_epoch / num_batches
            #         print('bg_attn_loss_epoch')
            #         print(bg_attn_loss_epoch)
                
            # if args.seg and args.mask and args.action_attn_weight>0. and args.bg_attn_weight >0.:
            #     iou = action_inter.sum / (action_union.sum + 1e-10)
            #     for i, val in enumerate(iou):
            #         print('Action IoU {0}: {1:.2f}'.format(i, val * 100))

            #     iou = bg_inter.sum / (bg_union.sum + 1e-10)
            #     for i, val in enumerate(iou):
            #         print('BG IoU {0}: {1:.2f}'.format(i, val * 100))

            f1_pred_actor_list = np.stack(f1_pred_actor_list, axis=0)
            map_pred_actor_list = np.stack(map_pred_actor_list, axis=0)
            label_actor_list = np.stack(label_actor_list, axis=0)
            
            f1_pred_actor_list = f1_pred_actor_list.reshape((f1_pred_actor_list.shape[0], num_actor_class))
            map_pred_actor_list = map_pred_actor_list.reshape((map_pred_actor_list.shape[0], num_actor_class))
            label_actor_list = label_actor_list.reshape((label_actor_list.shape[0], num_actor_class))
            f1_pred_actor_list = np.array(f1_pred_actor_list)
            map_pred_actor_list = np.array(map_pred_actor_list)
            label_actor_list = np.array(label_actor_list)
            
            if ('slot' in model_name and not args.fix_slot) or args.box:
                mean_f1 = f1_score(
                        label_actor_list.astype('int64'), 
                        f1_pred_actor_list.astype('int64'),
                        labels=[i for i in range(num_actor_class)],
                        average='samples',
                        zero_division=0)
            else:
                mean_f1 = f1_score(
                        label_actor_list.astype('int64'), 
                        f1_pred_actor_list.astype('int64'),
                        average='samples',
                        zero_division=0)
                p = precision_score(f1_pred_actor_list.astype('int64'), label_actor_list.astype('int64'), average='macro')
                print(f'(val) precision of the actor: {np.round(p,3)}')
            f1_class = f1_score(
                    label_actor_list.astype('int64'), 
                    f1_pred_actor_list.astype('int64'),
                    average=None,
                    zero_division=0)
            mAP = average_precision_score(
                    label_actor_list,
                    map_pred_actor_list.astype(np.float32),
                    )
            vehicle_mAP = average_precision_score(
                    label_actor_list[:, :12],
                    map_pred_actor_list[:, :12].astype(np.float32)
                    )
            ped_mAP = average_precision_score(
                    label_actor_list[:, 12:],
                    map_pred_actor_list[:, 12:].astype(np.float32),
                    )
            mAP_per_class = average_precision_score(
                    label_actor_list,
                    map_pred_actor_list.astype(np.float32), 
                    average=None)


            print(f'(val) mAP of the actor: {np.round(mAP,3)}')
            print(f'(val) mAP of the v: {np.round(vehicle_mAP,3)}')
            print(f'(val) mAP of the p: {np.round(ped_mAP,3)}')
            print(f'(val) f1 of the actor: {np.round(mean_f1,3)}')
            print(f'acc of the ego: {np.round((correct_ego/total_ego),3)}')
            mAP_per_class = np.round(mAP_per_class*100,1)
            print('mAP vehicle:')
            for value in mAP_per_class[:12]:
                print(value,end=' & ')
            # print(np.round(mAP_per_class[:12],3))
            print('\nmAP ped:')
            for value in mAP_per_class[12:]:
                print(value,end=' & ')
            # print(np.round(mAP_per_class[12:],3))

            # total_loss = total_loss / float(num_batches)
            # tqdm.write(f'Epoch {self.cur_epoch:03d}, Batch {batch_num:03d}:' + f' Loss: {total_loss:3.3f}')

            
torch.cuda.empty_cache() 
seq_len = args.seq_len
num_ego_class = 4
num_actor_class = 20


# Data
val_set = road_dataset.ROAD(args=args, seq_len=seq_len, training=False, seg=args.seg, num_class=num_actor_class, model_name=args.id, num_slots=args.num_slots, box=args.box)
dataloader_val = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

model = generate_model(args, args.id, num_ego_class, num_actor_class, args.seq_len).cuda()
trainer = Engine(args, bce_weight=args.bce, ego_weight=args.ego_weight)
# model.load_state_dict(torch.load(os.path.join(args.logdir, 'model_100.pth')))

model_path = os.path.join(args.logdir, args.cp)
model.load_state_dict(torch.load(model_path))

trainer.validate(model, dataloader_val, None, model_name=args.id, ce_weight=args.ce_weight)
