import sys
sys.path.append('../')

import argparse
import json
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
from parser import get_parser
import math
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')

from sklearn.metrics import average_precision_score, precision_score, f1_score, recall_score, accuracy_score, hamming_loss
from scipy.optimize import linear_sum_assignment

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models

torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

from datasets.nuscenes import NUSCENES
from model import generate_model
from loss import ActionSlotLoss
from utils import AverageMeter

def plot_result(result,args):
    """
        result : mAP, loss, f1
    """
    # text = ["mAP", "loss", "f1"]
    text = ["mAP", "loss"]
    x = [i+1 for i in range(0,args.epochs,args.val_every)]
    x = x if x[-1] == args.epochs else x+[args.epochs]
    fig, ax = plt.subplots(2,1,figsize=(10,6))
    
    for i in range(2):
        ax[i].plot(x,result[:,i])
        ax[i].title.set_text(text[i])
    plt.show()	

def lambda_lr(epoch):
    if epoch<11:
        lr = math.pow(1.1,epoch)
    else:
        lr = math.pow(0.7,int(epoch/3))
    return lr

def set_lr(model):
    params = list(filter(lambda kv: kv[0].startswith("head"), model.named_parameters()))
    base_params = list(filter(lambda kv: not kv[0].startswith("head") , model.named_parameters()))
    return [
        {'params': [temp[1] for temp in base_params]},
        {'params': [temp[1] for temp in params], 'lr': 2e-2}
    ]

class Engine(object):
    """Engine that runs training and inference.
    Args
        - cur_epoch (int): Current epoch.
        - print_every (int): How frequently (# batches) to print loss.
        - validate_every (int): How frequently (# epochs) to run validation.
        
    """

    def __init__(self, args, model, optimizer, num_actor_class, scheduler=None):
        self.args = args
  
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_actor_class = num_actor_class
        if hasattr(self.model, 'resolution'):
            attention_res = (self.model.resolution[0]*args.bg_upsample, self.model.resolution[1]*args.bg_upsample)
        else:
            attention_res = None
        self.criterion = ActionSlotLoss(args, num_actor_class, attention_res).to(self.args.device)

        self.cur_epoch = 0
        self.train_loss = []
        self.val_loss = []
        self.bestval = 1e10
        self.best_mAP = 1e-5
        self.reset_log()
        
    def reset_log(self):
        self.loss_epoch = 0.
        self.ego_loss_epoch = 0.
        self.seg_loss_epoch = 0.
        self.attn_loss_epoch= 0.
        self.action_attn_loss_epoch = 0.
        self.bg_attn_loss_epoch = 0.
        self.actor_loss_epoch = 0.
        self.correct_ego = 0
        self.total_ego = 0
        self.label_actor_list = []
        self.map_pred_actor_list = []
        self.action_inter = AverageMeter()
        self.action_union = AverageMeter()
        self.bg_inter = AverageMeter()
        self.bg_union = AverageMeter()

    def step(self,batch,mode):

        for k in batch:
            if isinstance(batch[k],torch.Tensor):
                batch[k] = batch[k].to(self.args.device)
        
        video_in = batch['videos']
        if self.args.box:
            box_in = batch['box']
            if isinstance(box_in,np.ndarray):
                boxes = torch.from_numpy(box_in).to(self.args.device, dtype=torch.float32)
            else:
                boxes = box_in.to(self.args.device, dtype=torch.float32)
        inputs = []
        for i in range(seq_len):
            inputs.append(video_in[i].to(self.args.device, dtype=torch.float32))

        # --------------------------------------------
        attn = None
        # object-based models
        if self.args.box:
            pred_ego, pred_actor = self.model(inputs, boxes)

        else:
            if 'slot' in self.args.model_name or 'mvit' in self.args.model_name:
                pred_ego, pred_actor, attn = self.model(inputs)
            else:
                pred_ego, pred_actor = self.model(inputs)
        loss_dict = self.criterion({'ego':pred_ego,'actor':pred_actor,'attn':attn},batch, False if mode == 'train' else True)
        if self.args.parallel:
            for _, v in loss_dict.items():
                if isinstance(v,torch.Tensor):
                    v = v.mean()

        ego_loss = loss_dict['ego']
        if ego_loss is None:
            ego_loss = torch.Tensor([0.0])
        actor_loss = loss_dict['actor']
        action_attn_loss, bg_attn_loss = loss_dict['attn']['attn_loss'], loss_dict['attn']['bg_attn_loss']

        if self.criterion.attn_loss_type == 1:
            attn_loss = action_attn_loss

        elif self.criterion.attn_loss_type == 2:
            attn_loss = action_attn_loss * self.args.action_attn_weight

            self.attn_loss_epoch += float(attn_loss.item())
            self.action_attn_loss_epoch += float(action_attn_loss.item())

        elif self.criterion.attn_loss_type == 3:
            attn_loss = self.args.action_attn_weight * action_attn_loss + self.args.bg_attn_weight * bg_attn_loss

            self.attn_loss_epoch += float(attn_loss.item())
            self.action_attn_loss_epoch += float(action_attn_loss.item())
            self.bg_attn_loss_epoch += float(bg_attn_loss.item())
            
        elif self.criterion.attn_loss_type == 4:
            attn_loss = self.args.bg_attn_weight * bg_attn_loss

            self.attn_loss_epoch += float(attn_loss.item())
            self.bg_attn_loss_epoch += float(bg_attn_loss.item())

        if 'slot' in self.args.model_name and (self.args.action_attn_weight>0. or  self.args.bg_attn_weight>0. or self.args.obj_mask):
            loss = actor_loss + self.args.ego_loss_weight*ego_loss + attn_loss
        else:
            loss = actor_loss + self.args.ego_loss_weight*ego_loss
        self.loss_epoch += float(loss.item())

        _, pred_ego = torch.max(pred_ego.data, 1)
        ego, actor = batch['ego'] ,batch['actor']
        self.total_ego += ego.size(0)
        self.correct_ego += (pred_ego == ego).sum().item()

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
            self.map_pred_actor_list.append(map_batch_new_pred_actor)
            self.label_actor_list.append(batch['slot_eval_gt'].cpu().numpy())
        else:
            pred_actor = torch.sigmoid(pred_actor)
            self.map_pred_actor_list.append(pred_actor.detach().cpu().numpy())
            self.label_actor_list.append(actor.detach().cpu().numpy())
        
        actor_loss, ego_loss = actor_loss.mean(), ego_loss.mean()
        self.actor_loss_epoch += float(actor_loss.item())
        self.ego_loss_epoch += float(ego_loss.item())

        # self.cur_iter += 1
        
        if mode == 'train':
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
        else:
            if loss_dict['attn']['action_inter'] is not None:
                self.action_inter.update(loss_dict['attn']['action_inter'])
                self.action_union.update(loss_dict['attn']['action_union'])
            if loss_dict['attn']['bg_inter'] is not None:
                self.bg_inter.update(loss_dict['attn']['bg_inter'])
                self.bg_union.update(loss_dict['attn']['bg_union'])

    def _parallel(self):
        self.model = nn.DataParallel(self.model)
        self.criterion = nn.DataParallel(self.criterion)

    def train(self):
        self.reset_log()
        model_name = self.args.model_name
        if scheduler is not None:
            print(f"Current epoch: {self.cur_epoch}, lr: {scheduler.get_last_lr()}")
        else:
            print(f"Current epoch: {self.cur_epoch}")

        self.model = self.model.train()
        # Train loop
        self.num_batches = len(dataloader_train)
        for data in tqdm(dataloader_train):
            self.step(data,'train')
        if scheduler is not None:
            scheduler.step()

        map_pred_actor_list = np.stack(self.map_pred_actor_list, axis=0)
        label_actor_list = np.stack(self.label_actor_list, axis=0)

        map_pred_actor_list = map_pred_actor_list.reshape((map_pred_actor_list.shape[0]*args.batch_size, num_actor_class))
        label_actor_list = label_actor_list.reshape((label_actor_list.shape[0]*args.batch_size, num_actor_class))

        mAP = average_precision_score(
            label_actor_list,
            map_pred_actor_list.astype(np.float32))

        loss_epoch = self.loss_epoch / self.num_batches	
        actor_loss_epoch = self.actor_loss_epoch / self.num_batches
        ego_loss_epoch = self.ego_loss_epoch / self.num_batches

        print(f'acc of the ego: {self.correct_ego/self.total_ego}')

        print('total loss')
        print(loss_epoch)

        if self.args.action_attn_weight > 0 or self.args.bg_attn_weight >0:
            attn_loss_epoch = self.attn_loss_epoch / self.num_batches
            print('attn loss:')
            print(attn_loss_epoch)
            if self.args.action_attn_weight > 0:
                action_attn_loss_epoch = self.action_attn_loss_epoch /self.num_batches
                print('action_attn_loss')
                print(action_attn_loss_epoch)
            if args.bg_attn_weight >0:
                bg_attn_loss_epoch = self.bg_attn_loss_epoch / self.num_batches
                print('bg_attn_loss_epoch')
                print(bg_attn_loss_epoch)

        print('-'*20)
        print('actor loss:')
        print(actor_loss_epoch)
        print('ego loss:')
        print(ego_loss_epoch)
        print(f'(train) mAP of the actor: {mAP}')
        self.train_loss.append(loss_epoch)
        self.cur_epoch += 1
        

    def validate(self, dataloader):
        self.model = self.model.eval()
        save_cp = False
        self.reset_log()
        with torch.no_grad():	
            for data in tqdm(dataloader):
                self.step(data,'val')
            
            if args.action_attn_weight>0. or args.bg_attn_weight>0.:
                attn_loss_epoch = self.attn_loss_epoch / self.num_batches
                print('attn loss:')
                print(attn_loss_epoch)
                if args.action_attn_weight>0.:
                    action_attn_loss_epoch = self.action_attn_loss_epoch /self.num_batches
                    print('action_attn_loss')
                    print(action_attn_loss_epoch)
                if args.bg_attn_weight>0.:
                    bg_attn_loss_epoch = self.bg_attn_loss_epoch / self.num_batches
                    print('bg_attn_loss_epoch')
                    print(bg_attn_loss_epoch)
                
            if args.action_attn_weight >0 and args.bg_attn_weight>0:
                iou = self.action_inter.sum / (self.action_union.sum + 1e-10)
                for i, val in enumerate(iou):
                    print('Action IoU {0}: {1:.2f}'.format(i, val * 100))

                iou = self.bg_inter.sum / (self.bg_union.sum + 1e-10)
                for i, val in enumerate(iou):
                    print('BG IoU {0}: {1:.2f}'.format(i, val * 100))

            map_pred_actor_list = np.stack(self.map_pred_actor_list, axis=0)
            label_actor_list = np.stack(self.label_actor_list, axis=0)
            
            map_pred_actor_list = map_pred_actor_list.reshape((map_pred_actor_list.shape[0], num_actor_class))
            label_actor_list = label_actor_list.reshape((label_actor_list.shape[0], num_actor_class))
            map_pred_actor_list = np.array(map_pred_actor_list)
            label_actor_list = np.array(label_actor_list)

            mAP = average_precision_score(
                    np.concatenate([label_actor_list[:, :36], label_actor_list[:, -16:]], axis=1),
                    np.concatenate([map_pred_actor_list[:, :36], map_pred_actor_list[:, -16:]], axis=1).astype(np.float32),
                    )
            c_mAP = average_precision_score(
                    label_actor_list[:, :12],
                    map_pred_actor_list[:, :12].astype(np.float32)
                    )
            b_mAP = average_precision_score(
                    label_actor_list[:, 24:36],
                    map_pred_actor_list[:, 24:36].astype(np.float32)
                    )
            p_mAP = average_precision_score(
                    label_actor_list[:, 48:56],
                    map_pred_actor_list[:, 48:56].astype(np.float32),
                    )
            group_c_mAP = average_precision_score(
                    label_actor_list[:, 12:24],
                    map_pred_actor_list[:, 12:24].astype(np.float32)
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


            print(f'(val) mAP: {mAP}')
            print(f'(val) mAP of the c: {c_mAP}')
            print(f'(val) mAP of the b: {b_mAP}')
            print(f'(val) mAP of the p: {p_mAP}')
            print(f'(val) mAP of the c+: {group_c_mAP}')
            print(f'(val) mAP of the b+: {group_b_mAP}')
            print(f'(val) mAP of the p+: {group_p_mAP}')

            print(f'acc of the ego: {self.correct_ego/self.total_ego}')
            writer.add_scalar('ego', self.correct_ego/self.total_ego, self.cur_epoch)
            if mAP > self.best_mAP:
                self.best_mAP = mAP
                self.best_log = [
                    f'(val) mAP: {mAP}',
                    f'(val) mAP of the c: {c_mAP}',
                    f'(val) mAP of the b: {b_mAP}',
                    f'(val) mAP of the p: {p_mAP}',
                    f'(val) mAP of the c+: {group_c_mAP}',
                    f'(val) mAP of the b+: {group_b_mAP}',
                    f'(val) mAP of the p+: {group_p_mAP}'
                ]
                save_cp = True
            print(f'best mAP : {self.best_mAP}')

            with open(os.path.join(logdir, 'mAP.txt'), 'a') as f:
                f.write('epoch: ' + str(self.cur_epoch))
                f.write('\n')
                f.write('best mAP: %.4f' % self.best_mAP)
                f.write('\n')
                f.write('mAP: %.4f' % mAP)
                f.write('\n')
                f.write('mAP of c: %.4f' % c_mAP)
                f.write('\n')
                f.write('mAP of b: %.4f' % b_mAP)
                f.write('\n')
                f.write('mAP of p: %.4f' % p_mAP)
                f.write('\n')
                f.write('mAP of c+: %.4f' % group_c_mAP)
                f.write('\n')
                f.write('mAP of b+: %.4f' % group_b_mAP)
                f.write('\n')
                f.write('mAP of p+: %.4f' % group_p_mAP)
                f.write('\n')

                f.write('c per class: + \n')
                for ap in mAP_per_class[:12].tolist():
                    f.write("%.4f " % ap)
                f.write('\n')
                f.write('b per class: \n')
                for ap in mAP_per_class[12:24].tolist():
                    f.write("%.4f " % ap)
                f.write('\n')
                f.write('c+ per class: \n')
                for ap in mAP_per_class[24:36].tolist():
                    f.write("%.4f " % ap)
                f.write('\n')
                f.write('b+ per class: \n')
                for ap in mAP_per_class[36:48].tolist():
                    f.write("%.4f " % ap)
                f.write('\n')
                f.write('p per class: \n')
                for ap in mAP_per_class[48:56].tolist():
                    f.write("%.4f " % ap)
                f.write('\n')
                f.write('p+ per class: \n')
                for ap in mAP_per_class[56:64].tolist():
                    f.write("%.4f " % ap)
                f.write('\n')
                f.write('*'*15 + '\n')

            total_loss = self.loss_epoch / float(self.num_batches)
            tqdm.write(f'Epoch {self.cur_epoch:03d} Loss: {total_loss:3.3f}')

            # writer.add_scalar('val_loss', total_loss, self.cur_epoch)
            
            self.val_loss.append(total_loss)
        # return save_cp, [mAP, total_loss, mean_f1]
        return save_cp, [mAP, total_loss]

    def save(self, is_best):

        save_best = False
        if is_best:
            self.bestval = self.val_loss[-1]
            self.bestval_epoch = self.cur_epoch
            save_best = True
        
        # Create a dictionary of all data to save

        # log_table = {
        # 	'epoch': self.cur_epoch,
        # 	'iter': self.cur_iter,
        # 	'bestval': float(self.bestval.data),
        # 	'bestval_epoch': self.bestval_epoch,
        # 	'train_loss': self.train_loss,
        # 	'val_loss': self.val_loss,
        # }

        # Save ckpt for every epoch
        # torch.save(model.state_dict(), os.path.join(logdir, 'model_%d.pth'%self.cur_epoch))
        # tqdm.write('====== Saved recent model ======>')
        
        if save_best:
            torch.save(model.state_dict(), os.path.join(logdir, 'best_model.pth'))
            # torch.save(optimizer.state_dict(), os.path.join(logdir, 'best_optim.pth'))
            tqdm.write('====== Overwrote best model ======>')

if __name__ == '__main__':
    args, logdir = get_parser()
    print(args)
    writer = SummaryWriter(log_dir=logdir)
    seq_len = args.seq_len

    num_ego_class = 4
    num_actor_class = 64


    # Data
    train_set = NUSCENES(args=args)
    val_set = NUSCENES(args=args, training=False)
    label_stat = []
    for i in range(7):
        label_stat.append({})
        for k in train_set.label_stat[i].keys():
            label_stat[i][k] = train_set.label_stat[i][k] + val_set.label_stat[i][k]
    print('*'*20)
    print('nuScenes Dataset')
    print('c_stat:')
    print(label_stat[0])
    print('b_stat:')
    print(label_stat[1])
    print('c+_stat:')
    print(label_stat[2])
    print('b+_stat:')
    print(label_stat[3])
    print('p_stat:')
    print(label_stat[4])
    print('p+_stat:')
    print(label_stat[5])
    print('ego_stat')
    print(label_stat[6])
        
    dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    dataloader_val = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    # Model
    model = generate_model(args, num_ego_class, num_actor_class).cuda()

    if args.pretrain != '' :
        model_path = os.path.join(args.cp)
        checkpoint = torch.load(model_path)
        checkpoint = {k: v for k, v in checkpoint.items() if (k in checkpoint and 'fc' not in k)}
        model.load_state_dict(checkpoint, strict=False)
        if args.pretrain =='oats' and args.model_name == 'action_slot':
            model.slot_attention.extend_slots()
        # elif args.model_name == 'slot_vps':
        #     model.extend_slots()
        # else:
        #     model.load_state_dict(checkpoint, strict=False)
        # else:
    if 'mvit' == args.model_name:
        params = set_lr(model)#
    else:
        params = [{'params':model.parameters()}]
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    if args.scheduler:
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
    else:
        scheduler = None

    # -----------	
    trainer = Engine(args, model, optimizer, num_actor_class, scheduler)

    # Create logdir
    print(f'Checkpoint path: {logdir}')

    result_list = []
    if not args.test:
        for epoch in range(trainer.cur_epoch, args.epochs): 
            
            trainer.train()
            if (epoch % args.val_every == 0 or epoch == args.epochs-1): 
                    is_best, res = trainer.validate(dataloader_val)
                    # trainer.validate(dataloader_val_train, None)
                    trainer.save(is_best)
                    result_list.append(res)
    print('********** Best model **********')
    for s in trainer.best_log:
        print(s)
    plot_result(np.array(result_list),args)
