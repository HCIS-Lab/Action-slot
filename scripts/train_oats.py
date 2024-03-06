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

sys.path.append('../datasets')
sys.path.append('../configs')
sys.path.append('../models')


import oats

from sklearn.metrics import average_precision_score, precision_score, recall_score, accuracy_score


from PIL import Image

from model import generate_model
from utils import *
from torchvision import models
import matplotlib.image
import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
from scipy.optimize import linear_sum_assignment
from parser import get_parser
import math


def plot_result(result):
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

torch.cuda.empty_cache()
args, logdir = get_parser()
print(args)
writer = SummaryWriter(log_dir=logdir)

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

	def __init__(self, args, cur_epoch=0, cur_iter=0):
		self.args = args
		self.cur_epoch = cur_epoch
		self.cur_iter = cur_iter
		self.bestval_epoch = cur_epoch
		self.train_loss = []
		self.val_loss = []
		self.bestval = 1e10
		self.best_mAP = 1e-5

	def train(self, model, optimizer, epoch, scheduler=None):
		model_name = args.model_name
		if scheduler is not None:
			print(f"Current epoch: {epoch}, lr: {scheduler.get_last_lr()}")
		else:
			print(f"Current epoch: {epoch}")

		loss_epoch = 0.
		seg_loss_epoch = 0.
		attn_loss_epoch= 0.
		action_attn_loss_epoch = 0.
		bg_attn_loss_epoch = 0.
		actor_loss_epoch = 0.
		num_batches = 0



		label_actor_list = []
		map_pred_actor_list = []
  
		# seg_ce = nn.CrossEntropyLoss(reduction='mean')

		model.train()

		if args.parallel:
			model = nn.DataParallel(model)

		# ce loss for object-based models
		ce_weights = torch.ones(num_actor_class+1)*args.ce_pos_weight
		ce_weights[-1] = self.args.ce_neg_weight

		# bce loss for slot-based models
		if ('slot' in args.model_name and not self.args.allocated_slot) or self.args.box:
			instance_ce = nn.CrossEntropyLoss(reduction='mean', weight=ce_weights).cuda()
		elif 'slot' in args.model_name and args.allocated_slot:
			bce_weights = torch.ones([num_actor_class])*args.bce_pos_weight
			bce = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=bce_weights).cuda()
		else:
			pos_weight = torch.ones([num_actor_class])*args.bce_pos_weight


		# Train loop
		for data in tqdm(dataloader_train):
			video_in = data['videos']
			if args.bg_mask:
				bg_seg_in = data['bg_seg']
			if args.obj_mask:
				obj_mask = data['obj_masks']
			if args.box:
				box_in = data['box']

			inputs = []
			bg_seg = []
			obj_mask_list = []
			for i in range(seq_len):
				inputs.append(video_in[i].to(args.device, dtype=torch.float32))
    
			if args.box:
				if isinstance(box_in,np.ndarray):
					boxes = torch.from_numpy(box_in).to(args.device, dtype=torch.float32)
				else:
					boxes = box_in.to(args.device, dtype=torch.float32)
			if args.bg_mask:
				for i in range(args.seq_len//args.mask_every_frame):
					bg_seg.append(bg_seg_in[i].to(args.device, dtype=torch.float32))
			if args.obj_mask:
				for i in range(args.seq_len):
					if i%args.mask_every_frame==0 or args.mask_every_frame==1:
						obj_mask_list.append(obj_mask[i//args.mask_every_frame].to(args.device, dtype=torch.float32))

			batch_size = inputs[0].shape[0]
			if ('slot' in args.model_name and not args.allocated_slot) or args.box:
				actor = data['actor'].to(args.device)
			else:
				actor = torch.FloatTensor(data['actor']).to(args.device)
			
			if args.pretrain == 'taco':
				ds_size = (32,96)
			else:
				ds_size = (28,28)

			optimizer.zero_grad()
			if args.bg_mask:
				h, w = bg_seg[0].shape[-2], bg_seg[0].shape[-1]
				bg_seg = torch.stack(bg_seg, 0)
				bg_seg = torch.permute(bg_seg, (1, 0, 2, 3)) #[batch, len, h, w]
				b, l, h, w = bg_seg.shape
				
				bg_seg = torch.reshape(bg_seg, (b*l, 1, h, w))
				bg_seg = F.interpolate(bg_seg, size=ds_size)
				bg_seg = torch.reshape(bg_seg, (b, l, ds_size[0], ds_size[1]))
			if args.obj_mask:
				obj_mask_list = torch.stack(obj_mask_list, 0)
				obj_mask_list = torch.permute(obj_mask_list, (1, 0, 2, 3, 4)) #[batch, len, n, h, w]
				b, l, n, h, w = obj_mask_list.shape

			# object-based models
			if args.box:
				_, pred_actor = model(inputs, boxes)

			else:
				if 'slot' in args.model_name or 'mvit' in args.model_name:
					pred_actor, attn = model(inputs)
				else:
					pred_actor = model(inputs)


			# hungarian matching
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
				if args.obj_mask:
					obj_bce = nn.BCELoss()
					loss_mask = 0.0
					b, l, n, h, w = attn.shape
					attn = torch.reshape(attn, (-1, 1 ,8, 24))
					attn = F.interpolate(attn, size=(32, 96))
					attn = torch.reshape(attn, (b, l , n, 32, 96))
					sup_idx = [1,3,5,7,9,11,13,15]
					attn = attn[:,sup_idx,:,:,:].reshape((b,8,n,32,96))
					b, seq, n_obj, h, w = obj_mask_list.shape
					mask_detach = attn.detach().flatten(3,4)
					mask_detach = mask_detach.cpu().numpy()
					mask_gt_np = obj_mask_list.flatten(3,4)
					mask_gt_np = mask_gt_np.detach().cpu().numpy()
					scores = np.zeros((b, 8, n, n_obj))
					for i in range(b):
						for j in range(8):
						    cross_entropy_cur = np.matmul(np.log( mask_detach[i,j]), mask_gt_np[i,j].T) + np.matmul(np.log(1 - mask_detach[i,j]), (1 - mask_gt_np[i,j]).T)
						    scores[i,j] += cross_entropy_cur
					for i in range(b):
						for j in range(8):
							matches = linear_sum_assignment(-scores[i,j])
							id_slot, id_gt = matches 
							loss_mask += obj_bce(attn[i,j,id_slot,:,:], obj_mask_list[i,j,id_gt,:,:])
					attn_loss = loss_mask

			elif 'slot' in args.model_name and args.allocated_slot:
				# pred_actor = pred_actor [1]
				actor_loss = bce(pred_actor, actor)
				if args.bg_slot and not args.bg_mask and args.action_attn_weight>0:
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
					action_attn_bce = nn.BCELoss()
					action_attn_loss = action_attn_bce(action_attn[class_idx], attn_gt[class_idx])
					attn_loss = args.action_attn_weight*action_attn_loss

					action_attn_pred = action_attn[class_idx] > 0.5
					# inter, union = inter_and_union(action_attn_pred.reshape(-1, h, w), attn_gt[class_idx].reshape(-1, h, w), 1, 0)
					# action_inter.update(inter)
					# action_union.update(union)

					attn_loss_epoch += float(attn_loss.item())
					action_attn_loss_epoch += float(action_attn_loss.item())
				elif args.obj_mask:
					obj_bce = nn.BCELoss()
					loss_mask = 0.0
					b, l, n, h, w = attn.shape
					attn = torch.reshape(attn, (-1, 1 ,8, 24))
					attn = F.interpolate(attn, size=(32, 96))
					attn = torch.reshape(attn, (b, l , n, 32, 96))
					sup_idx = [1,3,5,7,9,11,13,15]
					attn = attn[:,sup_idx,:,:,:].reshape((b,8,n,32,96))
					b, seq, n_obj, h, w = obj_mask_list.shape
					mask_detach = attn.detach().flatten(3,4)
					mask_detach = mask_detach.cpu().numpy()
					mask_gt_np = obj_mask_list.flatten(3,4)
					mask_gt_np = mask_gt_np.detach().cpu().numpy()
					scores = np.zeros((b, 8, n, n_obj))
					for i in range(b):
						for j in range(8):
						    cross_entropy_cur = np.matmul(np.log( mask_detach[i,j]), mask_gt_np[i,j].T) + np.matmul(np.log(1 - mask_detach[i,j]), (1 - mask_gt_np[i,j]).T)
						    scores[i,j] += cross_entropy_cur
					for i in range(b):
						for j in range(8):
							matches = linear_sum_assignment(-scores[i,j])
							id_slot, id_gt = matches 
							loss_mask += obj_bce(attn[i,j,id_slot,:,:], obj_mask_list[i,j,id_gt,:,:])
					attn_loss = loss_mask
					
				elif args.bg_slot and args.bg_mask and args.action_attn_weight>0. and args.bg_attn_weight>0.:
					b, l, n, h, w = attn.shape

					if args.bg_upsample != 1:
						attn = attn.reshape(-1, 1, h, w)
						attn = F.interpolate(attn, size=ds_size, mode='bilinear')
						_, _, h, w = attn.shape
						attn = attn.reshape(b, l, n, h, w)

					action_attn = attn[:, :, :num_actor_class, :, :]
					bg_attn = attn[:, ::args.mask_every_frame, -1, :, :].reshape(b, l//args.mask_every_frame, h, w)

					class_idx = actor == 0.0
					# bg_idx = torch.ones(b, dtype=torch.bool).cuda()
					# bg_idx = torch.reshape(bg_idx, (b, 1))
					class_idx = class_idx.view(b, num_actor_class, 1, 1, 1).repeat(1, 1, l, h, w)
					class_idx = torch.permute(class_idx, (0, 2, 1, 3, 4))

					attn_gt = torch.zeros([b, l, num_actor_class, h, w], dtype=torch.float32).cuda()

					attn_bce = nn.BCELoss()
					action_attn_loss = attn_bce(action_attn[class_idx], attn_gt[class_idx])

					bg_attn_loss = attn_bce(bg_attn, bg_seg)
					attn_loss = args.action_attn_weight*action_attn_loss + args.bg_attn_weight*bg_attn_loss


					attn_loss_epoch += float(attn_loss.item())
					action_attn_loss_epoch += float(action_attn_loss.item())
					bg_attn_loss_epoch += float(bg_attn_loss.item())
				elif args.bg_slot and args.bg_mask and args.bg_attn_weight>0. and args.action_attn_weight ==0 :
					b, l, n, h, w = attn.shape
					if args.bg_upsample != 1:
						attn = attn.reshape(-1, 1, h, w)
						attn = F.interpolate(attn, size=ds_size, mode='bilinear')
						_, _, h, w = attn.shape
						attn = attn.reshape(b, l, n, h, w)

					bg_attn = attn[:, ::args.mask_every_frame, -1, :, :].reshape(b, l//args.mask_every_frame, h, w)

					bg_bce = nn.BCELoss()
					bg_attn_loss = bg_bce(bg_attn, bg_seg)
					attn_loss = args.bg_attn_weight*bg_attn_loss

					attn_loss_epoch += float(attn_loss.item())
					bg_attn_loss_epoch += float(bg_attn_loss.item())

			else:
				bce = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight).cuda()
				actor_loss = bce(pred_actor, actor)

			if (args.bg_slot and args.action_attn_weight>0.) or (args.bg_mask and args.bg_attn_weight>0.) or args.obj_mask:
				loss = actor_loss + attn_loss
			else:
				loss = actor_loss
			if args.parallel:
				loss = loss.mean()
			loss.backward()
			loss_epoch += float(loss.item())

# ------------------------------------------------------------------------------------
			if ('slot' in args.model_name and not args.allocated_slot) or args.box:
				pred_actor = torch.nn.functional.softmax(pred_actor, dim=-1)
				_, pred_actor_idx = torch.max(pred_actor.data, -1)
				pred_actor_idx = pred_actor_idx.detach().cpu().numpy().astype(int)
				# f1_batch_new_pred_actor = []
				map_batch_new_pred_actor = []
				for i, b in enumerate(pred_actor_idx):
					# f1_new_pred = np.zeros(num_actor_class, dtype=int)
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
			actor_loss = actor_loss.mean()
			actor_loss_epoch += float(actor_loss.item())

			num_batches += 1
			optimizer.step()

			self.cur_iter += 1
		if scheduler is not None:
			scheduler.step()

		map_pred_actor_list = np.stack(map_pred_actor_list, axis=0)
		label_actor_list = np.stack(label_actor_list, axis=0)

		map_pred_actor_list = map_pred_actor_list.reshape((map_pred_actor_list.shape[0]*args.batch_size, num_actor_class))
		label_actor_list = label_actor_list.reshape((label_actor_list.shape[0]*args.batch_size, num_actor_class))

		mAP = average_precision_score(
					label_actor_list,
					map_pred_actor_list.astype(np.float32))

		loss_epoch = loss_epoch / num_batches	
		actor_loss_epoch = actor_loss_epoch / num_batches


		print('total loss')
		print(loss_epoch)

		if args.bg_slot and args.bg_mask:
			attn_loss_epoch = attn_loss_epoch / num_batches
			print('attn loss:')
			print(attn_loss_epoch)

			action_attn_loss_epoch = action_attn_loss_epoch /num_batches
			print('action_attn_loss')
			print(action_attn_loss_epoch)
			bg_attn_loss_epoch = bg_attn_loss_epoch / num_batches
			print('bg_attn_loss_epoch')
			print(bg_attn_loss_epoch)


		print('-'*20)
		print('actor loss:')
		print(actor_loss_epoch)

		print(f'(train) mAP of the actor: {mAP}')
		self.train_loss.append(loss_epoch)
		self.cur_epoch += 1
		

	def validate(self, model, dataloader, epoch, cam=False):
		model.eval()
		save_cp = False


		mask_bce = nn.BCELoss()
		if args.parallel:
			mask_bce = nn.DataParallel(mask_bce)
		mask_bce.cuda()

		if ('slot' in args.model_name and not args.allocated_slot) or args.box:
			ce_weights = torch.ones(num_actor_class+1)*args.ce_pos_weight
			ce_weights[-1] = self.args.ce_neg_weight
			instance_ce = nn.CrossEntropyLoss(reduction='mean', weight=ce_weights)
			if args.parallel:
				instance_ce = nn.DataParallel(instance_ce)
			instance_ce = instance_ce.cuda()
		elif 'slot' in args.model_name and args.allocated_slot:
			bce_weights = torch.ones([num_actor_class])*args.bce_pos_weight
			bce = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=bce_weights)
			if args.parallel:
				bce = nn.DataParallel(bce)
			bce = bce.cuda()
		else:
			bce_weights = torch.ones([num_actor_class])*args.bce_pos_weight
			bce = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=bce_weights)
			if args.parallel:
				bce = nn.DataParallel(bce)
			bce = bce.cuda()
		with torch.no_grad():	
			num_batches = 0
			total_loss = 0.
			loss = 0.
			attn_loss_epoch= 0.
			action_attn_loss_epoch = 0.
			bg_attn_loss_epoch = 0.
			total_actor = 0

			correct_actor = 0
			label_actor_list = []
			map_pred_actor_list = []

			action_inter = AverageMeter()
			action_union = AverageMeter()
			bg_inter = AverageMeter()
			bg_union = AverageMeter()

			for batch_num, data in enumerate(tqdm(dataloader)):
				video_in = data['videos']

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
				if args.bg_mask:
					for i in range(args.seq_len//args.mask_every_frame):
						bg_seg.append(bg_seg_in[i].to(args.device, dtype=torch.float32))

				batch_size = inputs[0].shape[0]
				if ('slot' in args.model_name and not args.allocated_slot) or args.box:
					actor = data['actor'].to(args.device)
				else:
					actor = torch.FloatTensor(data['actor']).to(args.device)
				if args.pretrain == 'taco':
					ds_size = (32,96)
				else:
					ds_size = (28,28)
				if args.bg_mask:
					h, w = bg_seg[0].shape[-2], bg_seg[0].shape[-1]
					bg_seg = torch.stack(bg_seg, 0)
					bg_seg = torch.permute(bg_seg, (1, 0, 2, 3)) #[batch, len, h, w]
					b, l, h, w = bg_seg.shape
					# ds_size = (model.resolution[0]*args.bg_upsample, model.resolution[1]*args.bg_upsample)
					bg_seg = torch.reshape(bg_seg, (b*l, 1, h, w))
					bg_seg = F.interpolate(bg_seg, size=ds_size)
					bg_seg = torch.reshape(bg_seg, (b, l, ds_size[0], ds_size[1]))
				if ('slot' in args.model_name) or args.box or 'mvit' in args.model_name:
					if args.box:
						_, pred_actor = model(inputs, boxes)
					else:
						pred_actor, attn = model(inputs)
				else:
					pred_actor = model(inputs)


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
					# pred_actor = pred_actor[1]
					actor_loss = bce(pred_actor, actor)
					if args.bg_slot and not args.bg_mask and args.action_attn_weight >0:
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

					elif args.bg_slot and args.bg_mask and args.action_attn_weight >0. and args.bg_attn_weight>0.:
						b, l, n, h, w = attn.shape

						if args.bg_upsample != 1:
							attn = attn.reshape(-1, 1, h, w)
							attn = F.interpolate(attn, size=ds_size, mode='bilinear')
							_, _, h, w = attn.shape
							attn = attn.reshape(b, l, n, h, w)

						action_attn = attn[:, :, :num_actor_class, :, :]
						bg_attn = attn[:, ::args.mask_every_frame, -1, :, :].reshape(b, l//args.mask_every_frame, h, w)

						class_idx = actor == 0.0
						# bg_idx = torch.ones(b, dtype=torch.bool).cuda()
						# bg_idx = torch.reshape(bg_idx, (b, 1))
						# class_idx = torch.cat((class_idx, bg_idx), -1)
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
					elif args.bg_slot and args.bg_mask and args.bg_attn_weight>0. and args.action_attn_weight == 0:
						b, l, n, h, w = attn.shape

						if args.bg_upsample != 1:
							attn = attn.reshape(-1, 1, h, w)
							attn = F.interpolate(attn, size=ds_size, mode='bilinear')
							_, _, h, w = attn.shape
							attn = attn.reshape(b, l, n, h, w)

						bg_attn = attn[:, ::args.mask_every_frame, -1, :, :].reshape(b, l//args.mask_every_frame, h, w)

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
				
				if (args.bg_slot and args.action_attn_weight>0.) or (args.bg_mask and args.bg_attn_weight>0.):
					loss = actor_loss + attn_loss
				else:
					loss = actor_loss
				
				num_batches += 1
				total_loss += float(loss.item())
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

			if (args.bg_slot and args.action_attn_weight>0.) or (args.bg_mask and args.bg_attn_weight>0.):
				attn_loss_epoch = attn_loss_epoch / num_batches
				print('attn loss:')
				print(attn_loss_epoch)
				if args.bg_slot:
					action_attn_loss_epoch = action_attn_loss_epoch /num_batches
					print('action_attn_loss')
					print(action_attn_loss_epoch)
				if args.bg_mask:
					bg_attn_loss_epoch = bg_attn_loss_epoch / num_batches
					print('bg_attn_loss_epoch')
					print(bg_attn_loss_epoch)
				
			if args.bg_mask and args.bg_slot and args.action_attn_weight >0 and args.bg_attn_weight>0:
				iou = action_inter.sum / (action_union.sum + 1e-10)
				for i, val in enumerate(iou):
					print('Action IoU {0}: {1:.2f}'.format(i, val * 100))

				iou = bg_inter.sum / (bg_union.sum + 1e-10)
				for i, val in enumerate(iou):
					print('BG IoU {0}: {1:.2f}'.format(i, val * 100))

			map_pred_actor_list = np.stack(map_pred_actor_list, axis=0)
			label_actor_list = np.stack(label_actor_list, axis=0)
			
			map_pred_actor_list = map_pred_actor_list.reshape((map_pred_actor_list.shape[0], num_actor_class))
			label_actor_list = label_actor_list.reshape((label_actor_list.shape[0], num_actor_class))
			map_pred_actor_list = np.array(map_pred_actor_list)
			label_actor_list = np.array(label_actor_list)
			
			# index
			c_label = [3, 4, 6, 7, 8, 9, 10, 12, 14, 15, 23, 25]
			k_label = [24, 32, 34]
			c_group_label = [0, 27, 30, 33]
			k_group_label = []
			p_label = [2, 5, 11, 13, 16, 18, 21, 31]
			p_group_label = [1, 17, 19, 20, 22, 26, 28, 29]

			mAP = average_precision_score(
					label_actor_list,
					map_pred_actor_list.astype(np.float32),
					)
			c_mAP = average_precision_score(
					label_actor_list[:, c_label],
			        map_pred_actor_list[:, c_label].astype(np.float32)
			        )
			b_mAP = average_precision_score(
					label_actor_list[:, k_label],
			        map_pred_actor_list[:, k_label].astype(np.float32)
			        )
			p_mAP = average_precision_score(
					label_actor_list[:, p_label],
			        map_pred_actor_list[:, p_label].astype(np.float32),
			        )
			group_c_mAP = average_precision_score(
					label_actor_list[:, c_group_label],
			        map_pred_actor_list[:, c_group_label].astype(np.float32)
			        )
			# group_b_mAP = average_precision_score(
			# 		label_actor_list[:, 36:48],
			#         map_pred_actor_list[:, 36:48].astype(np.float32)
			#         )
			group_p_mAP = average_precision_score(
					label_actor_list[:,p_group_label],
			        map_pred_actor_list[:, p_group_label].astype(np.float32),
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
			# print(f'(val) mAP of the b+: {group_b_mAP}')
			print(f'(val) mAP of the p+: {group_p_mAP}')
			print(f'best mAP : {self.best_mAP}')
			if mAP > self.best_mAP:
				self.best_mAP = mAP
				save_cp = True
				tmp_s = []
				tmp_s.append(f'(val) mAP of the actor: {mAP}')
				tmp_s.append(f'(val) mAP of the c: {c_mAP}')
				tmp_s.append(f'(val) mAP of the b: {b_mAP}')
				tmp_s.append(f'(val) mAP of the p: {p_mAP}')
				tmp_s.append(f'(val) mAP of the c+: {group_c_mAP}')
				# print(f'(val) mAP of the b+: {group_b_mAP}')
				tmp_s.append(f'(val) mAP of the p+: {group_p_mAP}')
				self.best_performance = tmp_s

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
				# f.write('mAP of b+: %.4f' % group_b_mAP)
				# f.write('\n')
				f.write('mAP of p+: %.4f' % group_p_mAP)
				f.write('\n')
				f.write('*'*5 + '\n')
				f.write('per class: + \n')
				for ap in mAP_per_class.tolist():
					f.write("%.4f " % ap)
				f.write('\n')
				f.write('*'*15 + '\n')

			total_loss = total_loss / float(num_batches)
			tqdm.write(f'Epoch {self.cur_epoch:03d}, Batch {batch_num:03d}:' + f' Loss: {total_loss:3.3f}')

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

		# Save ckpt for every epoch
		# torch.save(model.state_dict(), os.path.join(logdir, 'model_%d.pth'%self.cur_epoch))

		# Save the recent model/optimizer states
		# torch.save(model.state_dict(), os.path.join(logdir, 'model.pth'))
		# torch.save(optimizer.state_dict(), os.path.join(logdir, 'recent_optim.pth'))

		# Log other data corresponding to the recent model
		# with open(os.path.join(args.logdir, 'recent.log'), 'w') as f:
		# 	f.write(json.dumps(log_table))

		# tqdm.write('====== Saved recent model ======>')
		
		if save_best:
			torch.save(model.state_dict(), os.path.join(logdir, 'best_model.pth'))
			# torch.save(optimizer.state_dict(), os.path.join(logdir, 'best_optim.pth'))
			tqdm.write('====== Overwrote best model ======>')

torch.cuda.empty_cache() 
seq_len = args.seq_len

num_ego_class = 0
num_actor_class = 35


# Data
train_set = oats.OATS(args=args)
val_set = oats.OATS(args=args, training=False)
label_stat = []
for i in range(6):
	label_stat.append({})
	for k in train_set.label_stat[i].keys():
		label_stat[i][k] = train_set.label_stat[i][k] + val_set.label_stat[i][k]
	
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
	
dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
dataloader_val = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
# Model
model = generate_model(args, num_ego_class, num_actor_class).cuda()
if args.pretrain != '' :
        model_path = os.path.join(args.cp)
        if args.pretrain == 'taco':
            checkpoint = torch.load(model_path)
            checkpoint = {k: v for k, v in checkpoint.items() if (k in checkpoint and 'fc' not in k)}
            model.load_state_dict(checkpoint, strict=False)
            if 'slot' in args.model_name:
                model.slot_attention.extract_slots_for_oats()

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
trainer = Engine(args)

result_list = []
if not args.test:
	for epoch in range(trainer.cur_epoch, args.epochs): 
		trainer.train(model, optimizer, epoch, scheduler=scheduler)
		if (epoch % args.val_every == 0 or epoch == args.epochs-1): 
				is_best, res = trainer.validate(model, dataloader_val, None)
				# trainer.validate(dataloader_val_train, None)
				trainer.save(is_best)
				result_list.append(res)

else:
	trainer.validate(cam=cam)
print("***************** Best map *****************")
print(f'best mAP : {trainer.best_mAP}')
for s in trainer.best_performance:
    print(s)
plot_result(np.array(result_list))
