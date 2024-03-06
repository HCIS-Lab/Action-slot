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

from PIL import Image

# import cnnlstm_backbone
from model import generate_model
from utils import *
from torchvision import models
import matplotlib.image
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from scipy.optimize import linear_sum_assignment
from parser import get_parser
import math



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

	def __init__(self, args, cur_epoch=0, cur_iter=0, ego_weight=1):
		self.cur_epoch = cur_epoch
		self.cur_iter = cur_iter
		self.bestval_epoch = cur_epoch
		self.train_loss = []
		self.val_loss = []
		self.bestval = 1e10
		self.best_mAP = 1e-5
		self.ego_weight = ego_weight
		self.args = args

	def train(self, model, optimizer, epoch, model_name='', scheduler=None, ce_weight=5):
		if scheduler is not None:
			print(f"Current epoch: {epoch}, lr: {scheduler.get_last_lr()}")
		else:
			print(f"Current epoch: {epoch}")
		loss_epoch = 0.
		ego_loss_epoch = 0.
		seg_loss_epoch = 0.
		attn_loss_epoch= 0.
		action_attn_loss_epoch = 0.
		bg_attn_loss_epoch = 0.
		actor_loss_epoch = 0.
		num_batches = 0
		correct_ego = 0
		total_ego = 0
		action_inter = AverageMeter()
		action_union = AverageMeter()
		bg_inter = AverageMeter()
		bg_union = AverageMeter()


		label_actor_list = []
		map_pred_actor_list = []
		f1_pred_actor_list = []
  
		ego_ce = nn.CrossEntropyLoss(reduction='mean')
		seg_ce = nn.CrossEntropyLoss(reduction='mean')
		model.train()
		if args.parallel:
			model = nn.DataParallel(model)
			ego_ce, seg_ce = nn.DataParallel(ego_ce), nn.DataParallel(ego_ce)
		ego_ce, seg_ce = ego_ce.cuda(), seg_ce.cuda()

		empty_weight = torch.ones(num_actor_class+1)*ce_weight
		empty_weight[-1] = self.args.empty_weight
		if ('slot' in model_name and not self.args.fix_slot) or self.args.box:
			slot_ce = nn.CrossEntropyLoss(reduction='mean', weight=empty_weight).cuda()
		elif 'slot' in model_name and args.fix_slot:
			pos_weight = torch.ones([num_actor_class])*5
			slot_ce = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight).cuda()
		else:
			pos_weight = torch.ones([num_actor_class])*args.weight


		# Train loop
		for data in tqdm(dataloader_train):
			# create batch and move to GPU
			fronts_in = data['fronts']
			if args.seg:
				seg_front_in = data['seg_front']
			if args.obj_mask:
				obj_mask = data['obj_masks']
			if args.box:
				box_in = data['box']

			inputs = []
			seg_front = []
			obj_mask_list = []
			for i in range(seq_len):
				inputs.append(fronts_in[i].to(args.device, dtype=torch.float32))
    
			if args.box:
				if isinstance(box_in,np.ndarray):
					boxes = torch.from_numpy(box_in).to(args.device, dtype=torch.float32)
				else:
					boxes = box_in.to(args.device, dtype=torch.float32)
			if args.seg:
				for i in range(args.seq_len):
					seg_front.append(seg_front_in[i].to(args.device, dtype=torch.float32))
			if args.obj_mask:
				for i in range(args.seq_len):
					if i%2:
						obj_mask_list.append(obj_mask[i//2].to(args.device, dtype=torch.float32))

			batch_size = inputs[0].shape[0]
			ego = data['ego'].cuda()
			if ('slot' in model_name and not args.fix_slot) or args.box:
				actor = data['actor'].to(args.device)
			else:
				actor = torch.FloatTensor(data['actor']).to(args.device)
			
			optimizer.zero_grad()
			if args.seg:
				# print(seg_front[0].shape)
				h, w = seg_front[0].shape[-2], seg_front[0].shape[-1]
				# print(h, w)
				seg_front = torch.stack(seg_front, 0)
				# print(seg_front.shape)
				seg_front = torch.permute(seg_front, (1, 0, 2, 3)) #[batch, len, h, w]
				b, l, h, w = seg_front.shape
				ds_size = (model.resolution[0]*args.upsample, model.resolution[1]*args.upsample)
				# seg_front = torch.reshape(seg_front, (b*l, 1, h, w))
				# seg_front = F.interpolate(seg_front, size=ds_size)
				# seg_front = torch.reshape(seg_front, (b, l, ds_size[0], ds_size[1]))
			if args.obj_mask:
				obj_mask_list = torch.stack(obj_mask_list, 0)
				obj_mask_list = torch.permute(obj_mask_list, (1, 0, 2, 3, 4)) #[batch, len, n, h, w]
				b, l, n, h, w = obj_mask_list.shape
				# obj_mask_list = torch.reshape(obj_mask_list, (b*l*n, 1, h, w))
				# obj_mask_list = F.interpolate(obj_mask_list, size=(8,24))
				# obj_mask_list = torch.reshape(obj_mask_list, (b, l, n, 8, 24))
			if args.box:
				pred_ego, pred_actor = model(inputs, boxes)
			else:
				if 'slot' in model_name or 'mvit' in model_name:
					pred_ego, pred_actor, attn = model(inputs)
				else:
					pred_ego, pred_actor = model(inputs)

			ego_loss = ego_ce(pred_ego, ego)

			# hungarian matching
			if ('slot' in model_name and not args.fix_slot) or args.box:
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
				actor_loss = slot_ce(pred_actor.transpose(1, 2), target_classes)

				if args.obj_mask:
					bcecriterion = nn.BCELoss()
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
					# out_attn = attn.flatten(3,4).clone().detach().cpu().numpy()
					# obj_mask_gt = obj_mask_list.clone().cpu().numpy()
					# out_attn = out_attn.flatten(3,4)
					# obj_mask_np = obj_mask_gt.flatten(3,4)
					scores = np.zeros((b, 8, n, n_obj))
					for i in range(b):
						for j in range(8):
						    cross_entropy_cur = np.matmul(np.log( mask_detach[i,j]), mask_gt_np[i,j].T) + np.matmul(np.log(1 - mask_detach[i,j]), (1 - mask_gt_np[i,j]).T)
						    scores[i,j] += cross_entropy_cur
					for i in range(b):
						for j in range(8):
							matches = linear_sum_assignment(-scores[i,j])
							id_slot, id_gt = matches 
							loss_mask += bcecriterion(attn[i,j,id_slot,:,:], obj_mask_list[i,j,id_gt,:,:])
					attn_loss = loss_mask

			elif 'slot' in model_name and args.fix_slot:
				actor_loss = slot_ce(pred_actor, actor)
				if args.mask and not args.seg and args.action_attn_weight>0:
					b, l, n, h, w = attn.shape
					if args.upsample != 1.0:
						attn = attn.reshape(-1, 1, h, w)
						attn = F.interpolate(attn, size=ds_size, mode='bilinear')
						_, _, h, w = attn.shape
						attn = attn.reshape(b, l, n, h, w)
					action_attn = attn[:, :, :num_actor_class, :, :]

					class_idx = actor == 0.0
					class_idx = class_idx.view(b, num_actor_class, 1, 1, 1).repeat(1, 1, l, h, w)
					class_idx = torch.permute(class_idx, (0, 2, 1, 3, 4))

					attn_gt = torch.zeros([b, l, num_actor_class, h, w], dtype=torch.float32).cuda()
					bcecriterion = nn.BCELoss()
					action_attn_loss = bcecriterion(action_attn[class_idx], attn_gt[class_idx])
					attn_loss = args.action_attn_weight*action_attn_loss

					action_attn_pred = action_attn[class_idx] > 0.5
					inter, union = inter_and_union(action_attn_pred.reshape(-1, h, w), attn_gt[class_idx].reshape(-1, h, w), 1, 0)
					action_inter.update(inter)
					action_union.update(union)

					attn_loss_epoch += float(attn_loss.item())
					action_attn_loss_epoch += float(action_attn_loss.item())
				if args.obj_mask:
					bcecriterion = nn.BCELoss()
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
					# out_attn = attn.flatten(3,4).clone().detach().cpu().numpy()
					# obj_mask_gt = obj_mask_list.clone().cpu().numpy()
					# out_attn = out_attn.flatten(3,4)
					# obj_mask_np = obj_mask_gt.flatten(3,4)
					scores = np.zeros((b, 8, n, n_obj))
					for i in range(b):
						for j in range(8):
						    cross_entropy_cur = np.matmul(np.log( mask_detach[i,j]), mask_gt_np[i,j].T) + np.matmul(np.log(1 - mask_detach[i,j]), (1 - mask_gt_np[i,j]).T)
						    scores[i,j] += cross_entropy_cur
					for i in range(b):
						for j in range(8):
							matches = linear_sum_assignment(-scores[i,j])
							id_slot, id_gt = matches 
							loss_mask += bcecriterion(attn[i,j,id_slot,:,:], obj_mask_list[i,j,id_gt,:,:])
					attn_loss = loss_mask
					
				elif args.mask and args.seg and args.action_attn_weight>0. and args.bg_attn_weight>0.:
					b, l, n, h, w = attn.shape

					if args.upsample != 1.0:
						attn = attn.reshape(-1, 1, h, w)
						attn = F.interpolate(attn, size=ds_size, mode='bilinear')
						_, _, h, w = attn.shape
						attn = attn.reshape(b, l, n, h, w)

					action_attn = attn[:, :, :num_actor_class, :, :]
					bg_attn = attn[:, :, -1, :, :].reshape(b, l, h, w)

					class_idx = actor == 0.0
					bg_idx = torch.ones(b, dtype=torch.bool).cuda()
					bg_idx = torch.reshape(bg_idx, (b, 1))
					class_idx = class_idx.view(b, num_actor_class, 1, 1, 1).repeat(1, 1, l, h, w)
					class_idx = torch.permute(class_idx, (0, 2, 1, 3, 4))

					attn_gt = torch.zeros([b, l, num_actor_class, h, w], dtype=torch.float32).cuda()

					bcecriterion = nn.BCELoss()
					action_attn_loss = bcecriterion(action_attn[class_idx], attn_gt[class_idx])

					bg_attn_loss = bcecriterion(bg_attn, seg_front)
					attn_loss = args.action_attn_weight*action_attn_loss + args.bg_attn_weight*bg_attn_loss

					action_attn_pred = action_attn[class_idx] > 0.5
					inter, union = inter_and_union(action_attn_pred.reshape(-1, h, w), attn_gt[class_idx].reshape(-1, h, w), 1, 0)
					action_inter.update(inter)
					action_union.update(union)

					bg_attn_pred = bg_attn > 0.5
					inter, union = inter_and_union(bg_attn_pred.reshape(-1, h, w), seg_front.reshape(-1, h, w), 1, 1)
					bg_inter.update(inter)
					bg_union.update(union)

					attn_loss_epoch += float(attn_loss.item())
					action_attn_loss_epoch += float(action_attn_loss.item())
					bg_attn_loss_epoch += float(bg_attn_loss.item())
				elif not args.mask and args.seg and args.bg_attn_weight>0.:
						b, l, n, h, w = attn.shape

						if args.upsample != 1.0:
							attn = attn.reshape(-1, 1, h, w)
							attn = F.interpolate(attn, size=ds_size, mode='bilinear')
							_, _, h, w = attn.shape
							attn = attn.reshape(b, l, n, h, w)

						bg_attn = attn[:, :, -1, :, :].reshape(b, l, h, w)

						bg_idx = torch.ones(b, dtype=torch.bool).cuda()
						bg_idx = torch.reshape(bg_idx, (b, 1))

						bcecriterion = nn.BCELoss()
						bg_attn_loss = bcecriterion(bg_attn, seg_front)
						attn_loss = args.bg_attn_weight*bg_attn_loss

						bg_attn_pred = bg_attn > 0.5
						inter, union = inter_and_union(bg_attn_pred.reshape(-1, h, w), seg_front.reshape(-1, h, w), 1, 1)
						bg_inter.update(inter)
						bg_union.update(union)

						attn_loss_epoch += float(attn_loss.item())
						bg_attn_loss_epoch += float(bg_attn_loss.item())

			else:
				bce = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight).cuda()
				actor_loss = bce(pred_actor, actor)

			if (args.mask and args.action_attn_weight>0.) or (args.seg and args.bg_attn_weight>0.) or args.obj_mask:
				loss = actor_loss + 1.5*ego_loss + attn_loss
			else:
				loss = actor_loss + args.ego_weight*ego_loss
			if args.parallel:
				loss = loss.mean()
			loss.backward()
			loss_epoch += float(loss.item())

# ------------------------------------------------------------------------------------
			_, pred_ego = torch.max(pred_ego.data, 1)
			total_ego += ego.size(0)
			correct_ego += (pred_ego == ego).sum().item()

			if ('slot' in model_name and not args.fix_slot) or args.box:
				pred_actor = torch.nn.functional.softmax(pred_actor, dim=-1)
				_, pred_actor_idx = torch.max(pred_actor.data, -1)
				pred_actor_idx = pred_actor_idx.detach().cpu().numpy().astype(int)
				f1_batch_new_pred_actor = []
				map_batch_new_pred_actor = []
				for i, b in enumerate(pred_actor_idx):
					f1_new_pred = np.zeros(num_actor_class, dtype=int)
					map_new_pred = np.zeros(num_actor_class, dtype=np.float32)+1e-5
					for j, pred in enumerate(b):
						if pred != num_actor_class:
							f1_new_pred[pred] = 1
							if pred_actor[i, j, pred] > map_new_pred[pred]:
								map_new_pred[pred] = pred_actor[i, j, pred]
					f1_batch_new_pred_actor.append(f1_new_pred)
					map_batch_new_pred_actor.append(map_new_pred)
				f1_batch_new_pred_actor = np.array(f1_batch_new_pred_actor)
				map_batch_new_pred_actor = np.array(map_batch_new_pred_actor)
				f1_pred_actor_list.append(f1_batch_new_pred_actor)
				map_pred_actor_list.append(map_batch_new_pred_actor)
				label_actor_list.append(data['slot_eval_gt'])
			else:
				pred_actor = torch.sigmoid(pred_actor)
				f1_pred_actor = pred_actor > 0.5
				f1_pred_actor = f1_pred_actor.float()
				map_pred_actor_list.append(pred_actor.detach().cpu().numpy())
				f1_pred_actor_list.append(f1_pred_actor.detach().cpu().numpy())
				label_actor_list.append(actor.detach().cpu().numpy())
			actor_loss, ego_loss = actor_loss.mean(), ego_loss.mean()
			actor_loss_epoch += float(actor_loss.item())
			ego_loss_epoch += float(ego_loss.item())

			num_batches += 1
			optimizer.step()
			# writer.add_scalar('train_loss', loss.item(), self.cur_iter)

			self.cur_iter += 1
		if scheduler is not None:
			scheduler.step()

		f1_pred_actor_list = np.stack(f1_pred_actor_list, axis=0)
		map_pred_actor_list = np.stack(map_pred_actor_list, axis=0)
		label_actor_list = np.stack(label_actor_list, axis=0)

		f1_pred_actor_list = f1_pred_actor_list.reshape((f1_pred_actor_list.shape[0]*8, num_actor_class))
		map_pred_actor_list = map_pred_actor_list.reshape((map_pred_actor_list.shape[0]*8, num_actor_class))
		label_actor_list = label_actor_list.reshape((label_actor_list.shape[0]*8, num_actor_class))



		if ('slot' in model_name and not args.fix_slot) or args.box:
			mean_f1 = f1_score(
					label_actor_list.astype('int64'), 
					f1_pred_actor_list.astype('int64'),
					average='samples',
					labels=[i for i in range(num_actor_class)],
					zero_division=0)
		else:
			mean_f1 = f1_score(
					label_actor_list.astype('int64'), 
					f1_pred_actor_list.astype('int64'),
					average='samples',
					zero_division=0)
		mAP = average_precision_score(
					label_actor_list,
					map_pred_actor_list.astype(np.float32))

		loss_epoch = loss_epoch / num_batches	
		actor_loss_epoch = actor_loss_epoch / num_batches
		ego_loss_epoch = ego_loss_epoch / num_batches

		print(f'acc of the ego: {correct_ego/total_ego}')

		print('total loss')
		print(loss_epoch)

		if args.mask and args.seg:
			attn_loss_epoch = attn_loss_epoch / num_batches
			print('attn loss:')
			print(attn_loss_epoch)
			if args.mask:
				action_attn_loss_epoch = action_attn_loss_epoch /num_batches
				print('action_attn_loss')
				print(action_attn_loss_epoch)
			if args.seg:
				bg_attn_loss_epoch = bg_attn_loss_epoch / num_batches
				print('bg_attn_loss_epoch')
				print(bg_attn_loss_epoch)

			iou = action_inter.sum / (action_union.sum + 1e-10)
			for i, val in enumerate(iou):
				print('Action IoU {0}: {1:.2f}'.format(i, val * 100))

			iou = bg_inter.sum / (bg_union.sum + 1e-10)
			for i, val in enumerate(iou):
				print('BG IoU {0}: {1:.2f}'.format(i, val * 100))

		print('-'*20)
		print('actor loss:')
		print(actor_loss_epoch)
		print('ego loss:')
		print(ego_loss_epoch)
		print(f'(train) f1 of the actor: {mean_f1}')
		print(f'(train) mAP of the actor: {mAP}')
		# writer.add_scalar('train_f1', mean_f1, epoch)
		self.train_loss.append(loss_epoch)
		self.cur_epoch += 1
		

	def validate(self, model, dataloader, epoch, cam=False, model_name='', ce_weight=5):
		model.eval()
		save_cp = False
		ego_ce = nn.CrossEntropyLoss(reduction='mean')
		seg_ce = nn.CrossEntropyLoss(reduction='mean')
		if args.parallel:
			ego_ce, seg_ce = nn.DataParallel(ego_ce), nn.DataParallel(ego_ce)
		ego_ce, seg_ce = ego_ce.cuda(), seg_ce.cuda()

		bcecriterion = nn.BCELoss()
		if args.parallel:
			bcecriterion = nn.DataParallel(bcecriterion)
		bcecriterion.cuda()

		if ('slot' in model_name and not args.fix_slot) or args.box:
			empty_weight = torch.ones(num_actor_class+1)*ce_weight
			empty_weight[-1] = self.args.empty_weight
			slot_ce = nn.CrossEntropyLoss(reduction='mean', weight=empty_weight)
			if args.parallel:
				slot_ce = nn.DataParallel(slot_ce)
			slot_ce = slot_ce.cuda()
		elif 'slot' in model_name and args.fix_slot:
			pos_weight = torch.ones([num_actor_class])*args.weight
			slot_ce = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
			if args.parallel:
				slot_ce = nn.DataParallel(slot_ce)
			slot_ce = slot_ce.cuda()
		else:
			pos_weight = torch.ones([num_actor_class])*args.weight
			bce = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
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

				if args.seg:
					seg_front_in = data['seg_front']
				if args.box:
					box_in = data['box']
				inputs = []
				seg_front = []

				for i in range(seq_len):

					inputs.append(fronts_in[i].to(args.device, dtype=torch.float32))
     
				if args.box:
					if isinstance(box_in,np.ndarray):
						boxes = torch.from_numpy(box_in).to(args.device, dtype=torch.float32)
					else:
						boxes = box_in.to(args.device, dtype=torch.float32)
				if args.seg:
					for i in range(args.seq_len):
						seg_front.append(seg_front_in[i].to(args.device, dtype=torch.float32))

				batch_size = inputs[0].shape[0]
				ego = data['ego'].to(args.device)
				if ('slot' in model_name and not args.fix_slot) or args.box:
					actor = data['actor'].to(args.device)
				else:
					actor = torch.FloatTensor(data['actor']).to(args.device)

				if args.seg:
					h, w = seg_front[0].shape[-2], seg_front[0].shape[-1]
					seg_front = torch.stack(seg_front, 0)
					seg_front = torch.permute(seg_front, (1, 0, 2, 3)) #[batch, len, h, w]
					b, l, h, w = seg_front.shape
					ds_size = (model.resolution[0]*args.upsample, model.resolution[1]*args.upsample)
					seg_front = torch.reshape(seg_front, (b*l, 1, h, w))
					seg_front = F.interpolate(seg_front, size=ds_size)
					seg_front = torch.reshape(seg_front, (b, l, ds_size[0], ds_size[1]))
				if ('slot' in model_name) or args.box or 'mvit' in model_name:
					if args.box:
						pred_ego, pred_actor = model(inputs, boxes)
					else:
						pred_ego, pred_actor, attn = model(inputs)
				else:
					pred_ego, pred_actor = model(inputs)

				ego_loss = ego_ce(pred_ego, ego)

				if ('slot' in model_name and not args.fix_slot) or args.box:
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
					actor_loss = slot_ce(pred_actor.transpose(1, 2), target_classes)

				elif 'slot' in model_name and args.fix_slot:
					actor_loss = slot_ce(pred_actor, actor)
					if args.mask and not args.seg and args.action_attn_weight >0:
						b, l, n, h, w = attn.shape
						if args.upsample != 1.0:
							attn = attn.reshape(-1, 1, h, w)
							attn = F.interpolate(attn, size=ds_size, mode='bilinear')
							_, _, h, w = attn.shape
							attn = attn.reshape(b, l, n, h, w)
						action_attn = attn[:, :, :num_actor_class, :, :]
						class_idx = actor == 0.0
						class_idx = class_idx.view(b, num_actor_class, 1, 1, 1).repeat(1, 1, l, h, w)
						class_idx = torch.permute(class_idx, (0, 2, 1, 3, 4))

						attn_gt = torch.zeros([b, l, num_actor_class, h, w], dtype=torch.float32).cuda()
						action_attn_loss = bcecriterion(action_attn[class_idx], attn_gt[class_idx])
						attn_loss = args.action_attn_weight*action_attn_loss

						action_attn_pred = action_attn[class_idx] > 0.5
						inter, union = inter_and_union(action_attn_pred.reshape(-1, h, w), attn_gt[class_idx].reshape(-1, h, w), 1, 0)
						action_inter.update(inter)
						action_union.update(union)

						attn_loss_epoch += float(attn_loss.item())
						action_attn_loss_epoch += float(action_attn_loss.item())

					elif args.mask and args.seg and args.action_attn_weight >0. and args.bg_attn_weight>0.:
						b, l, n, h, w = attn.shape

						if args.upsample != 1.0:
							attn = attn.reshape(-1, 1, h, w)
							attn = F.interpolate(attn, size=ds_size, mode='bilinear')
							_, _, h, w = attn.shape
							attn = attn.reshape(b, l, n, h, w)

						action_attn = attn[:, :, :num_actor_class, :, :]
						bg_attn = attn[:, :, -1, :, :].reshape(b, l, h, w)

						class_idx = actor == 0.0
						bg_idx = torch.ones(b, dtype=torch.bool).cuda()
						bg_idx = torch.reshape(bg_idx, (b, 1))
						# class_idx = torch.cat((class_idx, bg_idx), -1)
						class_idx = class_idx.view(b, num_actor_class, 1, 1, 1).repeat(1, 1, l, h, w)
						class_idx = torch.permute(class_idx, (0, 2, 1, 3, 4))

						attn_gt = torch.zeros([b, l, num_actor_class, h, w], dtype=torch.float32).cuda()
						# seg_front = torch.reshape(seg_front, (b, l, 1, h, w))
						# attn_gt = torch.cat((attn_gt, seg_front), 2)
						action_attn_loss = bcecriterion(action_attn[class_idx], attn_gt[class_idx])
						bg_attn_loss = bcecriterion(bg_attn, seg_front)
						attn_loss = 1*action_attn_loss + 2*bg_attn_loss

						action_attn_pred = action_attn[class_idx] > 0.5
						inter, union = inter_and_union(action_attn_pred.reshape(-1, h, w), attn_gt[class_idx].reshape(-1, h, w), 1, 0)
						action_inter.update(inter)
						action_union.update(union)

						bg_attn_pred = bg_attn > 0.5
						inter, union = inter_and_union(bg_attn_pred, seg_front, 1, 1)
						bg_inter.update(inter)
						bg_union.update(union)

						attn_loss_epoch += float(attn_loss.item())
						action_attn_loss_epoch += float(action_attn_loss.item())
						bg_attn_loss_epoch += float(bg_attn_loss.item())
					elif not args.mask and args.seg and args.bg_attn_weight>0.:
						b, l, n, h, w = attn.shape

						if args.upsample != 1.0:
							attn = attn.reshape(-1, 1, h, w)
							attn = F.interpolate(attn, size=ds_size, mode='bilinear')
							_, _, h, w = attn.shape
							attn = attn.reshape(b, l, n, h, w)

						bg_attn = attn[:, :, -1, :, :].reshape(b, l, h, w)

						bg_idx = torch.ones(b, dtype=torch.bool).cuda()
						bg_idx = torch.reshape(bg_idx, (b, 1))

						bcecriterion = nn.BCELoss()
						bg_attn_loss = bcecriterion(bg_attn, seg_front)
						attn_loss = args.bg_attn_weight*bg_attn_loss

						bg_attn_pred = bg_attn > 0.5
						inter, union = inter_and_union(bg_attn_pred.reshape(-1, h, w), seg_front.reshape(-1, h, w), 1, 1)
						bg_inter.update(inter)
						bg_union.update(union)

						attn_loss_epoch += float(attn_loss.item())
						bg_attn_loss_epoch += float(bg_attn_loss.item())

				else:
					actor_loss = bce(pred_actor, actor)
				
				if (args.mask and args.action_attn_weight>0.) or (args.seg and args.bg_attn_weight>0.):
					loss = actor_loss + args.ego_weight*ego_loss + attn_loss
				else:
					loss = actor_loss + args.ego_weight*ego_loss
				num_batches += 1
				total_loss += float(loss.item())
				pred_ego = torch.nn.functional.softmax(pred_ego, dim=1)
				_, pred_ego = torch.max(pred_ego.data, 1)
				if ('slot' in model_name and not args.fix_slot) or args.box:
					pred_actor = torch.nn.functional.softmax(pred_actor, dim=-1)
					_, pred_actor_idx = torch.max(pred_actor.data, -1)
					pred_actor_idx = pred_actor_idx.detach().cpu().numpy().astype(int)
					f1_batch_new_pred_actor = []
					map_batch_new_pred_actor = []
					for i, b in enumerate(pred_actor_idx):
						f1_new_pred = np.zeros(num_actor_class, dtype=int)
						map_new_pred = np.zeros(num_actor_class, dtype=np.float32)+1e-5

						for j, pred in enumerate(b):
							if pred != num_actor_class:
								f1_new_pred[pred] = 1
								if pred_actor[i, j, pred] > map_new_pred[pred]:
									map_new_pred[pred] = pred_actor[i, j, pred]
						f1_batch_new_pred_actor.append(f1_new_pred)
						map_batch_new_pred_actor.append(map_new_pred)
					f1_batch_new_pred_actor = np.array(f1_batch_new_pred_actor)
					map_batch_new_pred_actor = np.array(map_batch_new_pred_actor)
					f1_pred_actor_list.append(f1_batch_new_pred_actor)
					map_pred_actor_list.append(map_batch_new_pred_actor)
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

			if (args.mask and args.action_attn_weight>0.) or (args.seg and args.bg_attn_weight>0.):
				attn_loss_epoch = attn_loss_epoch / num_batches
				print('attn loss:')
				print(attn_loss_epoch)
				if args.mask:
					action_attn_loss_epoch = action_attn_loss_epoch /num_batches
					print('action_attn_loss')
					print(action_attn_loss_epoch)
				if args.seg:
					bg_attn_loss_epoch = bg_attn_loss_epoch / num_batches
					print('bg_attn_loss_epoch')
					print(bg_attn_loss_epoch)
				
			if args.seg and args.mask:
				iou = action_inter.sum / (action_union.sum + 1e-10)
				for i, val in enumerate(iou):
					print('Action IoU {0}: {1:.2f}'.format(i, val * 100))

				iou = bg_inter.sum / (bg_union.sum + 1e-10)
				for i, val in enumerate(iou):
					print('BG IoU {0}: {1:.2f}'.format(i, val * 100))

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


			print(f'(val) mAP of the actor: {mAP}')
			print(f'(val) mAP of the v: {vehicle_mAP}')
			print(f'(val) mAP of the p: {ped_mAP}')
			print(f'(val) f1 of the actor: {mean_f1}')

			print(f'acc of the ego: {correct_ego/total_ego}')
			# writer.add_scalar('ego', correct_ego/total_ego, epoch)
			if mAP > self.best_mAP:
				self.best_mAP = mAP
				save_cp = True
			print(f'best mAP : {self.best_mAP}')
			print('mAP vehicle:')
			print(mAP_per_class[:12])
			print('mAP ped:')
			print(mAP_per_class[12:])
			# print('f1 vehicle:')
			# print(f1_class[:12])
			# print('f1 ped:')
			# print(f1_class[12:])


			total_loss = total_loss / float(num_batches)
			tqdm.write(f'Epoch {self.cur_epoch:03d}, Batch {batch_num:03d}:' + f' Loss: {total_loss:3.3f}')

			# writer.add_scalar('val_loss', total_loss, self.cur_epoch)
			
			self.val_loss.append(total_loss)
		return save_cp, [mAP, total_loss, mean_f1]


	def save(self, is_best):

		save_best = False
		if is_best:
			self.bestval = self.val_loss[-1]
			self.bestval_epoch = self.cur_epoch
			save_best = True
		

		# Save ckpt for every epoch
		torch.save(model.state_dict(), os.path.join(args.logdir, 'model_%d.pth'%self.cur_epoch))

		# Save the recent model/optimizer states
		torch.save(model.state_dict(), os.path.join(args.logdir, 'model.pth'))
		torch.save(optimizer.state_dict(), os.path.join(args.logdir, 'recent_optim.pth'))

		# Log other data corresponding to the recent model
		# with open(os.path.join(args.logdir, 'recent.log'), 'w') as f:
		# 	f.write(json.dumps(log_table))

		tqdm.write('====== Saved recent model ======>')
		
		if save_best:
			torch.save(model.state_dict(), os.path.join(args.logdir, 'best_model.pth'))
			torch.save(optimizer.state_dict(), os.path.join(args.logdir, 'best_optim.pth'))
			tqdm.write('====== Overwrote best model ======>')


args = get_parser()
print(args)
torch.cuda.empty_cache() 
seq_len = args.seq_len

num_ego_class = 4
num_actor_class = 20


# Data
train_set = road_dataset.ROAD(args=args, seq_len=seq_len, training=True, seg=args.seg, num_class=num_actor_class, model_name=args.id, num_slots=args.num_slots, box=args.box)
val_set = road_dataset.ROAD(args=args, seq_len=seq_len, training=False, seg=args.seg, num_class=num_actor_class, model_name=args.id, num_slots=args.num_slots, box=args.box)	
dataloader_train = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
dataloader_val = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
# Model
model = generate_model(args, args.id, num_ego_class, num_actor_class, args.seq_len).cuda()

if 'mvit' == args.id:
	params = set_lr(model)#
else:
    params = [{'params':model.parameters()}]
optimizer = optim.AdamW(params, lr=5e-4, weight_decay=0.1)
if args.scheduler:
	scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
else:
	scheduler = None
trainer = Engine(args, ego_weight=args.ego_weight)


if args.cp != '':
	model_path = os.path.join(args.logdir, args.cp)
	model.load_state_dict(torch.load(model_path))
	args.logdir = 'road_ft/' + args.logdir
else:
	args.logdir = 'road_scratch/' + args.logdir
print(args.logdir)
# Create logdir
if not os.path.isdir(args.logdir):
	os.makedirs(args.logdir)


result_list = []
if not args.test:
	for epoch in range(100): 
		
		trainer.train(model, optimizer, epoch, model_name=args.id, scheduler=scheduler,ce_weight=args.ce_weight)
		if (epoch % args.val_every == 0 or epoch == args.epochs-1): 
				is_best, res = trainer.validate(model, dataloader_val, None, model_name=args.id, ce_weight=args.ce_weight)
				# trainer.validate(dataloader_val_train, None)
				trainer.save(is_best)
				result_list.append(res)

else:
	trainer.validate(cam=cam)

plot_result(np.array(result_list))
