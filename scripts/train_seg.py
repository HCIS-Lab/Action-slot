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
# sys.path.append('/data/hanku/Interaction_benchmark/datasets')
sys.path.append('/data/hanku/Interaction-benchmark/datasets')
sys.path.append('/data/hanku/Interaction-benchmark/config')
sys.path.append('/data/hanku/Interaction-benchmark/models')

# from .configs.config import GlobalConfig
import seg_data


from sklearn.metrics import average_precision_score, precision_score, f1_score, recall_score, accuracy_score, hamming_loss
# from torchmetrics import F1Score

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image
from PIL import Image

# import cnnlstm_backbone
from model import generate_deeplab
from utils import *
from torchvision import models
import matplotlib.image

torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str, default='deeplab', help='Unique experiment identifier.')

parser.add_argument('--no_ped', help="", action="store_true")

parser.add_argument('--seq_len', type=int, default=16, help='')

parser.add_argument('--device', type=str, default='cuda', help='Device to use')
parser.add_argument('--epochs', type=int, default=100, help='Number of train epochs.')
parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate.')
parser.add_argument('--val_every', type=int, default=5, help='Validation frequency (epochs).')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')


args = parser.parse_args()
args.logdir = os.path.join(args.logdir, args.id)
print(args)
writer = SummaryWriter(log_dir=args.logdir)

class Engine(object):

  def __init__(self,  cur_epoch=0, cur_iter=0):
    self.cur_epoch = cur_epoch
    self.cur_iter = cur_iter
    self.bestval_epoch = cur_epoch
    self.train_loss = []
    self.val_loss = []
    self.bestval = 1e10

  def train(self, epoch):
    loss_epoch = 0.
    ego_loss_epoch = 0.
    ce_loss_epoch = 0.
    bce_loss_epoch = 0.
    num_batches = 0

    inter_meter = AverageMeter()
    union_meter = AverageMeter()

    model.train()
    # Train loop
    for data in tqdm(dataloader_train):
      
      # efficiently zero gradients
      # for p in model.parameters():
      #   p.grad = None

      # create batch and move to GPU
      fronts_in = data['fronts']

      seg_front_in = data['seg_front']


      inputs = []
      seg_front = []
      for i in range(seq_len):
        inputs.append(fronts_in[i].to(args.device, dtype=torch.float32))
        seg_front.append(seg_front_in[i].to(args.device))
          
      
      # labels
      batch_size = inputs[0].shape[0]
      h, w = seg_front[0].shape[-2], seg_front[0].shape[-1]
      seg_front = torch.stack(seg_front, 0)
      seg_front = torch.permute(seg_front, (1, 0, 2, 3)) #[batch, len, 1, h, w]
      seg_front = seg_front.contiguous().view(batch_size*seq_len, h, w).long()

      h, w = inputs[0].shape[-2], inputs[0].shape[-1]
      inputs = torch.stack(inputs, 0)
      inputs = torch.permute(inputs, (1, 0, 2, 3, 4)) #[batch, len, 1, h, w]
      inputs = inputs.contiguous().view(batch_size*seq_len, 3, h, w)

      pred_seg = model(inputs)

      
      ce = nn.CrossEntropyLoss(reduction='mean', ignore_index=0).cuda()
      
      ce_loss = ce(pred_seg, seg_front)
      loss = ce_loss
      loss.backward()

      loss_epoch += float(loss.item())

      ce_loss_epoch +=float(ce_loss.item())
      _, pred_seg = torch.max(pred_seg, 1)
      pred_seg = pred_seg.data.cpu().numpy().squeeze().astype(np.uint8)
      seg_front = seg_front.cpu()
      mask = seg_front.numpy().astype(np.uint8)
      inter, union = inter_and_union(pred_seg, mask, 5)
      inter_meter.update(inter)
      union_meter.update(union)

      num_batches += 1

      optimizer.step()
      optimizer.zero_grad()
      writer.add_scalar('train_loss', loss.item(), self.cur_iter)
      self.cur_iter += 1

  

    loss_epoch = loss_epoch / num_batches
    ce_loss_epoch = ce_loss_epoch / (num_batches)



    print('total loss')
    print(loss_epoch)

    
    iou = inter_meter.sum / (union_meter.sum + 1e-10)
    for i, val in enumerate(iou):
        print('IoU {0}: {1:.2f}'.format(i, val * 100))
    print('Mean IoU: {0:.2f}'.format(iou.mean() * 100))

    self.train_loss.append(loss_epoch)
    self.cur_epoch += 1

  def validate(self, dataloader, epoch):
    model.eval()
    with torch.no_grad(): 
      num_batches = 0
      loss = 0.

      inter_meter = AverageMeter()
      union_meter = AverageMeter()

      for batch_num, data in enumerate(tqdm(dataloader), 0):
        id = data['id'][0]
        v = data['variants'][0]
        fronts_in = data['fronts']
        seg_front_in = data['seg_front']
        inputs = []
        seg_front = []

        for i in range(seq_len):

          inputs.append(fronts_in[i].to(args.device, dtype=torch.float32))
          seg_front.append(seg_front_in[i].to(args.device, dtype=torch.long))
          
        batch_size = inputs[0].shape[0]

        h, w = seg_front[0].shape[-2], seg_front[0].shape[-1]
        seg_front = torch.stack(seg_front, 0)
        seg_front = torch.permute(seg_front, (1, 0, 2, 3)) #[batch, len, 1, h, w]
        seg_front = seg_front.contiguous().view(batch_size*seq_len, h, w).long()

        h, w = inputs[0].shape[-2], inputs[0].shape[-1]
        inputs = torch.stack(inputs, 0)
        inputs = torch.permute(inputs, (1, 0, 2, 3, 4)) #[batch, len, 1, h, w]
        inputs = inputs.contiguous().view(batch_size*seq_len, 3, h, w)
        pred_seg = model(inputs)
        
        ce = nn.CrossEntropyLoss(reduction='mean', ignore_index=0).cuda()
        ce_loss = ce(pred_seg, seg_front)        
        loss = ce_loss
        
        num_batches += 1

        _, pred_seg = torch.max(pred_seg, 1)
        pred_seg = pred_seg.data.cpu().numpy()
        seg_front = seg_front.cpu().numpy().astype(np.uint8)
        for i in range(seq_len):
          pred_seg_temp = pred_seg[i]
          # print(pred_seg_temp.shape)
          pred_seg_temp = pred_seg_temp.squeeze().astype(np.uint8)
          seg_front_temp = seg_front[i]
          inter, union = inter_and_union(pred_seg_temp, seg_front_temp, 5)
          inter_meter.update(inter)
          union_meter.update(union)
          # if epoch == args.epochs -1:
          scene_name = os.path.join(args.logdir, id+'_'+v)
          if not os.path.isdir(scene_name):
            os.makedirs(scene_name)
          if not os.path.isdir(scene_name):
            os.makedirs(scene_name)

          matplotlib.image.imsave(os.path.join(scene_name, str(i))+'_pred'+'.png', pred_seg_temp)
          matplotlib.image.imsave(os.path.join(scene_name, str(i))+'_gt'+'.png', seg_front_temp)


      iou = inter_meter.sum / (union_meter.sum + 1e-10)
      for i, val in enumerate(iou):
        print('IoU {0}: {1:.2f}'.format(i, val * 100))
      print('Mean IoU: {0:.2f}'.format(iou.mean() * 100))

      loss = loss / float(num_batches)
      tqdm.write(f'Epoch {self.cur_epoch:03d}, Batch {batch_num:03d}:' + f' Loss: {loss:3.3f}')

      writer.add_scalar('val_loss', loss, self.cur_epoch)
      
      self.val_loss.append(loss.data)

  def save(self):

    save_best = False
    if self.val_loss[-1] <= self.bestval:
      self.bestval = self.val_loss[-1]
      self.bestval_epoch = self.cur_epoch
      save_best = True
    
    # Create a dictionary of all data to save

    # log_table = {
    #   'epoch': self.cur_epoch,
    #   'iter': self.cur_iter,
    #   'bestval': float(self.bestval.data),
    #   'bestval_epoch': self.bestval_epoch,
    #   'train_loss': self.train_loss,
    #   'val_loss': self.val_loss,
    # }

    # Save ckpt for every epoch
    torch.save(model.state_dict(), os.path.join(args.logdir, 'model_%d.pth'%self.cur_epoch))

    # Save the recent model/optimizer states
    torch.save(model.state_dict(), os.path.join(args.logdir, 'model.pth'))
    torch.save(optimizer.state_dict(), os.path.join(args.logdir, 'recent_optim.pth'))

    # Log other data corresponding to the recent model
    # with open(os.path.join(args.logdir, 'recent.log'), 'w') as f:
    #   f.write(json.dumps(log_table))

    tqdm.write('====== Saved recent model ======>')
    
    if save_best:
      torch.save(model.state_dict(), os.path.join(args.logdir, 'best_model.pth'))
      torch.save(optimizer.state_dict(), os.path.join(args.logdir, 'best_optim.pth'))
      tqdm.write('====== Overwrote best model ======>')

# Config
# config = GlobalConfig()
torch.cuda.empty_cache() 
seq_len = args.seq_len

# Data

train_set = seg_data.Seg_Data(seq_len=seq_len)
val_set = seg_data.Seg_Data(seq_len=seq_len, training=False)
dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=20, pin_memory=True)
dataloader_val = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=20, pin_memory=True)
# Model
backbone_params, last_params, model = generate_deeplab(args.id)
model = model.cuda()
# backbone_params = (
#     list(model.conv1.parameters()) +
#     list(model.bn1.parameters()) +
#     list(model.layer1.parameters()) +
#     list(model.layer2.parameters()) +
#     list(model.layer3.parameters()) +
#     list(model.layer4.parameters()))
# last_params = list(model.aspp.parameters())

# for t in backbone_params:
#   t.requires_grad=False

# optimizer = optim.SGD([
#   {'params': filter(lambda p: p.requires_grad, backbone_params)},
#   {'params': filter(lambda p: p.requires_grad, last_params)}],
#   lr=args.lr, momentum=0.9, weight_decay=0.0001)
optimizer = optim.SGD([
  {'params': filter(lambda p: p.requires_grad, last_params)}],
  lr=args.lr, momentum=0.9, weight_decay=0.0001)
# optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
trainer = Engine()




# Create logdir
if not os.path.isdir(args.logdir):
  os.makedirs(args.logdir)
  print('Created dir:', args.logdir)
elif os.path.isfile(os.path.join(args.logdir, 'recent.log')):
  print('Loading checkpoint from ' + args.logdir)
  with open(os.path.join(args.logdir, 'recent.log'), 'r') as f:
    log_table = json.load(f)

  # Load variables
  trainer.cur_epoch = log_table['epoch']
  if 'iter' in log_table: trainer.cur_iter = log_table['iter']
  trainer.bestval = log_table['bestval']
  trainer.train_loss = log_table['train_loss']
  trainer.val_loss = log_table['val_loss']

  # Load checkpoint
  model.load_state_dict(torch.load(os.path.join(args.logdir, 'model.pth')))
  optimizer.load_state_dict(torch.load(os.path.join(args.logdir, 'recent_optim.pth')))

# Log args
with open(os.path.join(args.logdir, 'args.txt'), 'w') as f:
  json.dump(args.__dict__, f, indent=2)
for epoch in range(trainer.cur_epoch, args.epochs): 
  trainer.train(epoch)
  if epoch % args.val_every == 0 or epoch == args.epochs-1: 
      trainer.validate(dataloader_val, None)
      trainer.save()

