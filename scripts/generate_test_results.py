import argparse
import os
import sys
from tqdm import tqdm
import pickle

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append('../datasets')
sys.path.append('../configs')
sys.path.append('../models')   
import taco_test
from model import generate_model
from utils import *
from parser_test import get_test_parser

torch.backends.cudnn.benchmark = True

                 
torch.cuda.empty_cache()
args = get_test_parser()
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

    def test(self, model, dataloader, epoch):
        save_results = {}
        model.eval()
        with torch.no_grad():   
            for batch_num, data in enumerate(tqdm(dataloader)):

                # -------get video name------
                map = data['map'][0]
                id = data['id'][0]
                v = data['variants'][0]
                video_in = data['videos']
                scenario = map + '/'+id + '/' + v

                # -------get input------                    
                inputs = []
                for i in range(seq_len):
                    inputs.append(video_in[i].to(args.device, dtype=torch.float32))
                # -------object boxes for object-detector-based------    
                if args.box:
                    box_in = data['box']
                    if isinstance(box_in,np.ndarray):
                        boxes = torch.from_numpy(box_in).to(args.device, dtype=torch.float32)
                    else:
                        boxes = box_in.to(args.device, dtype=torch.float32)
                batch_size = inputs[0].shape[0]

                # -------get prediction------  
                if ('slot' in args.model_name) or args.box or 'mvit' in args.model_name:
                    if args.box:
                        pred_ego, pred_actor = model(inputs, boxes)
                    else:
                        pred_ego, pred_actor, attn = model(inputs)
                else:
                    pred_ego, pred_actor = model(inputs)

                # -------transform object-detector-based's output (instance-level) to multilabel ------
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
                    save_results[scenario] = map_batch_new_pred_actor
                # -------output multilabel for video-level models and Action-slot------
                else:
                    pred_actor = torch.sigmoid(pred_actor)
                    pred_actor = pred_actor.detach().cpu().numpy()
                    save_results[scenario] = pred_actor

        with open('prediction_results.pkl', 'wb') as handle:
            pickle.dump(save_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

# -------start testing------
torch.cuda.empty_cache() 
seq_len = args.seq_len
num_ego_class = 4
num_actor_class = 64

# Data
test_set = taco_test.TACO_TEST(args=args, split=args.split)
dataloader_test = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

model = generate_model(args, num_ego_class, num_actor_class).cuda()
trainer = Engine(args)

model_path = os.path.join(args.cp)
model.load_state_dict(torch.load(model_path))

trainer.test(model, dataloader_test, None)
