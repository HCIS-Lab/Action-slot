import os
import json
from PIL import Image
import cv2
import torch.nn.functional as F

import numpy as np
import torch 
from torch.utils.data import Dataset
from tqdm import tqdm
import sys
import json 
import random
# from tool import get_rot
import torchvision.transforms as transforms



class TACO_FP(Dataset):

    def __init__(self, 
                args,
                training=True,
                root='/data/carla_dataset/data_collection',
                Max_N=20):

        root = args.root

        self.training = training
        self.model_name = args.model_name
        self.seq_len = args.seq_len

        self.maps = []
        self.id = []
        self.variants = []
        self.args =args

        self.videos_list = []


        self.idx = []
        self.gt_ego = []
        self.gt_actor = []
        self.slot_eval_gt = []


        self.step = []
        self.start_idx = []
        self.num_class = 64



        type_list = ['ap_Town10HD']
        n=0

        # statistic
        c_stat = {'c:z1-z2': 0, 'c:z1-z3':0, 'c:z1-z4':0,
                    'c:z2-z1': 0, 'c:z2-z3': 0, 'c:z2-z4': 0,
                    'c:z3-z1': 0, 'c:z3-z2': 0, 'c:z3-z4': 0,
                    'c:z4-z1': 0, 'c:z4-z2': 0, 'c:z4-z3': 0}
        b_stat = {'b:z1-z2': 0, 'b:z1-z3':0, 'b:z1-z4':0,
                    'b:z2-z1': 0, 'b:z2-z3': 0, 'b:z2-z4': 0,
                    'b:z3-z1': 0, 'b:z3-z2': 0, 'b:z3-z4': 0,
                    'b:z4-z1': 0, 'b:z4-z2': 0, 'b:z4-z3': 0}


        c_plus_stat = {'c+:z1-z2': 0, 'c+:z1-z3':0, 'c+:z1-z4':0,
                    'c+:z2-z1': 0, 'c+:z2-z3': 0, 'c+:z2-z4': 0,
                    'c+:z3-z1': 0, 'c+:z3-z2': 0, 'c+:z3-z4': 0,
                    'c+:z4-z1': 0, 'c+:z4-z2': 0, 'c+:z4-z3': 0}
        b_plus_stat = {'b+:z1-z2': 0, 'b+:z1-z3':0, 'b+:z1-z4':0,
                    'b+:z2-z1': 0, 'b+:z2-z3': 0, 'b+:z2-z4': 0,
                    'b+:z3-z1': 0, 'b+:z3-z2': 0, 'b+:z3-z4': 0,
                    'b+:z4-z1': 0, 'b+:z4-z2': 0, 'b+:z4-z3': 0}


        p_stat = {'p:c1-c2': 0, 'p:c1-c4': 0, 
                        'p:c2-c1': 0, 'p:c2-c3': 0, 
                        'p:c3-c2': 0, 'p:c3-c4': 0, 
                        'p:c4-c1': 0, 'p:c4-c3': 0 
                        }
        p_plus_stat = {'p+:c1-c2': 0, 'p+:c1-c4': 0, 
                        'p+:c2-c1': 0, 'p+:c2-c3': 0, 
                        'p+:c3-c2': 0, 'p+:c3-c4': 0, 
                        'p+:c4-c1': 0, 'p+:c4-c3': 0 
                        }

        ego_stat = {'e:z1-z1': 0,'e:z1-z2': 0, 'e:z1-z3':0, 'e:z1-z4': 0}


        # ----------------------
        for t, type in enumerate(type_list):
            basic_scenarios = [os.path.join(root, type, s) for s in os.listdir(os.path.join(root, type))]
            # iterate scenarios
            print('searching data')
            for s in tqdm(basic_scenarios, file=sys.stdout):
                # a basic scenario
                scenario_id = s.split('/')[-1]

                variants_path = os.path.join(s, 'variant_scenario')
                if os.path.isdir(variants_path):
                    variants = [os.path.join(variants_path, v) for v in os.listdir(variants_path)]
                    
                    for v in variants:
                        # print(v)
                        v_id = v.split('/')[-1]
                        if 'DS' in v_id:
                            continue
                        if os.path.isfile(v+'/retrieve_gt.txt'):
                            continue
                        gt_ego = 0
                        gt_ego = torch.tensor(gt_ego)

                        gt_actor = [0]*64
                        gt_actor = torch.FloatTensor(gt_actor)

                        video_folder = ['downsampled/', 'downsampled_224/']
                        if args.model_name == 'mvit' or args.model_name == 'videoMAE':
                            video_folder = video_folder[1]
                        else:
                            video_folder = video_folder[0]
                        if os.path.isdir(v+"/rgb/" + video_folder):
                            check_data = [v+"/rgb/"+video_folder+img for img in os.listdir(v+"/rgb/"+video_folder) if os.path.isfile(v+"/rgb/"+video_folder+ img)]
                            check_data.sort()
                        else:
                            continue
                        if len(check_data) < 50:
                            continue

                        videos = []
                        segs = []
                        obj_f = []
                        idx = []

                        start_frame = int(check_data[0].split('/')[-1].split('.')[0])
                        end_frame = int(check_data[-1].split('/')[-1].split('.')[0])
                        num_frame = end_frame - start_frame + 1
                        step = num_frame // self.seq_len

                        max_num = 50
                        # step = ((end_frame-start + 1) // (seq_len+1)) -1
                        for m in range(max_num):
                            start = start_frame + m
                            # step = ((end_frame-start + 1) // (seq_len+1)) -1
                            if start_frame + (self.seq_len-1)*step > end_frame:
                                break
                            videos_temp = []
                            idx_temp = []

                            for i in range(start, end_frame+1, step):
                                imgname = f"{str(i).zfill(8)}.jpg"
                                segname = f"{str(i).zfill(8)}.png"
                                boxname = f"{str(i).zfill(8)}.json"
                                objname = f"{str(i).zfill(8)}.npy"
                                if os.path.isfile(v+"/rgb/"+video_folder+imgname):
                                    videos_temp.append(v+"/rgb/"+video_folder +imgname)
                                    idx_temp.append(i-start_frame)

                                if len(videos_temp) == self.seq_len:
                                    break

                            if len(videos_temp) == self.seq_len:
                                videos.append(videos_temp)
                                idx.append(idx_temp)
                        if len(videos) == 0:
                            continue

                        self.maps.append(type)
                        self.id.append(s.split('/')[-1])
                        self.variants.append(v.split('/')[-1])
                        self.videos_list.append(videos)
                        self.idx.append(idx)
                        self.gt_ego.append(gt_ego)
                        self.gt_actor.append(gt_actor)

 
    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.videos_list)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        data['videos'] = []
        data['raw'] = []
        data['ego'] = self.gt_ego[index]
        data['actor'] = self.gt_actor[index]
        data['id'] = self.id[index]
        data['variants'] = self.variants[index]

        # if self.args.plot_mode == '':
        data['map'] = self.maps[index]

        if self.training:
            sample_idx = random.randint(0, len(self.videos_list[index])-1)
        else:
            sample_idx = len(self.videos_list[index])//2

        seq_videos = self.videos_list[index][sample_idx]

        for i in range(self.seq_len):
            x = Image.open(seq_videos[i]).convert('RGB')
            # x = scale(x, 2, self.args.model_name)
            data['videos'].append(x)
            if self.args.plot:
                data['raw'].append(x)
      
        if self.args.plot:
            data['raw'] = to_np_no_norm(data['raw'])
    
        data['videos'] = to_np(data['videos'], self.args.model_name, self.args.backbone)

        return data



def scale(image, scale=2.0, model_name=None):

    if scale == -1.0:
        (width, height) = (224, 224)
    else:
        (width, height) = (int(image.width // scale), int(image.heighft // scale))
    # (width, height) = (int(image.width // scale), int(image.height // scale))
    im_resized = image.resize((width, height), Image.ANTIALIAS)

    return im_resized



def to_np(v, model_name, backbone):
    if backbone != 'inception':
        transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])])
    else:
        transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    for i, _ in enumerate(v):
        v[i] = transform(v[i])
    return v

def to_np_no_norm(v):
    transform = transforms.Compose([
                transforms.ToTensor(),
                ])
    for i, _ in enumerate(v):
        v[i] = transform(v[i])
    return v



    
