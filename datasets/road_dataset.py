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
from tool import get_rot
import torchvision.transforms as transforms
# from torchvideotransforms import video_transforms, volume_transforms
from collections import namedtuple

def parse_file_name(file_name):
    name = file_name.split('/')
    name[-1] = name[-1][:-3] + "json"
    name[-3] = "bbox"
    name = '/'+os.path.join(*name)
    return name

class ROAD(Dataset):


    def __init__(self, 
                args,
                seq_len=16, 
                training=True,
                seg=False,
                num_class=12,
                model_name='i3d',
                num_slots=21,
                box=False,
                root='/data/carla_dataset/data_collection',
                Max_N=63):
        # root = '/work/u8526971/data_collection'
        # root = '/home/hcis-s19/Desktop/data_collection'
        # root = '/home/hcis-s20/Desktop/data_collection'
        root = '/media/hankung/ssd/road/rgb-images'
        self.training = training
        self.scale=args.scale
        self.seg = seg
        self.model_name = model_name
        self.variants = []
        self.args =args

        self.front_list = []
        self.seg_front_list = []

        self.idx = []
        self.ego_list = []
        self.actor_list = []
        self.slot_eval_gt = []


        self.step = []
        self.start_idx = []
        self.num_class = num_class
        self.seq_len = seq_len
        self.box = box
        self.Max_N = Max_N


        max_num_label_a_video = 0
        total_label = 0
        max_frame_a_video = 0
        min_frame_a_video = 100
        total_frame = 0
        total_videos = 0
        train_video_list = ['1','2','3','4', '5', '6','8','10','11','12','14', '15']
        # val_video_list = ['1','2','3','4', '5', '6','8','10','11','12','14','15','16', '17', '18']
        val_video_list = ['16', '17', '18']
        n=0

        # statistic
        actor_stat_table = {'12': 0, '13':0, '14':0,
                    '21': 0, '23': 0, '24': 0,
                    '31': 0, '32': 0, '34': 0,
                    '41': 0, '42': 0, '43': 0}
        ped_stat = {'c12': 0, 'c14': 0, 
                        'c21': 0, 'c23': 0, 
                        'c32': 0, 'c34': 0, 
                        'c41': 0, 'c43': 0 
                        }
        ego_stat = {'1': 0,'2': 0, '3':0, '4': 0}


        if self.training:
            video_list = train_video_list
        else:
            video_list = val_video_list
        # ----------------------
        for t, v_id in enumerate(video_list):
            print(v_id)
            video_path = os.path.join(root, v_id)
            seg_video_path = os.path.join(root, v_id+'_segmentation')
            with open('../datasets/road/'+v_id+'.txt') as f:
                for line in f:
                    line = line.replace('\n', '')
                    line = line.split(',')
                    start_frame, end_frame, ego_gt, actor_gt = line[0], line[1], line[2], line[3]
                    actor_gt = actor_gt.split(' ')
                    ego_gt, actor_gt, actor_stat_table, ped_stat, ego_stat = get_multi_class(ego_gt, actor_gt, actor_stat_table, ped_stat, ego_stat, num_class)
                    total_label += torch.count_nonzero(actor_gt)
                    fronts = []
                    segs_f = []
                    idx = []

                    start_frame = int(start_frame)
                    end_frame = int(end_frame)
                    num_frame = end_frame - start_frame + 1
                    if num_frame < 1:
                        print(line)
                        return
                    step = num_frame // seq_len

                    max_num = 50
                    for m in range(max_num):
                        start = start_frame + m
                        # step = ((end_frame-start + 1) // (seq_len+1)) -1
                        if start_frame + (seq_len-1)*step > end_frame:
                            break
                        front_temp = []
                        seg_f_temp = []
                        idx_temp = []
                        for i in range(start, end_frame+1, step):
                            imgname = f"{str(i).zfill(5)}.jpg"
                            segname = f"{str(i).zfill(5)}.png"
                            if os.path.isfile(video_path+"/"+imgname):
                                front_temp.append(video_path+"/"+imgname)
                                idx_temp.append(i-start)
                            if os.path.isfile(seg_video_path+"/"+segname):
                                seg_f_temp.append(seg_video_path+"/"+segname)

                            if len(front_temp) == seq_len:
                                break

                        if len(front_temp) == seq_len:
                            fronts.append(front_temp)
                            segs_f.append(seg_f_temp)
                            idx.append(idx_temp)


                    self.front_list.append(fronts)
                    self.seg_front_list.append(segs_f)
                    self.idx.append(idx)
                    self.ego_list.append(ego_gt)
                    
                    self.actor_list.append(actor_gt)

                    # -----------statstics--------------
                    if num_frame > max_frame_a_video:
                        max_frame_a_video = num_frame
                    if num_frame < min_frame_a_video:
                        min_frame_a_video = num_frame
                    total_frame += num_frame
                    total_videos += 1

        print('actor_stat:')
        print(actor_stat_table)
        print('ped_stat')
        print(ped_stat)
        print('ego_stat')
        print(ego_stat)

        # -----------------
        print('max_num_label_a_video: '+ str(max_num_label_a_video))
        print('total_label: '+ str(total_label))
        print('max_frame_a_video: '+ str(max_frame_a_video))
        print('min_frame_a_video: '+ str(min_frame_a_video))
        print('total_frame: '+ str(total_frame))
        print('total_videos: '+ str(total_videos))
        

        

    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.front_list)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        data['fronts'] = []
        data['seg_front'] = []
        data['ego'] = self.ego_list[index]
        data['actor'] = self.actor_list[index]


        # if ('slot' in self.model_name and not self.args.fix_slot) or self.args.box:
        #     data['slot_eval_gt'] = self.slot_eval_gt[index]

        if self.training:
            sample_idx = random.randint(0, len(self.front_list[index])-1)
        else:
            sample_idx = len(self.front_list[index])//2

        seq_fronts = self.front_list[index][sample_idx]
        if self.seg:
            seq_seg_front = self.seg_front_list[index][sample_idx]


        for i in range(self.seq_len):
            x = Image.open(seq_fronts[i]).convert('RGB')
            x = scale(x)
            data['fronts'].append(x)
            if self.seg:
                data['seg_front'].append(self.get_stuff_mask(seq_seg_front[i]))
        data['fronts'] = to_np(data['fronts'], self.model_name)

        return data

    # @classmethod
    # def encode_target(cls, target):
    #     return cls.id_to_train_id[np.array(target)]

    def get_stuff_mask(self, seg_path):
        img = cv2.imread(os.path.join(seg_path), cv2.IMREAD_COLOR)
        img = torch.from_numpy(img).type(torch.int).permute(2,0,1)
        #c, h, w
        img = torch.sum(img, dim=0)
        # target = Image.open(os.path.join(seg_path))
        # target = self.encode_target(target)
        condition = img == 320
        condition += img == 511
        condition += img == 300
        condition += img == 255
        condition += img == 142
        condition += img == 70
        condition += img == 100
        condition += img == 90
        condition += img == 110
        condition += img == 230
        condition += img == 162
        condition += img == 142

        #     CityscapesClass('unlabeled',            0, 255, 'void', 0, False, True, (0, 0, 0)),
        #     CityscapesClass('ego vehicle',          1, 255, 'void', 0, False, True, (0, 0, 0)),
        #     CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        #     CityscapesClass('out of roi',           3, 255, 'void', 0, False, True, (0, 0, 0)),
        #     CityscapesClass('static',               4, 255, 'void', 0, False, True, (0, 0, 0)),
        #     CityscapesClass('dynamic',              5, 255, 'void', 0, False, True, (111, 74, 0)), #185
        #     CityscapesClass('ground',               6, 255, 'void', 0, False, True, (81, 0, 81)),
          # CityscapesClass('road',                 7, 0, 'flat', 1, False, False, (128, 64, 128)), #256+64=320
        # # CityscapesClass('sidewalk',             8, 1, 'flat', 1, False, False, (244, 35, 232)),279+232=511
        #     CityscapesClass('parking',              9, 255, 'flat', 1, False, True, (250, 170, 160)), #420+160=580
        #     CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
        #     CityscapesClass('building',             11, 2, 'construction', 2, False, False, (70, 70, 70)),
        #     CityscapesClass('wall',                 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        #     CityscapesClass('fence',                13, 4, 'construction', 2, False, False, (190, 153, 153)),
        #     CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
        #     CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
        #     CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
        #     CityscapesClass('pole',                 17, 5, 'object', 3, False, False, (153, 153, 153)),
        #     CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
        #     CityscapesClass('traffic light',        19, 6, 'object', 3, False, False, (250, 170, 30)),
        #     CityscapesClass('traffic sign',         20, 7, 'object', 3, False, False, (220, 220, 0)),
        #     CityscapesClass('vegetation',           21, 8, 'nature', 4, False, False, (107, 142, 35)),
        #     CityscapesClass('terrain',              22, 9, 'nature', 4, False, False, (152, 251, 152)),
        #     CityscapesClass('sky',                  23, 10, 'sky', 5, False, False, (70, 130, 180)),
        # # CityscapesClass('person',               24, 11, 'human', 6, True, False, (220, 20, 60)),
        # # CityscapesClass('rider',                25, 12, 'human', 6, True, False, (255, 0, 0)),
        # # CityscapesClass('car',                  26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        # # CityscapesClass('truck',                27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        # # CityscapesClass('bus',                  28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        # # CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        # # CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        # # CityscapesClass('train',                31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        # # CityscapesClass('motorcycle',           32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        # # CityscapesClass('bicycle',              33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        # # CityscapesClass('license plate',        -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
        condition = ~condition
        condition = condition.type(torch.int)
        condition[:4, :] = 1
        condition[-2:, :] = 1
        condition = condition.type(torch.float32)

        return condition



def scale(image):

    im_resized = image.resize((768, 256), Image.ANTIALIAS)
    return im_resized


def to_np(v, model_name):
    for i, _ in enumerate(v):
        transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])])
        v[i] = transform(v[i])
    return v


def get_multi_class(ego_gt, actor_gt, actor_stat_table, ped_stat, ego_stat, num_class, num_slots=20):   

    ego_table = {'1': 0, '2':1, '3': 2, '4': 3}

    actor_table = { '12': 0, '13':1, '14':2,
                    '21': 3, '23': 4, '24': 5,
                    '31': 6, '32': 7, '34': 8,
                    '41': 9, '42': 10, '43': 11,

                    'c12': 12, 'c14': 13, 
                    'c21': 14, 'c23': 15, 
                    'c32': 16, 'c34': 17, 
                    'c41': 18, 'c43': 19 
                    }

    ego_class = ego_table[ego_gt]
    ego_stat[ego_gt] +=1
    actor_class = [0]*num_class
    for gt in actor_gt:
        gt = gt.lower()
        if gt[0] == 'c':
            ped_stat[gt]+=1
            actor_class[actor_table[gt]] = 1
        else:
            if not gt in actor_table.keys():
                print(gt)
                return
            else:
                actor_class[actor_table[gt]] = 1
                actor_stat_table[gt]+=1

    ego_label = torch.tensor(ego_class)

    actor_class = torch.FloatTensor(actor_class)

    return ego_label, actor_class, actor_stat_table, ped_stat, ego_stat
    
