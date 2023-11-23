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
# from torchvideotransforms import video_transforms, volume_transforms

# from pytorchvideo.transforms import (
#     Normalize,
#     NormalizeVideo,
#     ToTensorVideo
# )

def parse_file_name(file_name):
    name = file_name.split('/')
    name[-1] = name[-1][:-3] + "json"
    name[-3] = "bbox"
    name = '/'+os.path.join(*name)
    return name

class NUSCENES(Dataset):

    def __init__(self, 
                args,
                training=True,
                Max_N=20):
        root = args.root
        # /media/hcis-s20/SRL/nuscenes/trainval/samples/CAM_FRONT
        self.training = training
        self.model_name = args.model_name
        self.seq_len = 16

        self.city = []
        self.scenario = []
        self.args =args

        self.video_list = []
        self.seg_list = []

        self.idx = []
        self.gt_ego = []
        self.gt_actor = []
        self.slot_eval_gt = []


        self.step = []
        self.start_idx = []
        self.num_class = 64
        
        
        self.Max_N = Max_N


        max_num_label_a_video = 0
        total_label = 0
        max_frame_a_video = 0
        min_frame_a_video = 100
        total_frame = 0
        total_videos = 0

        label_files = ['nuscenes_boston_labels', 'nuscenes_singapore_labels']

        n=0

        # statistic
        c_stat = {'c12': 0, 'c13':0, 'c14':0,
                    'c21': 0, 'c23': 0, 'c24': 0,
                    'c31': 0, 'c32': 0, 'c34': 0,
                    'c41': 0, 'c42': 0, 'c43': 0}
        b_stat = {'k12': 0, 'k13':0, 'k14':0,
                    'k21': 0, 'k23': 0, 'k24': 0,
                    'k31': 0, 'k32': 0, 'k34': 0,
                    'k41': 0, 'k42': 0, 'k43': 0}


        c_plus_stat = {'c+12': 0, 'c+13':0, 'c+14':0,
                    'c+21': 0, 'c+23': 0, 'c+24': 0,
                    'c+31': 0, 'c+32': 0, 'c+34': 0,
                    'c+41': 0, 'c+42': 0, 'c+43': 0}
        b_plus_stat = {'k+12': 0, 'k+13':0, 'k+14':0,
                    'k+21': 0, 'k+23': 0, 'k+24': 0,
                    'k+31': 0, 'k+32': 0, 'k+34': 0,
                    'k+41': 0, 'k+42': 0, 'k+43': 0}


        p_stat = {'p12': 0, 'p14': 0, 
                    'p21': 0, 'p23': 0, 
                    'p32': 0, 'p34': 0, 
                    'p41': 0, 'p43': 0 
                    }
        p_plus_stat = {'p+12': 0, 'p+14': 0, 
                        'p+21': 0, 'p+23': 0, 
                        'p+32': 0, 'p+34': 0, 
                        'p+41': 0, 'p+43': 0 
                        }

        ego_stat = {'1': 0,'2': 0, '3':0, '4': 0}

        label_stat = [c_stat, b_stat, c_plus_stat, b_plus_stat, p_stat, p_plus_stat, ego_stat]



        seg_folder = ['segmentation_28x28', 'segmentation_32x96']
        if args.pretrain == 'oats':
            seg_folder = seg_folder[0]
        else:
            seg_folder = seg_folder[1]
        all_imgs = [os.path.join(root, 'CAM_FRONT',img) for img in os.listdir(os.path.join(root, 'CAM_FRONT')) if os.path.isfile(os.path.join(root, 'CAM_FRONT',img))]
        all_imgs.sort()

        for label_file in label_files:
            with open('../datasets/' + label_file + '.txt') as f:
                for line_idx, line in enumerate(f):
                    # test set
                    if not training and line_idx%5 != 0:
                        continue
                    # train set
                    if training and line_idx%5 == 0:
                        continue
                    line = line.replace('\n', '')
                    line = line.split(',')
                    start_frame, ego_gt, actor_gt = line[0], line[1], line[2]
                    actor_gt = actor_gt.split(' ')

                    if self.args.box:
                        label_stat, proposal_train_label, gt_ego, gt_actor = get_labels(args, label_stat, ego_gt, actor_gt, num_slots=self.Max_N)
                    elif 'slot' in args.model_name and not args.allocated_slot:
                        label_stat, proposal_train_label, gt_ego, gt_actor = get_labels(args, label_stat, ego_gt, actor_gt, num_slots=args.num_slots)
                    else:
                        label_stat, gt_ego, gt_actor = get_labels(args, label_stat, ego_gt, actor_gt, num_slots=args.num_slots)
                    start_frame = os.path.join(root, 'CAM_FRONT', start_frame)
                    start_frame_idx = all_imgs.index(start_frame)
                    video = all_imgs[start_frame_idx-1:start_frame_idx-1+16]

                    seg = [os.path.join(root, seg_folder, os.path.basename(img)[:-4]+'.png') for img in video]
                    # ------------statistics-------------
                    if torch.count_nonzero(gt_actor) > max_num_label_a_video:
                        max_num_label_a_video = torch.count_nonzero(gt_actor)
                    total_label += torch.count_nonzero(gt_actor)

                    self.city.append(label_file)
                    self.scenario.append(os.path.basename(video[0])[:-4])
                    self.video_list.append(video)
                    self.seg_list.append(seg)
                    # self.obj_seg_list.append(obj_f)

                    self.gt_ego.append(gt_ego)
                    
                    if ('slot' in args.model_name and not args.allocated_slot) or args.box:
                        self.gt_actor.append(proposal_train_label)
                        self.slot_eval_gt.append(gt_actor)
                    else:
                        self.gt_actor.append(gt_actor)
                    if self.args.val_confusion:
                        self.confusion_label_list.append(confusion_label)

                    # -----------statstics--------------

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

        self.label_stat = label_stat
        if args.box:
            self._parse_tracklet(training)
    
    def _save_as_txt(self,training):
        osp = os.path.join
        if not os.path.isdir(osp(self.args.root,'pred')):
            os.mkdir(osp(self.args.root,'pred')) 
        if training:
            path = osp(self.args.root,'pred','train')
        else:
            path = osp(self.args.root,'pred','val')
        if not os.path.isdir(path):
            os.mkdir(path) 
        for i, samples in enumerate(self.video_list):
            sample_path = osp(path,str(i))
            if not os.path.isdir(sample_path):
                os.mkdir(sample_path) 
            f = open(osp(sample_path,'imgs.txt'), 'w')
            for p in samples:
                f.write(p)
                f.write('\n')
            f.close()

    def _parse_tracklet(self,training):

        def parse_tracklet():
            # frame_id: {id: [x,y,w,h]}
            out = {}
            for i in range(self.args.seq_len):
                out[i+1] = {}
            for line in tracklet:
                line = line.split(' ')[:6]
                frame = int(line[0])
                obj_id = int(line[1])
                box = [int(line[2]),int(line[3]),int(line[2])+int(line[4]),int(line[3])+int(line[5])]
                out[frame][obj_id] = box
            return out
        
        osp = os.path.join
        if training:
            path = osp(self.args.root,'pred','train')
        else:
            path = osp(self.args.root,'pred','val')

        self.box = []
        for i, _ in enumerate(self.video_list):
            f = open(osp(path,str(i),'CAM_FRONT.txt'))
            tracklet = f.readlines()
            tracklet = parse_tracklet()

            out = np.zeros((self.seq_len,self.Max_N,4))
            obj_id_dict = {}
            # tracklet id
            count = 0
            # img frame id
            for j in range(self.args.seq_len):
                for obj_id in tracklet[j+1]:
                    if obj_id not in obj_id_dict:
                        if count >=20:
                            break
                        obj_id_dict[obj_id] = count
                        count += 1
                    try:
                        out[j][obj_id_dict[obj_id]] = tracklet[j+1][obj_id]
                    except:
                        continue
            self.box.append(out)
        assert len(self.video_list) == len(self.box)
            
            # for debug
            # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            # out_video = cv2.VideoWriter(os.path.join('/media/hcis-s20/SRL/debug',f"{i}.mp4"), fourcc, 2.0, (768,  256))
            # for img, boxs in zip(samples,out):
            #     img = cv2.imread(img)
            #     img = cv2.resize(img, (768, 256), interpolation=cv2.INTER_NEAREST)
            #     for box in boxs:
            #         cv2.rectangle(img, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (255,0,0, 255), 1)  
            #     out_video.write(img)
            # out_video.release()

    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.video_list)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        data['city'] = self.city[index]
        data['scenario'] = self.scenario[index]
        data['videos'] = []
        data['bg_seg'] = []
        data['obj_masks'] = []
        data['ego'] = self.gt_ego[index]
        data['actor'] = self.gt_actor[index]
        data['raw'] = []

        if ('slot' in self.args.model_name and not self.args.allocated_slot) or self.args.box:
            data['slot_eval_gt'] = self.slot_eval_gt[index]


        seq_videos = self.video_list[index]
        if self.args.bg_mask:
            seq_seg = self.seg_list[index]
        # if self.args.obj_mask:
        #     obj_masks_list = self.obj_seg_list[index][sample_idx]


        # add tracklets
        if self.args.box:
            data['box'] = self.box[index]

        for i in range(self.seq_len):
            x = Image.open(seq_videos[i]).convert('RGB')
            x = scale(x, self.args.model_name, self.args.pretrain)
            data['videos'].append(x)
            if self.args.plot:
                data['raw'].append(x)

            if self.args.bg_mask and i %self.args.mask_every_frame == 0:
                data['bg_seg'].append(self.get_stuff_mask(seq_seg[i]))

        data['videos'] = to_np(data['videos'], self.args.model_name, self.args.backbone)
        if self.args.plot:
            data['raw'] = to_np_no_norm(data['raw'])
        return data

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
        # condition[:4, :] = 1
        # condition[-2:, :] = 1
        condition = condition.type(torch.float32)
        h, w = condition.shape[0], condition.shape[1]
        condition = torch.reshape(condition, (1, h, w))
        return condition
def get_obj_mask(obj_path):
    seg_dict = np.load(obj_path)
    obj_masks = list(seg_dict.values())
    if len(obj_masks) == 0:
        obj_masks = torch.zeros([40, 32, 96], dtype=torch.int32)
    else:
        obj_masks = torch.from_numpy(np.stack(obj_masks, 0))
    # img = torch.flip(torch.from_numpy(img).type(torch.int).permute(2,0,1),[0])
    obj_masks = obj_masks.type(torch.int)
    pad_num = 40 - obj_masks.shape[0]
    obj_masks = torch.cat((obj_masks, torch.zeros([pad_num, 32, 96], dtype=torch.int32)), dim=0)
    obj_masks = obj_masks.type(torch.float32)
    # obj_masks = torch.reshape(obj_masks, (-1, 1, 32, 96))
    # obj_masks = F.interpolate(obj_masks, size=(8,24))
    # obj_masks = torch.reshape(obj_masks, (-1, 8, 24))

    return obj_masks

def read_box(box_path):
    f = open(box_path)
    box = json.load(f)
    box_list = []
    for b in box:
        v = list(b.values())[0]
        norm_box = [v[0]/512, v[2]/512, v[1]/1536, v[3]/1536]
        box_list.append(torch.FloatTensor(norm_box))
    while (len(box_list)<35):
        box_list.append(torch.FloatTensor([0, 0, 0, 0]))
    box_list = torch.stack(box_list, dim=0)
    return box_list


def scale(image, model_name=None, pretrain=None):

    if pretrain == 'oats':
        (width, height) = (224, 224)
    else:
        (width, height) = (768, 256)
    # else:
    #     (width, height) = (int(image.width // scale), int(image.height // scale))
    im_resized = image.resize((width, height), Image.Resampling.LANCZOS)

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

def get_labels(args, label_stat, ego_gt, gt_list, num_slots=64):   
    num_class = 64
    model_name = args.model_name
    allocated_slot = args.allocated_slot

    ego_table = {'1': 0, '2': 1, '3':2, '4': 3}

    taco_actor_table = { 'c12': 0, 'c13':1, 'c14':2,
                    'c21': 3, 'c23': 4, 'c24': 5,
                    'c31': 6, 'c32': 7, 'c34': 8,
                    'c41': 9, 'c42': 10, 'c43': 11,
                    'c+12': 12, 'c+13':13, 'c+14':14,
                    'c+21': 15, 'c+23': 16, 'c+24': 17,
                    'c+31': 18, 'c+32': 19, 'c+34': 20,
                    'c+41': 21, 'c+42': 22, 'c+43': 23,
                    'k12': 24, 'k13':25, 'k14':26,
                    'k21': 27, 'k23': 28, 'k24': 29,
                    'k31': 30, 'k32': 31, 'k34': 32,
                    'k41': 33, 'k42': 34, 'k43': 35,
                    'k+12': 36, 'k+13':37, 'k+14':38,
                    'k+21': 39, 'k+23': 40, 'k+24': 41,
                    'k+31': 42, 'k+32': 43, 'k+34': 44,
                    'k+41': 45, 'k+42': 46, 'k+43': 47,
                    'p12': 48, 'p14': 49, 
                    'p21': 50, 'p23': 51, 
                    'p32': 52, 'p34': 53, 
                    'p41': 54, 'p43': 55,
                    'p+12': 56, 'p+14': 57, 
                    'p+21': 58, 'p+23': 59, 
                    'p+32': 60, 'p+34': 61, 
                    'p+41': 62, 'p+43': 63 
                    }

    oats_actor_map={
        # from oats
        'c+13':0, 'p+12':1, 'p21':2, 'c31':3, 'c21':4,
        'p43':5, 'c13':6, 'c12':7, 'c24':8, 'c42':9, 
        'c34':10,  'p23':11,  'c41':12,  'p34':13,  'c14':14,
         'c23':15,  'p12':16,'p+23':17,  'p32':18,  'p+34':19,
        'p+43':20 ,  'p14':21, 'p+32': 22,  'c32':23,  'k31':24,
        'c43':25,  'p+14':26, 'c+42':27, 'p+41':28,  'p+21':29,
         'c+31':30,  'p41':31,  'k24':32, 'c+24':33, 'k13':34,
        # absence in oats
        'c+12':35, 'c+14':36, 'c+21':37, 'c+23':38,  'c+32':39,
        'c+34':40,  'c+41':41,  'c+43':42,  'k12':43,  'k14':44,
        'k21':45,  'k23':46,  'k32':47,  'k34':48,  'k41':49,
         'k42':50, 'k43':51, 'k+12': 52,  'k+13':53, 'k+14':54,
        'k+21':55, 'k+23':56,  'k+24':57,  'k+31':58,  'k+32':59,
         'k+34':60,  'k+41':61,  'k+42':62,  'k+43':63
        }
    if args.pretrain == 'oats':
        class_table = oats_actor_map
    else:
        class_table = taco_actor_table

    actor_class = [0]*64

    proposal_train_label = []
    for gt in gt_list:
        gt = gt.lower()
        if ('slot' in model_name and not allocated_slot) or 'ARG'in model_name or 'ORN'in model_name:
            if not class_table[gt] in proposal_train_label:
                proposal_train_label.append(class_table[gt])
        actor_class[class_table[gt]] = 1
        if 'c+' == gt[:2]:
            label_stat[2][gt]+=1
        elif 'k+' == gt[:2]:
            label_stat[3][gt]+=1
        elif 'p+' == gt[:2]:
            label_stat[5][gt]+=1
        elif 'p' == gt[0]:
            label_stat[4][gt]+=1
        elif 'c' == gt[0]:
            label_stat[0][gt]+=1
        elif 'k' == gt[0]:
            label_stat[1][gt]+=1

            

    ego_class = ego_table[ego_gt]
    label_stat[6][ego_gt] += 1
    ego_label = torch.tensor(ego_class)

    if ('slot' in model_name and not allocated_slot) or 'ARG'in model_name or 'ORN'in model_name :
        while (len(proposal_train_label)!= num_slots):
            proposal_train_label.append(num_class)
        proposal_train_label = torch.LongTensor(proposal_train_label)
        # actor_class = actor_class[:-1]
        actor_class = torch.FloatTensor(actor_class)
        return label_stat, proposal_train_label, ego_label, actor_class
    else:
        actor_class = torch.FloatTensor(actor_class)
        return label_stat, ego_label, actor_class
    
