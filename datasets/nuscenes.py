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
                Max_N=63):
        root = args.root
        # /media/hcis-s20/SRL/nuscenes/trainval/samples/CAM_FRONT
        self.training = training
        self.model_name = args.model_name
        self.seq_len = 16

        self.city = []
        self.scenarios = []
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
        b_plus_stat = {'k12': 0, 'k+13':0, 'k+14':0,
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




        all_imgs = [os.path.join(root, img) for img in os.listdir(root) if os.path.isfile(os.path.join(root, img))]
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
                        proposal_train_label, gt_ego, gt_actor = get_labels(args, label_stat, ego_gt, actor_gt, num_slots=self.Max_N)
                    elif 'slot' in args.model_name and not args.allocated_slot:
                        label_stat, proposal_train_label, gt_ego, gt_actor = get_labels(args, label_stat, ego_gt, actor_gt, num_slots=args.num_slots)
                    else:
                        label_stat, gt_ego, gt_actor = get_labels(args, label_stat, ego_gt, actor_gt, num_slots=args.num_slots)
                    start_frame = os.path.join(root, start_frame)
                    start_frame_idx = all_imgs.index(start_frame)
                    video = all_imgs[start_frame_idx-1:start_frame_idx-1+16]

                    # ------------statistics-------------
                    if torch.count_nonzero(gt_actor) > max_num_label_a_video:
                        max_num_label_a_video = torch.count_nonzero(gt_actor)
                    total_label += torch.count_nonzero(gt_actor)

                    self.city.append(label_file)
                    self.video_list.append(video)
                    # self.seg_list.append(segs)
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
        if False:
            self.parse_tracklets() 
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

        

    def parse_tracklets_detection(self):
        """
            read {scenario}/tracking_pred_2/tracks/front.txt
            format: frame, id, x, y, w, h
        """
        
        def parse_tracklet():
            # frame_id: {id: [x,y,w,h]}
            out = {}
            for line in tracklet:
                line = line.split(' ')[:6]
                frame = int(line[0])
                obj_id = int(line[1])
                box = [int(line[2]),int(line[3]),int(line[2])+int(line[4]),int(line[3])+int(line[5])]
                if frame not in out:
                    out[frame] = {}
                out[frame][obj_id] = box
            return out
            
        
        for data,idx in tqdm(zip(self.videos_list,self.idx)):
            root = data[0][0].split('/')
            root = root[:-3]
            root = '/'+os.path.join(*root)
            if not os.path.isdir(os.path.join(root,'tracks_pred')):
                os.mkdir(os.path.join(root,'tracks_pred'))
            f = open(os.path.join(root,'tracking_pred_2','tracks','front.txt'))
            tracklet = f.readlines()
            # parse_tracklet
            tracklet = parse_tracklet()
            f.close()
            # for every sample]
            assert len(data) == len(idx)
            for i,idx_list in enumerate(idx):
                out = np.zeros((self.seq_len,self.Max_N,4))
                obj_id_dict = {}
                # tracklet id
                count = 0
                # img frame id
                for j,index in enumerate(idx_list):
                    try:
                        for obj_id in tracklet[int(index)+1]:
                            if obj_id not in obj_id_dict:
                                obj_id_dict[obj_id] = count
                                count += 1
                            try:
                                out[j][obj_id_dict[obj_id]] = tracklet[int(index)+1][obj_id]
                            except:
                                continue
                    except:
                        continue
                np.save(os.path.join(root,'tracks_pred','%s' % (i)),out)
                        
        

    def parse_tracklets(self):
        """
            tracklet (List[List[Dict]]):
                T , boxes per_frame , key: obj_id
            return:
                T x N x 4
        """
        def parse_tracklet(tracklet,root,index):
            out = np.zeros((self.seq_len,self.Max_N,4))
            obj_id_dict = {}
            count = 0
            for i,track in enumerate(tracklet):
                for boxes in track:
                    for obj in boxes:
                        if obj not in obj_id_dict:
                            obj_id_dict[obj] = count
                            count += 1
                        out[i][obj_id_dict[obj]] = boxes[obj]
            np.save(os.path.join(root,'tracks','%s' % (index)),out)
            # with open(os.path.join(root,'tracks','%s.json' % (index)), 'w') as f:
            #     json.dump(out, f)
                        
            
        # for each data
        for data in tqdm(self.videos_list):
            root = data[0][0].split('/')
            root = root[:-3]
            root = '/'+os.path.join(*root)
            if not os.path.isdir(os.path.join(root,'tracks')):
                os.mkdir(os.path.join(root,'tracks'))
            if not os.path.isdir(os.path.join(root,'tracks','gt')):
                os.mkdir(os.path.join(root,'tracks','gt'))
            if not os.path.isdir(os.path.join(root,'tracks','pred')):
                os.mkdir(os.path.join(root,'tracks','pred'))
            # read bbox.json
            f = open(os.path.join(root,'bbox.json'))
            bboxs = json.load(f)
            f.close()
            for i,sample in enumerate(data):
                out = np.zeros((self.seq_len,self.Max_N,4))
                obj_id_dict = {}
                count = 0
                # iterate each imgs
                for j,frame_idx in enumerate(sample):
                    frame_idx = frame_idx.split('/')[-1][:-4]
                    for obj_id, box in bboxs[frame_idx].items():
                        if obj_id not in obj_id_dict:
                            obj_id_dict[obj_id] = count
                            count += 1
                        out[j][obj_id_dict[obj_id]] = box
                np.save(os.path.join(root,'tracks','gt','%s' % (i)),out)
            
                # if not os.path.isdir(os.path.join(root,'tracks')):
                #     os.mkdir(os.path.join(root,'tracks'))
                # for img in sample:
                #     # read bbox
                #     box_path = parse_file_name(img)
                #     f = open(box_path)
                #     track = json.load(f)
                #     temp.append(track)
                #     f.close()
                # parse_tracklet(temp,root,i)
        

    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.video_list)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        data['videos'] = []
        data['bg_seg'] = []
        data['obj_masks'] = []
        # data['box'] = []
        data['ego'] = self.gt_ego[index]
        data['actor'] = self.gt_actor[index]


        if ('slot' in self.args.model_name and not self.args.allocated_slot) or self.args.box:
            data['slot_eval_gt'] = self.slot_eval_gt[index]


        seq_videos = self.video_list[index]
        # if self.args.bg_mask:
        #     seq_seg = self.seg_list[index][sample_idx]
        # if self.args.obj_mask:
        #     obj_masks_list = self.obj_seg_list[index][sample_idx]


        # add tracklets
        if self.args.box:
            track_path = seq_videos[0].split('/')
            track_path = track_path[:-3]
            if self.args.gt:
                track_path = '/' + os.path.join(*track_path,'tracks','gt',str(sample_idx)) + '.npy'
            else:
                track_path = '/' + os.path.join(*track_path,'tracks','pred',str(sample_idx)) + '.npy'
            tracklets = np.load(track_path)
            data['box'] = tracklets

        for i in range(self.seq_len):
            x = Image.open(seq_videos[i]).convert('RGB')
            x = scale(x, self.args.model_name, self.args.pretrain)
            data['videos'].append(x)

            if self.args.bg_mask and i %self.args.mask_every_frame == 0:
                data['bg_seg'].append(Image.open(seq_seg[i]).convert('L'))

        data['videos'] = to_np(data['videos'], self.args.model_name, self.args.backbone)
        data['bg_seg'] = to_np_no_norm(data['bg_seg'])
        return data


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

    actor_table = { 'c12': 0, 'c13':1, 'c14':2,
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


    actor_class = [0]*64

    proposal_train_label = []
    for gt in gt_list:
        gt = gt.lower()
        if ('slot' in model_name and not allocated_slot) or 'ARG'in model_name or 'ORN'in model_name:
            if not actor_table[gt] in proposal_train_label:
                proposal_train_label.append(actor_table[gt])
        actor_class[actor_table[gt]] = 1
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
        elif 'k:' == gt[0]:
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
    
