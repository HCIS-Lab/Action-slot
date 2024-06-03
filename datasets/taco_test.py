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
import torchvision.transforms as transforms

class TACO_TEST(Dataset):

    def __init__(self, 
                args,
                split='val',
                root='/data/carla_dataset/data_collection',
                Max_N=20):
        root = args.root

        self.split = split
        self.model_name = args.model_name
        self.seq_len = args.seq_len
        self.maps = []
        self.id = []
        self.variants = []
        self.scenario_name = []
        self.args =args

        self.videos_list = []
        self.seg_list = []
        self.obj_seg_list = []

        self.idx = []
        self.gt_ego = []
        self.gt_actor = []
        self.slot_eval_gt = []


        self.step = []
        self.start_idx = []
        self.num_class = 64
        self.max_num_obj = []
        
        self.Max_N = Max_N


        max_num_label_a_video = 0
        total_label = 0
        max_frame_a_video = 0
        min_frame_a_video = 100
        total_frame = 0
        total_videos = 0


        n=0

        f = open('../datasets/taco_'+split+'_data.json')
        scenario_list = json.load(f)
        for scenario in tqdm(scenario_list):
                              
            video_folder = ['downsampled/', 'downsampled_224/']
            if args.model_name == 'mvit' or args.model_name == 'videoMAE':
                video_folder = video_folder[1]
            else:
                video_folder = video_folder[0]

            parent_folder, basic, variant = scenario.split('/')
            scenario_path = os.path.join(root,parent_folder,basic,'variant_scenario',variant)
            video_folder_path = os.path.join(scenario_path,'rgb',video_folder)
            if os.path.isdir(video_folder_path):
                check_data = [os.path.join(video_folder_path,img) for img in os.listdir(video_folder_path) if os.path.isfile(os.path.join(video_folder_path,img))]
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
            for m in range(max_num):
                start = start_frame + m
                if start_frame + (self.seq_len-1)*step > end_frame:
                    break
                videos_temp = []
                seg_temp = []
                idx_temp = []
                obj_temp = []
                for i in range(start, end_frame+1, step):
                    imgname = f"{str(i).zfill(8)}.jpg"
                    segname = f"{str(i).zfill(8)}.png"
                    boxname = f"{str(i).zfill(8)}.json"
                    objname = f"{str(i).zfill(8)}.npy"
                    if os.path.isfile(os.path.join(video_folder_path,imgname)):
                        videos_temp.append(os.path.join(video_folder_path,imgname))
                        idx_temp.append(i-start_frame)
                    if os.path.isfile(os.path.join(scenario_path,'mask','background',segname)):
                        seg_temp.append(os.path.join(scenario_path,'mask','background',segname))
                    if os.path.isfile(os.path.join(scenario_path,'mask','object',objname)):
                        obj_temp.append(os.path.join(scenario_path,'mask','object',objname))
                    if len(videos_temp) == self.seq_len:
                        break
                if len(videos_temp) == self.seq_len:
                    videos.append(videos_temp)
                    idx.append(idx_temp)
                    segs.append(seg_temp)
                    obj_f.append(obj_temp)

            self.maps.append(parent_folder)
            self.id.append(basic)
            self.variants.append(variant)
            self.scenario_name.append(os.path.join(parent_folder, basic, variant))
            self.videos_list.append(videos)
            self.idx.append(idx)
            self.seg_list.append(segs)
            self.obj_seg_list.append(obj_f)
            

        if args.box:
            if args.gt:
                self.parse_tracklets() 
            else:
                self.parse_tracklets_detection()

        print('num_videos: ' + str(len(self.variants)))


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
            f = open(os.path.join(root,'tracks','pred','downsampled.txt'))
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
                np.save(os.path.join(root,'tracks','pred','%s' % (i)),out)
                        
    def tracklet_counter(self):
        """
            tracklet (List[List[Dict]]):
                T , boxes per_frame , key: obj_id
            return:
                T x N x 4
        """

        for idx, data in enumerate(self.videos_list):
            num_samples = len(data)
            root = data[num_samples//2][0].split('/')
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
            obj_id_dict = {}
            count = 0
            remove_data_list = []
            sample = data[num_samples//2]
            for j,frame_idx in enumerate(sample):
                frame_idx = frame_idx.split('/')[-1][:-4]
                for obj_id, box in bboxs[frame_idx].items():
                    if obj_id not in obj_id_dict:
                        obj_id_dict[obj_id] = count
                        count += 1

                if self.args.num_objects == 10 and count > 10:
                    remove_data_list.append(data)
                    break
                if self.args.num_objects == 20 and count < 10 and count > 20:
                    # self.videos_list.remove(data)
                    # del self.videos_list[idx]
                    remove_data_list.append(data)
                    break
                if self.args.num_objects == 21 and count < 20:
                    # self.videos_list.remove(data)
                    # del self.videos_list[idx]
                    remove_data_list.append(data)
                    break
        for video in self.videos_list:
            remove = False
            for remove_data in remove_data_list:
                print(video)
                print(remove_data)
                if remove_data == video:
                    remove = True
                    break
            if remove:
                self.videos_list.remove(video)


    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.videos_list)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        data['videos'] = []
        data['id'] = self.id[index]
        data['variants'] = self.variants[index]

        data['map'] = self.maps[index]

        sample_idx = len(self.videos_list[index])//2

        seq_videos = self.videos_list[index][sample_idx]

        # add tracklets
        if self.args.box:
            track_path = seq_videos[0].split('/')
            track_path = track_path[:-3]
            track_path = '/' + os.path.join(*track_path,'tracks','pred',str(sample_idx)) + '.npy'
            tracklets = np.load(track_path)
            data['box'] = tracklets

        for i in range(self.seq_len):
            x = Image.open(seq_videos[i]).convert('RGB')
            data['videos'].append(x)
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