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
                split='test',
                root='/data/carla_dataset/data_collection',
                Max_N=20):
        root = args.root

        self.split = split
        self.model_name = args.model_name
        self.seq_len = args.seq_len

        self.maps = []
        self.id = []
        self.variants = []
        self.args =args

        self.videos_list = []
        self.seg_list = []
        self.obj_seg_list = []

        self.idx = []
        
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

        folder_list = [
        # scenarios from RikBench
        'interactive', 'non-interactive', 
        # scenarios collected by auto-pilot
        'ap_Town01','ap_Town02','ap_Town03', 'ap_Town04', 'ap_Town05', 'ap_Town06', 'ap_Town07', 'ap_Town10HD', 
        # scenarios collected by scenario-runner
        'runner_Town03','runner_Town05', 'runner_Town10HD']
        n=0

        # ------------search dataset-------------
        # The TACO dataset is construct with 3 different collection:
        #   1. 'interactive' and 'non-interactive' from RiskBench
        #   2. 'ap_Townxx' collected by auto-pilot
        #   3. 'runner_Townxx' collected by the scenario runner

        # - All data collections follow the hierarchy:
        #       - parant folder (e.g., interactive, ap_Town10HD, or runner_Town05)
        #           - basic scenario (e.g., 3_t1-2_0_m_f_l_1_0, or t2)
        #               - various scenario (e.g., 1)

        # - The folder 'interactive' and 'non-interactive' contained various maps, e.g., Town01, ... Town10HD.
        # - We use Town10HD as the test set. 

        # iterate parent folder
        for t, folder in enumerate(folder_list):
            basic_scenarios = [os.path.join(root, folder, s) for s in os.listdir(os.path.join(root, folder))]

            # iterate basic scenarios
            print('searching data from '+folder+' folder')
            for s in tqdm(basic_scenarios, file=sys.stdout):

                # extract scenarios for train/val/test set
                scenario_id = s.split('/')[-1]

                if split == 'test':
                    if folder == 'interactive' or folder == 'non-interactive':
                        if scenario_id.split('_')[0] != '10':
                            continue
                    else:
                        testing_set = ['ap_Town10HD', 'runner_Town10HD']
                        if not folder in testing_set:
                            continue

                # iterate various scenarios
                variants_path = os.path.join(s, 'variant_scenario')
                if os.path.isdir(variants_path):
                    variants = [os.path.join(variants_path, v) for v in os.listdir(variants_path)]
                    
                    for v in variants:
                        v_id = v.split('/')[-1]
                        # extract ground-truth labels
                                      
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
                        for m in range(max_num):
                            start = start_frame + m
                            # step = ((end_frame-start + 1) // (seq_len+1)) -1
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
                                if os.path.isfile(v+"/rgb/"+video_folder+imgname):
                                    videos_temp.append(v+"/rgb/"+video_folder +imgname)
                                    idx_temp.append(i-start_frame)
                                if os.path.isfile(v+"/mask/background/"+segname):
                                    seg_temp.append(v+"/mask/background/"+segname)
                                if os.path.isfile(v+"/mask/object/"+objname):
                                    obj_temp.append(v+"/mask/object/"+objname)

                                if len(videos_temp) == self.seq_len and len(seg_temp) == self.seq_len and len(obj_temp) == self.seq_len:
                                    break

                            if len(videos_temp) == self.seq_len:
                                videos.append(videos_temp)
                                idx.append(idx_temp)
                                if len(seg_temp) == self.seq_len:
                                    segs.append(seg_temp)
                                    obj_f.append(obj_temp)

                        if len(videos) == 0 or len(segs) ==0 or len(obj_f)==0:
                            continue
                        if len(segs)!=len(videos):
                            continue

                        self.maps.append(folder)
                        self.id.append(s.split('/')[-1])
                        self.variants.append(v.split('/')[-1])
                        self.videos_list.append(videos)
                        self.idx.append(idx)
                        self.seg_list.append(segs)
                        self.obj_seg_list.append(obj_f)

                        
                        # -----------statstics--------------
                        if num_frame > max_frame_a_video:
                            max_frame_a_video = num_frame
                        if num_frame < min_frame_a_video:
                            min_frame_a_video = num_frame
                        total_frame += num_frame
                        total_videos += 1

        if args.box:
            if args.gt:
                self.parse_tracklets() 
            else:
                self.parse_tracklets_detection()
        # if args.plot:
        self.parse_tracklets()

        # -----------------
        print('max_frame_a_video: '+ str(max_frame_a_video))
        print('min_frame_a_video: '+ str(min_frame_a_video))
        print('total_frame: '+ str(total_frame))
        print('total_videos: '+ str(total_videos))
        
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
                        if obj not in obj_id_dict :
                            if count == 20:
                                continue
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
                            if count == 20:
                                continue
                            obj_id_dict[obj_id] = count
                            count += 1
                        out[j][obj_id_dict[obj_id]] = box
                np.save(os.path.join(root,'tracks','gt','%s' % (i)),out)
            self.max_num_obj.append(count)

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
        data['max_num_obj'] = self.max_num_obj[index]
        # data['box'] = []
        data['raw'] = []
        data['id'] = self.id[index]
        data['variants'] = self.variants[index]

        data['map'] = self.maps[index]

        sample_idx = len(self.videos_list[index])//2

        seq_videos = self.videos_list[index][sample_idx]

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
            # x = scale(x, 2, self.args.model_name)
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