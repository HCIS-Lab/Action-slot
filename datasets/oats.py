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


def parse_file_name(file_name):
    name = file_name.split('/')
    name[-1] = name[-1][:-3] + "json"
    name[-3] = "bbox"
    name = '/'+os.path.join(*name)
    return name

class OATS(Dataset):

    def __init__(self, 
                args,
                training=True,
                root='/data/carla_dataset/data_collection',
                Max_N=63):
        # root = '/work/u8526971/data_collection'
        # root = '/home/hcis-s19/Desktop/data_collection'
        # root = '/home/hcis-s20/Desktop/data_collection'
        root = '/media/hankung/ssd/oats/oats_data/'
        # root = '/media/hcis-s16/hank/taco'
        # root = '/media/hcis-s20/SRL/taco'
        # root = '/media/user/data/taco'
        # root = '/media/hcis-s19/DATA/taco'

        self.training = training
        self.model_name = args.model_name
        self.seq_len = args.seq_len

        self.id = []
        self.variants = []
        self.args =args

        self.videos_list = []
        self.seg_list = []
        self.obj_seg_list = []

        self.idx = []
        self.gt_ego = []
        self.gt_actor = []
        self.slot_eval_gt = []
        self.confusion_label_list = []


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

        # type_list = ['interactive', 'non-interactive', 'ap_Town01', 
        # 'ap_Town02','ap_Town03', 'ap_Town04', 'ap_Town05', 'ap_Town06', 'ap_Town07', 'ap_Town10HD', 
        # 'runner_Town05', 'runner_Town10HD']
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

        label_stat = [c_stat, b_stat, c_plus_stat, b_plus_stat, p_stat, p_plus_stat, ego_stat]



        all_splits = ['s1', 's2', 's3']
        if training:
            splits = all_splits.remove(args.split)
        else:
            splits = [args.split]

        scenarios = []
        for split in splits:
            if os.path.isfile(os.path.join(root, 'splits', split + '.txt')):
                with open(os.path.join(root, 'splits', split + '.txt')) as f:
                    for line in f:
                        line = line.replace('\n', '')
                        scenarios.append(line)

        for scenario in tqdm(scenarios, file=sys.stdout):
            scenario_path = os.path.join(root, 'images', 'scenario_' + scenario)
            annotation_path = os.path.join(root, 'annotations', 'scenario_' + scenario + '.npy')

            check_data = [os.path.join(scenario_path, img) for img in os.listdir(scenario_path) if os.path.isfile(scenario_path)]
            check_data.sort()
            # if len(check_data) < 50:
            #     continue

            videos = []
            segs = []
            obj_f = []
            idx = []

            start_frame = int(check_data[0].split('/')[-1].split('.')[0])
            end_frame = int(check_data[-1].split('/')[-1].split('.')[0])
            num_frame = end_frame - start_frame + 1
            step = num_frame // self.seq_len
            # ----------------------
            max_num = 50
            for m in range(max_num):
                start = start_frame + m
                # step = ((end_frame-start + 1) // (seq_len+1)) -1
                if start_frame + (self.seq_len-1)*step > end_frame:
                    break
                videos_temp = []
                for i in range(start, end_frame+1, step):
                    imgname = check_data[i]
                    if os.path.isfile(imgname):
                        videos_temp.append(imgname)
                        idx_temp.append(i-start)

                    if len(videos_temp) == self.seq_len:
                        break

                if len(videos_temp) == self.seq_len:
                    videos.append(videos_temp)
                    idx.append(idx_temp)

            if len(videos) == 0
                continue


            if self.args.box:
                proposal_train_label, gt_actor = get_labels(args, annotation_path, num_slots=self.Max_N)
            elif 'slot' in args.model_name and not args.allocated_slot:
                proposal_train_label, gt_actor = get_labels(args, annotation_path, num_slots=args.num_slots)
            else:
                gt_actor = get_labels(args, annotation_path, num_slots=args.num_slots)
                        

            # ------------statistics-------------
            if torch.count_nonzero(gt_actor) > max_num_label_a_video:
                max_num_label_a_video = torch.count_nonzero(gt_actor)
            total_label += torch.count_nonzero(gt_actor)


                        
                        



                        # -----
                        ego_class = 'e:z1-z1'
                        for g in gt:
                            g = g.lower()
                            if g[0] != 'e':
                                if g[:2] == 'c:':
                                    label_stat[0][g]+=1
                                elif g[:2] == 'b:':
                                    label_stat[1][g]+=1
                                elif g[:2] == 'c+':
                                    label_stat[2][g]+=1
                                elif g[:2] == 'b+':
                                    label_stat[3][g]+=1
                                elif g[:2] == 'p:':
                                    label_stat[4][g]+=1
                                elif g[:2] == 'p+':
                                    label_stat[5][g]+=1

                            elif g[0] == 'e':
                                ego_class = g
                            else:
                                g = g[2:]
                                if g != 'ne':
                                    if not gt in actor_table.keys():
                                        print(g)
                                        return

                        label_stat[6][ego_class] +=1


                        self.id.append(s.split('/')[-1])
                        self.variants.append(v.split('/')[-1])

                        self.videos_list.append(videos)
                        self.idx.append(idx)
                        self.seg_list.append(segs)
                        self.obj_seg_list.append(obj_f)

                        self.gt_ego.append(gt_ego)
                        
                        if ('slot' in args.model_name and not args.allocated_slot) or args.box:
                            self.gt_actor.append(proposal_train_label)
                            self.slot_eval_gt.append(gt_actor)
                        else:
                            self.gt_actor.append(gt_actor)
                        if self.args.val_confusion:
                            self.confusion_label_list.append(confusion_label)

                        # -----------statstics--------------
                        if num_frame > max_frame_a_video:
                            max_frame_a_video = num_frame
                        if num_frame < min_frame_a_video:
                            min_frame_a_video = num_frame
                        total_frame += num_frame
                        total_videos += 1
        if False:
            self.parse_tracklets_detection() 
        print('num_variant: ' + str(len(self.variants)))
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
        # -----------------
        print('max_num_label_a_video: '+ str(max_num_label_a_video))
        print('total_label: '+ str(total_label))
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
            for i,sample in enumerate(data):
                temp = []
                root = sample[0].split('/')
                root = root[:-3]
                root = '/'+os.path.join(*root)
                if not os.path.isdir(os.path.join(root,'tracks')):
                    os.mkdir(os.path.join(root,'tracks'))
                for img in sample:
                    # read bbox
                    box_path = parse_file_name(img)
                    f = open(box_path)
                    track = json.load(f)
                    temp.append(track)
                    f.close()
                parse_tracklet(temp,root,i)
        

    def __len__(self):
        """Returns the length of the dataset. """
        return len(self.videos_list)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        data['videos'] = []
        data['bg_seg'] = []
        data['obj_masks'] = []
        # data['box'] = []
        data['raw'] = []
        data['ego'] = self.gt_ego[index]
        data['actor'] = self.gt_actor[index]
        data['id'] = self.id[index]
        data['variants'] = self.variants[index]


        if ('slot' in self.args.model_name and not self.args.allocated_slot) or self.args.box:
            data['slot_eval_gt'] = self.slot_eval_gt[index]

        if self.training:
            sample_idx = random.randint(0, len(self.videos_list[index])-1)
        else:
            sample_idx = len(self.videos_list[index])//2

        seq_videos = self.videos_list[index][sample_idx]
        if self.args.bg_mask:
            seq_seg = self.seg_list[index][sample_idx]
        if self.args.obj_mask:
            obj_masks_list = self.obj_seg_list[index][sample_idx]
        # if self.box:
        #     seq_box = self.box_list[index][sample_idx]

        # add tracklets
        if self.args.box:
            track_path = seq_videos[0].split('/')
            track_path = track_path[:-3]
            if self.args.gt:
                track_path = '/' + os.path.join(*track_path,'tracks',str(sample_idx)) + '.npy'
            else:
                track_path = '/' + os.path.join(*track_path,'tracks_pred',str(sample_idx)) + '.npy'
            tracklets = np.load(track_path)
            data['box'] = tracklets

        for i in range(self.seq_len):
            x = Image.open(seq_videos[i]).convert('RGB')
            # x = scale(x, 2, self.args.model_name)
            data['videos'].append(x)
            if self.args.plot:
                data['raw'].append(x)
            if self.args.bg_mask and i %self.args.mask_every_frame == 0:
                data['bg_seg'].append(get_stuff_mask(seq_seg[i]))
            if self.args.obj_mask and i % self.mask_every_frame:
                data['obj_masks'].append(get_obj_mask(obj_masks_list[i]))
        if self.args.plot:
            data['raw'] = to_np_no_norm(data['raw'], self.args.model_name)
        data['videos'] = to_np(data['videos'], self.args.model_name)

        if self.args.val_confusion:
            data['confusion_label'] = self.confusion_label_list[index]
        return data

def get_stuff_mask(seg_path):
    img = cv2.imread(os.path.join(seg_path), cv2.IMREAD_COLOR)
    img = torch.flip(torch.from_numpy(img).type(torch.int).permute(2,0,1),[0])

    condition = img[0] == 4 
    condition += img[0] == 6
    condition += img[0] == 7
    condition += img[0] == 8
    condition += img[0] == 10
    condition += img[0] == 14
    condition = ~condition
    condition = condition.type(torch.int)
    condition = condition.type(torch.float32)

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


def scale(image, scale=2.0, model_name=None):

    if scale == -1.0:
        (width, height) = (224, 224)
    else:
        (width, height) = (int(image.width // scale), int(image.height // scale))
    # (width, height) = (int(image.width // scale), int(image.height // scale))
    im_resized = image.resize((width, height), Image.ANTIALIAS)

    return im_resized



def to_np(v, model_name):
    for i, _ in enumerate(v):
        transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])])
        v[i] = transform(v[i])
    return v

def to_np_no_norm(v, model_name):
    for i, _ in enumerate(v):
        transform = transforms.Compose([
                                transforms.ToTensor(),
                                ])
        v[i] = transform(v[i])
    return v

def get_labels(args, annotation_path, num_slots=64):   
    num_class = 35
    model_name = args.model_name
    allocated_slot = args.allocated_slot

    actor_class = [0]*35
    proposal_train_label = []

    annotation = np.load(annotation_path)
    for label in annotation:
        if ('slot' in model_name and not allocated_slot) or 'ARG'in model_name or 'ORN'in model_name:
            if not (label-1) in proposal_train_label:
                proposal_train_label.append(label-1)
        actor_class[label-1] = 1
    
    if ('slot' in model_name and not allocated_slot) or 'ARG'in model_name or 'ORN'in model_name :
        while (len(proposal_train_label)!= num_slots):
            proposal_train_label.append(num_class)
        proposal_train_label = torch.LongTensor(proposal_train_label)
        actor_class = torch.FloatTensor(actor_class)
        return proposal_train_label, actor_class
    else:
        actor_class = torch.FloatTensor(actor_class)
        return actor_class
    
