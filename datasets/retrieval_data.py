import os
import json
from PIL import Image

import numpy as np
import torch 
from torch.utils.data import Dataset
from tqdm import tqdm
import sys
import json 
import random
from tool import get_rot

###LSS
# from pyquaternion import Quaternion



class Retrieval_Data(Dataset):

    def __init__(self, 
                seq_len=8, 
                training=True,
                is_top=False,
                front_only=True,
                scale=4,
                viz=False,
                seg=False,
                lss=False,
                num_cam=1,
                root='/media/hankung/ssd/carla_13/CARLA_0.9.13/PythonAPI/examples/data_collection'):
        
        self.training = training
        self.is_top = is_top
        self.front_only = front_only
        self.seg = seg
        self.viz = viz
        self.id = []
        self.variants = []

        self.front_list = []
        self.left_list = []
        self.right_list = []
        self.top_list = []

        self.road_para = []
        self.gt_ego = []
        self.gt_actor = []
        self.lss = lss
        self.num_cam = num_cam
        self.step = []
        self.start_idx = []



        if training:
            self.scale = float(scale)
        else:
            self.scale = float(scale)

        self.seq_len = seq_len
        type_list = ['interactive', 'non-interactive']
        n=0
        save_actor = []
        for t, type in enumerate(type_list):
            basic_scenarios = [os.path.join(root, type, s) for s in os.listdir(os.path.join(root, type))]

            # iterate scenarios
            print('searching data')
            for s in tqdm(basic_scenarios, file=sys.stdout):
                # a basic scenario
                scenario_id = s.split('/')[-1]
                if training and scenario_id.split('_')[0] != '10' or not training and scenario_id.split('_')[0] == '10':

                    # if road_class != 5:
                    variants_path = os.path.join(s, 'variant_scenario')
                    variants = [os.path.join(variants_path, v) for v in os.listdir(variants_path)]
                    
                    for v in variants:

                        # get retrieval label
                        v_id = v.split('/')[-1]
                        if os.path.isfile(v+'/retrieve_gt.txt'):
                            with open(v+'/retrieve_gt.txt') as f:
                                gt = []

                                for line in f:
                                    line = line.replace('\n', '')
                                    if line != '\n':
                                        gt.append(line)
                                gt = list(set(gt))
                                if 'None' in gt:
                                    continue
                        else:
                            continue

                        try:
                            road_para, gt_ego, gt_actor = get_multi_class(gt, scenario_id, v_id)
                        except:
                            continue
                        
                        # a data sample

                        if not self.is_top and os.path.isdir(v+"/rgb/front/"):
                            check_data = [v+"/rgb/front/"+ img for img in os.listdir(v+"/rgb/front/") if os.path.isfile(v+"/rgb/front/"+ img)]
                            check_data.sort()

                        if self.is_top and self.seg and os.path.isdir(v+"/semantic_segmentation/lbc_seg/"):
                            check_data = [v+"/semantic_segmentation/lbc_seg/"+ img for img in os.listdir(v+"/semantic_segmentation/lbc_seg/") if os.path.isfile(v+"/semantic_segmentation/lbc_seg/"+ img)]
                            check_data.sort()


                        fronts = []
                        lefts = []
                        rights = []
                        tops = []

                        # step = 2
                        start_frame = int(check_data[0].split('/')[-1].split('.')[0])
                        end_frame = int(check_data[-1].split('/')[-1].split('.')[0])
                        # start = 60
                        max_num = 50

                        for m in range(max_num):
                            start = start_frame + m
                            step = ((end_frame-start + 1) // (seq_len+1)) -1
                            if step == 0:
                                break
                            front_temp = []
                            left_temp = []
                            right_temp = []
                            for i in range(start, end_frame+1, step):
                                # images
                                filename = f"{str(i).zfill(8)}.png"
                                if self.is_top:
                                    if self.seg:
                                        tops.append(v+"/semantic_segmentation/lbc_seg/"+filename)
                                    else:
                                        tops.append(v+"/rgb/top/"+filename)
                                else:
                                    if os.path.isfile(v+"/rgb/front/"+filename):
                                        front_temp.append(v+"/rgb/front/"+filename)
                                        if not self.front_only:
                                            if os.path.isfile(v+"/rgb/left/"+filename):
                                                left_temp.append(v+"/rgb/left/"+filename)
                                            else:
                                                break
                                            if os.path.isfile(v+"/rgb/right/"+filename):
                                                right_temp.append(v+"/rgb/right/"+filename)
                                            else:
                                                break

                                if not self.is_top and len(front_temp) == seq_len and self.front_only:
                                    break
                                if not self.is_top and not self.front_only and \
                                len(front_temp) == seq_len and len(left_temp) == seq_len and len(right_temp) == seq_len:
                                    break

                                if self.is_top and len(tops) == seq_len:
                                    break
                            if not self.is_top and len(front_temp) == seq_len:
                                fronts.append(front_temp)
                                if not self.front_only:
                                    lefts.append(left_temp)
                                    rights.append(right_temp)

                            if self.is_top and len(tops) == seq_len:
                                break
                            else:
                                tops = []

                        if not self.is_top and len(fronts) == 0:
                            continue

                        if self.is_top and len(tops) != seq_len:
                            continue   
                            # if len(fronts) != seq_len and self.front_only:
                            #     continue
                            # if (len(rights) != seq_len or len(lefts) != seq_len) and not self.front_only:
                            #     continue


                        self.id.append(s.split('/')[-1])
                        self.variants.append(v.split('/')[-1])
                        if self.is_top:
                            self.top.append(tops)
                        else:  
                            self.front_list.append(fronts)
                            if not self.front_only:
                                self.left_list.append(lefts)
                                self.right_list.append(rights)

                        self.road_para.append(road_para)
                        self.gt_ego.append(gt_ego)
                        self.gt_actor.append(gt_actor)
                        save_actor.append(gt_actor)

        out = [0]*36
        out = torch.FloatTensor(out)       
        for a in save_actor:
            out = out + a
        np.save(str(training)+'.npy', out)

            # print("Preloading " + str(len(preload_dict.item()['front'])) + " sequences from " + preload_file)

    def __len__(self):
        """Returns the length of the dataset. """
        if self.is_top:
            return len(self.top_list)
        else:
            return len(self.front_list)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        data['fronts'] = []
        data['lefts'] = []
        data['rights'] = []
        data['tops'] = []

        data['road_para'] = torch.reshape(self.road_para[index], (1,11))
        data['road_para'] = data['road_para'].repeat(self.seq_len, 1)
        data['ego'] = self.gt_ego[index]
        data['actor'] = self.gt_actor[index]
        data['id'] = self.id[index]
        data['variants'] = self.variants[index]

        if not self.is_top:
            if self.training:
                sample_idx = random.randint(0, len(self.front_list[index])-1)
            else:
                sample_idx = len(self.front_list[index])//2

        if self.viz:
            data['img_front'] = []

        if self.is_top:
            seq_tops = self.top[index]
        else:
            seq_fronts = self.front_list[index][sample_idx]
            if not self.front_only:
                seq_lefts = self.left_list[index][sample_idx]
                seq_rights = self.right_list[index][sample_idx]


        if self.lss:
        ########## LSS
            for i in range(self.num_cam):

                post_rot = torch.eye(2)
                post_tran = torch.zeros(2)

                # augmentation (resize, crop, horizontal flip, rotate)
                resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
                post_rot2, post_tran2 = lss_transform(post_rot, post_tran,
                                                         resize=resize,
                                                         resize_dims=resize_dims,
                                                         crop=crop,
                                                         flip=flip,
                                                         rotate=rotate,
                                                         )
                # for convenience, make augmentation matrices 3x3
                post_tran = torch.zeros(3)
                post_rot = torch.eye(3)
                post_tran[:2] = post_tran2
                post_rot[:2, :2] = post_rot2

                # imgs.append(normalize_img(img.convert('RGB')))

                post_rots.append(post_rot)
                post_trans.append(post_tran)

        

        ######################################
        for i in range(self.seq_len):
            if self.is_top:
                data['tops'].append(torch.from_numpy(np.array(
                    scale_and_crop_image(Image.open(seq_tops[i]).convert('RGB')))))
            else:
                if self.lss:
                    data['fronts'].append(torch.from_numpy(np.array(
                        lss_img(Image.open(seq_fronts[i]).convert('RGB'), post_rot[0], post_tran[0], resize, resize_dims, crop, flip, rotate))))

                else:
                    data['fronts'].append(torch.from_numpy(np.array(
                        scale_and_crop_image(Image.open(seq_fronts[i]).convert('RGB'), self.scale))))
                if self.viz:
                    data['img_front'].append(np.float32(np.array(scale(Image.open(seq_fronts[i]).convert('RGB'), self.scale)))/255)
                    data['img_front'].append(np.float32(np.array(Image.open(seq_fronts[i]).convert('RGB')))/255)

                if not self.front_only:
                    if self.lss:
                        data['lefts'].append(torch.from_numpy(np.array(
                        lss_img(Image.open(seq_lefts[i]).convert('RGB'), post_rot[1], post_tran[1], resize, resize_dims, crop, flip, rotate))))
                        data['rights'].append(torch.from_numpy(np.array(
                        lss_img(Image.open(seq_rights[i]).convert('RGB'), post_rot[2], post_tran[2], resize, resize_dims, crop, flip, rotate))))

                    else:
                        data['lefts'].append(torch.from_numpy(np.array(
                            scale_and_crop_image(Image.open(seq_lefts[i]).convert('RGB'), self.scale))))
                        data['rights'].append(torch.from_numpy(np.array(
                            scale_and_crop_image(Image.open(seq_rights[i]).convert('RGB'), self.scale))))

        if self.lss:
            data['post_tran'] = torch.stack(post_tran)
            data['post_rot'] = torch.stack(post_rot)
        # start_idx = random.choice(data['start_idx']) 
        # for i in range(start_idx, len(seq_fronts), data['step']):
        #     data['fronts'].append(torch.from_numpy(np.array(
        #         scale_and_crop_image(Image.open(seq_fronts[i]).convert('RGB')))))
        # for d in data['fronts']:
        #     print(d.shape)
        # print(len(data['fronts']))

        return data

    def sample_augmentation(self):
        H, W = 720, 1280
        fH, fW = 128, 352
 
        resize = max(fH/H, fW/W)
        resize_dims = (int(W*resize), int(H*resize))
        newW, newH = resize_dims
        crop_h = int((1 - np.mean((0.0, 0.22)))*newH) - fH
        crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        flip = False
        rotate = 0
        return resize, resize_dims, crop, flip, rotate

def lss_transform(post_rot, post_tran,
                  resize, resize_dims, crop,
                  flip, rotate):

    # post-homography transformation
    post_rot *= resize
    post_tran -= torch.Tensor(crop[:2])

    A = get_rot(rotate/180*np.pi)
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A.matmul(-b) + b
    post_rot = A.matmul(post_rot)
    post_tran = A.matmul(post_tran) + b

    return post_rot, post_tran

def lss_img(img, post_rot, post_tran,
                  resize, resize_dims, crop,
                  flip, rotate):
    # adjust image
    img = img.resize(resize_dims)
    img = img.crop(crop)

    img = img.rotate(rotate)

    return img

def scale_and_crop_image(image, scale=4.0, crop=256):
    """
    Scale and crop a PIL image, returning a channels-first numpy array.
    """
    # image = Image.open(filename)
    (width, height) = (int(image.width // scale), int(image.height // scale))

    im_resized = image.resize((width, height))
    image = np.asarray(im_resized)
    # start_x = height//2 - crop//2
    # start_y = width//2 - crop//2
    # cropped_image = image[start_x:start_x+crop, start_y:start_y+crop]
    # cropped_image = np.transpose(cropped_image, (2,0,1))
    cropped_image = np.transpose(image, (2,0,1))

    return cropped_image

def scale(image, scale=4.0):

    (width, height) = (int(image.width // scale), int(image.height // scale))
    im_resized = image.resize((width, height))
    image = np.asarray(im_resized)
    
    return image


def get_multi_class(gt_list, s_id, v_id):   
    road_type = {'i-': 0, 't1': 1, "t2": 2, "t3": 3, 's-': 4, 'r-': 5}

    # ego_table = {'e:z1-z1': 0, 'e:z1-z2': 1, 'e:z1-z3':2, 'e:z1-z4': 3,
    #                 'e:s-s': 4, 'e:s-sl': 5, 'e:s-sr': 6,
    #                 'e:ri-r1': 7, 'e:r1-r2': 8, 'e:r1-ro':9}

    ego_table = {'e:z1-z1': 0, 'e:z1-z2': 1, 'e:z1-z3':2, 'e:z1-z4': 3}

    # actor_table = {'c:z1-z1': 0, 'c:z1-z2': 1, 'c:z1-z3':2, 'c:z1-z4':3,
    #                 'c:z2-z1': 4, 'c:z2-z2':5, 'c:z2-z3': 6, 'c:z2-z4': 7,
    #                 'c:z3-z1': 8, 'c:z3-z2': 9, 'c:z3-z3': 10, 'c:z3-z4': 11,
    #                 'c:z4-z1': 12, 'c:z4-z2': 13, 'c:z4-z3': 14, 'c:z4-z4': 15,

    #                 'b:z1-z1': 16, 'b:z1-z2': 17, 'b:z1-z3': 18, 'b:z1-z4': 19,
    #                 'b:z2-z1': 20, 'b:z2-z2':21, 'b:z2-z3': 22, 'b:z2-z4': 23,
    #                 'b:z3-z1': 24, 'b:z3-z2': 25, 'b:z3-z3': 26, 'b:z3-z4': 27,
    #                 'b:z4-z1': 28, 'b:z4-z2': 29, 'b:z4-z3': 30, 'b:z4-z4': 31,

    #                 'c:s-s': 32, 'c:s-sl': 33, 'c:s-sr': 34,
    #                 'c:sl-s': 35, 'c:sl-sl': 36,
    #                 'c:sr-s': 37, 'c:sr-sr': 38,
    #                 'c:jl-jr': 39, 'c:jr-jl': 40,

    #                 'b:s-s': 41, 'b:s-sl': 42, 'b:s-sr': 43,
    #                 'b:sl-s': 44, 'b:sl-sl': 45,
    #                 'b:sr-s': 46, 'b:sr-sr': 47,
    #                 'b:jl-jr': 48, 'b:jr-jl': 49,

    #                 'p:c1-c2': 50, 'p:c1-c4': 51, 'p:c1-cl': 52,  'p:c1-cf': 53, 
    #                 'p:c2-c1': 54, 'p:c2-c3': 55, 'p:c2-cl': 56,
    #                 'p:c3-c2': 57, 'p:c3-c4': 58, 'p:c3-cr': 59,
    #                 'p:c4-c1': 60, 'p:c4-c3': 61, 'p:c4-cr': 62, 'p:c4-cf': 63,
    #                 'p:cf-c1': 64, 'p:cf-c4': 65,
    #                 'p:cl-c1': 66, 'p:cl-c2': 67,
    #                 'p:cr-c3': 68, 'p:cr-c4': 69,
    #                 'p:jl-jr': 70, 'p:jr-jl': 71,

    #                 'c:ri-r1': 72, 'c:rl-r1': 73, 'c:r1-r2': 74, 'c:r1-ro': 75, 'c:ri-r2': 76, 'c:r1-r2': 77,
    #                 'b:ri-r1': 78, 'b:rl-r1': 79, 'b:r1-r2': 80, 'b:r1-ro': 81, 'b:ri-r2': 82, 'b:r1-r2': 83}


    actor_table = {'z1-z1': 0, 'z1-z2': 1, 'z1-z3':2, 'z1-z4':3,
                    'z2-z1': 4, 'z2-z2':5, 'z2-z3': 6, 'z2-z4': 7,
                    'z3-z1': 8, 'z3-z2': 9, 'z3-z3': 10, 'z3-z4': 11,
                    'z4-z1': 12, 'z4-z2': 13, 'z4-z3': 14, 'z4-z4': 15,

                    'c1-c2': 16, 'c1-c4': 17, 'c1-cl': 18,  'c1-cf': 19, 
                    'c2-c1': 20, 'c2-c3': 21, 'c2-cl': 22,
                    'c3-c2': 23, 'c3-c4': 24, 'c3-cr': 25,
                    'c4-c1': 26, 'c4-c3': 27, 'c4-cr': 28, 'c4-cf': 29,
                    'cf-c1': 30, 'cf-c4': 31,
                    'cl-c1': 32, 'cl-c2': 33,
                    'cr-c3': 34, 'cr-c4': 35}



    road_class = s_id.split('_')[1][:2]
    road_class = road_type[road_class]

    # z1, z2, z3, z4, 0~3
    # c1, c2, c3, c4, cf, cl, cr, 4~10
    # s, sl, sr, jl, jr, 11~15
    # r1, r2, ri, ro 16~19


    if road_class == 0:
        road_para = [1,1,1,1,
                    1,1,1,1,0,0,0,
                    ]

    elif road_class == 1:
        road_para = [1,1,0,1,
                    1,0,0,1,1,0,0,
                    ]

    elif road_class == 2:
        road_para = [1,1,1,0,
                    1,1,0,0,0,1,0,
                    ]

    elif road_class == 3:
        road_para = [1,0,1,1,
                    0,0,1,1,0,0,1,
                    ]

    elif road_class == 4:
        road_para = [0,0,0,0,
                    0,0,0,0,0,0,0,
                    1,1,1,1,1,
                    0,0,0,0]
        return

    elif road_class == 5:
        road_para = [0,0,0,0,
                    0,0,0,0,0,0,0,
                    0,0,0,0,0,
                    1,1,1,1]
        return
    road_para = torch.FloatTensor(road_para)

    ego_class = None
    actor_class = [0]*36
    for gt in gt_list:
        gt = gt.lower()
        if gt[0] == 'e':
            ego_class = ego_table[gt]
        else:
            gt = gt[2:]
            if not gt in actor_table.keys():
                print(gt)
            actor_class[actor_table[gt]] = 1
    if ego_class == None:
        return

    ego_label = torch.tensor(ego_class)
    actor_label = torch.FloatTensor(actor_class)


    return road_para, ego_label, actor_label



