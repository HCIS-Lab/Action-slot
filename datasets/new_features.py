import os
import json
from PIL import Image
import threading

import numpy as np
import torch 
from torch.utils.data import Dataset
from tqdm import tqdm
import sys
import json 
import random
from tool import get_rot
sys.path.append('/data/hanku/Interaction-benchmark/models')
sys.path.append('/data/hanku/Interaction-benchmark/models/MaskFormer')
sys.path.append("/data/hanku/Interaction_benchmark/models/MaskFormer/configs/mapillary-vistas-65")


from MaskFormer.demo.demo import get_maskformer

###LSS
# from pyquaternion import Quaternion

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

def save_feature(features, path_list):
    for i in range(features.shape[0]):
        f = features[i,:,:,:]
        f = f.numpy()
        np.save(path_list[i], f)



torch.cuda.empty_cache()


# parser = argparse.ArgumentParser()
# parser.add_argument('--model', type=str, default='r5')
# args = parser.parse_args()
# print(args)

# if args.model == 'r5':
#     model = get_maskformer().cuda()
 
scale=1
# root='/media/hankung/ssd/carla_13/CARLA_0.9.13/PythonAPI/examples/data_collection'
# save_root='/media/hankung/ssd/carla_13/CARLA_0.9.13/PythonAPI/examples/data_collection'
root='/data/carla_dataset/data_collection'
save_root='/data/carla_dataset/data_collection'


# save = '/media/hankung/07FEBA3A204BD225/r4_4'
type_list = ['interactive']


if not os.path.isdir(save_root):
    os.mkdir(save_root)


model = get_maskformer().cuda()

model.eval()
for t, type in enumerate(type_list):
    basic_scenarios = [os.path.join(root, type, s) for s in os.listdir(os.path.join(root, type))]
    save = os.path.join(save_root, type)
    if not os.path.isdir(save):
        os.mkdir(save)
    # iterate scenarios
    print('searching data')
    for s in tqdm(basic_scenarios, file=sys.stdout):
        print(s)
        # a basic scenario
        scenario_id = s.split('/')[-1]
        if not os.path.isdir(os.path.join(save, scenario_id)):
            os.mkdir(os.path.join(save, scenario_id))
        save_scen = os.path.join(save, scenario_id, 'variant_scenario')
        if not os.path.isdir(save_scen):
            os.mkdir(save_scen)
        # if road_class != 5:
        variants_path = os.path.join(s, 'variant_scenario')
        variants = [os.path.join(variants_path, v) for v in os.listdir(variants_path) if os.path.isdir(os.path.join(variants_path, v))]
       
        for v in variants:
            print(v)
            v_id = v.split('/')[-1]
            save_v = os.path.join(save_scen, v_id)

            if not os.path.isdir(save_v):
                os.mkdir(save_v)

            # a data sample
            fronts = []

            lefts = []
            rights = []
            tops = []

            # f_path = v+"/rgb_f/" + '_'+str(int(scale))
            # if not os.path.isdir(f_path):
            #     os.mkdir(f_path)
            if not os.path.isdir(save_v + "/"+str(scale)):
                os.mkdir(save_v + "/r5_"+str(scale))

            if os.path.isdir(v+"/rgb/front/"):
                fronts = [v+"/rgb/front/"+ img for img in os.listdir(v+"/rgb/front/") if os.path.isfile(v+"/rgb/front/"+ img)]
                if not os.path.isdir(save_v + "/r5_"+str(scale)+"/front/"):
                    os.mkdir(save_v + "/r5_"+str(scale)+"/front/")
                n_fronts = [save_v + "/r5_"+str(scale)+"/front/"+ img[:9]+'npy' for img in os.listdir(v+"/rgb/front/")]
            # ---------------------
            # if os.path.isdir(v+"/rgb/right/"):
            #     rights = [v+"/rgb/right/"+ img for img in os.listdir(v+"/rgb/right/") if os.path.isfile(v+"/rgb/right/"+ img)]
            #     if not os.path.isdir(save_v + "/r5_"+str(scale)+"/right/"):
            #         os.mkdir(save_v + "/r5_"+str(scale)+"/right/")
            #     n_rights = [save_v +"/r5_"+str(scale)+"/right/"+ img[:9]+'npy' for img in os.listdir(v+"/rgb/right/")]
            # # -----------------------
            # if os.path.isdir(v+"/rgb/left/"):
            #     lefts = [v+"/rgb/left/"+ img for img in os.listdir(v+"/rgb/left/") if os.path.isfile(v+"/rgb/left/"+ img)]
            #     if not os.path.isdir(save_v + "/r5_"+str(scale)+"/left/"):
            #         os.mkdir(save_v + "/r5_"+str(scale)+"/left/")
            #     n_lefts = [save_v +"/r5_"+str(scale)+"/left/"+ img[:9]+'npy' for img in os.listdir(v+"/rgb/left/")]


            scale = float(scale)
            front_tensor = []
            for i in range(len(fronts)):
                try:
                    front_tensor.append(torch.from_numpy(np.array(
                        scale_and_crop_image(Image.open(fronts[i]).convert('RGB'), scale))).float())
                except:

                    print(fronts[i])
                    continue
            # left_tensor = []
            # for i in range(len(lefts)):
            #     try:
            #         left_tensor.append(torch.from_numpy(np.array(
            #             scale_and_crop_image(Image.open(lefts[i]).convert('RGB'), scale))).float())
            #     except:
            #         continue
            # right_tensor = []
            # for i in range(len(rights)):
            #     try:
            #         right_tensor.append(torch.from_numpy(np.array(
            #             scale_and_crop_image(Image.open(rights[i]).convert('RGB'), scale))).float())
            #     except:
            #         continue
            l = len(front_tensor)//4
            front_tensor_ls = []
            # left_tensor_ls = []
            # right_tensor_ls = []
            try:
                front_tensor_ls.append(torch.stack(front_tensor[:l]))
                front_tensor_ls.append(torch.stack(front_tensor[l:2*l]))
                front_tensor_ls.append(torch.stack(front_tensor[2*l:3*l]))
                front_tensor_ls.append(torch.stack(front_tensor[3*l:]))

                # left_tensor_ls.append(torch.stack(left_tensor[:l]))
                # left_tensor_ls.append(torch.stack(left_tensor[l:2*l]))
                # left_tensor_ls.append(torch.stack(left_tensor[2*l:3*l]))
                # left_tensor_ls.append(torch.stack(left_tensor[3*l:]))

                # right_tensor_ls.append(torch.stack(right_tensor[:l]))
                # right_tensor_ls.append(torch.stack(right_tensor[l:2*l]))
                # right_tensor_ls.append(torch.stack(right_tensor[2*l:3*l]))
                # right_tensor_ls.append(torch.stack(right_tensor[3*l:]))
            except:
                print('empty stack')
                continue

            
            with torch.no_grad():   
                for i, t in enumerate(front_tensor_ls):
                    t = t.to('cuda', dtype=torch.float32)
                    t = (t - model.pixel_mean) / model.pixel_std
                    front_tensor_ls[i] = model.backbone(t)['res5'].cpu()
                f_features = torch.cat((front_tensor_ls[0],front_tensor_ls[1],front_tensor_ls[2],front_tensor_ls[3]), dim=0)
                # f_features = torch.stack(front_tensor_ls)
            t_f = threading.Thread(target=save_feature, args=(f_features, n_fronts))
            t_f.start()


            # with torch.no_grad():   
            #     for i, t in enumerate(right_tensor_ls):
            #         t = t.to('cuda', dtype=torch.float32)
            #         t = (t - model.pixel_mean) / model.pixel_std
            #         right_tensor_ls[i] = model.backbone(t)['res5'].cpu()
            #     r_features = torch.cat((right_tensor_ls[0],right_tensor_ls[1],right_tensor_ls[2],right_tensor_ls[3]), dim=0)
            #     # r_features = torch.stack(right_tensor_ls)
            # t_r = threading.Thread(target=save_feature, args=(r_features, n_rights))
            # t_r.start()

            # with torch.no_grad():   
            #     for i, t in enumerate(left_tensor_ls):
            #         t = t.to('cuda', dtype=torch.float32)
            #         t = (t - model.pixel_mean) / model.pixel_std
            #         left_tensor_ls[i] = model.backbone(t)['res5'].cpu()
            #     l_features = torch.cat((left_tensor_ls[0],left_tensor_ls[1],left_tensor_ls[2],left_tensor_ls[3]), dim=0)
            #     # l_features = torch.stack(left_tensor_ls)
            # t_l = threading.Thread(target=save_feature, args=(l_features, n_lefts))
            # t_l.start()

            # ---------------------------------------------
            # left_tensor = left_tensor.to('cuda', dtype=torch.float32)
            # with torch.no_grad():   
            #     left_tensor = (left_tensor - model.pixel_mean) / model.pixel_std
            #     l_features = model.backbone(left_tensor)['res5']

            # t_l = threading.Thread(target=save_feature, args=(l_features, n_lefts))
            # t_l.start()


            # # ---------------------------------------------
            # right_tensor = right_tensor.to('cuda', dtype=torch.float32)
            # with torch.no_grad():   
            #     right_tensor = (right_tensor - model.pixel_mean) / model.pixel_std
            #     r_features = model.backbone(right_tensor)['res5']

            # t_r = threading.Thread(target=save_feature, args=(r_features, n_rights))
            # t_r.start()


            t_f.join()
            t_l.join()
            t_r.join()

            front_tensor = []
            # left_tensor = []
            # right_tensor = []


