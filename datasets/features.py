import os
import json
from PIL import Image
import threading
import torchvision.transforms as transforms

import numpy as np
import torch 
from torch.utils.data import Dataset
from tqdm import tqdm
import sys
import json 
import random
from tool import get_rot
sys.path.append('/media/hankung/ssd/retrieval/models')
sys.path.append('/media/hankung/ssd/retrieval/models/MaskFormer')
sys.path.append("/media/hankung/ssd/retrieval/models/MaskFormer/configs/mapillary-vistas-65")


from MaskFormer.demo.demo import get_maskformer


def scale(image, ds=2.0):

    (width, height) = (int(image.width // ds), int(image.height // ds))
    im_resized = image.resize((width, height), Image.ANTIALIAS)

    

    # image = np.asarray(im_resized)
    # image = np.transpose(image, (2,0,1))
    return im_resized

def to_np(v):
    for i, _ in enumerate(v):

        # transform = transforms.Compose([
        #                         transforms.ToTensor()])

        transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        v[i] = transform(v[i])
    return v


def save_feature(features, path_list):
    for i in range(features.shape[0]):
        if os.path.exists(path_list[i]):
            continue
        f = features[i,:,:,:]
        f = f.numpy()
        np.save(path_list[i], f)



torch.cuda.empty_cache()



ds= int(2)
# root='/media/hankung/ssd/carla_13/CARLA_0.9.13/PythonAPI/examples/data_collection'
# save_root='/media/hankung/ssd/carla_13/CARLA_0.9.13/PythonAPI/examples/data_collection'
root='/media/hankung/ssd/carla_13/CARLA_0.9.13/PythonAPI/examples/data_collection'
save_root='//media/hankung/ssd/carla_13/CARLA_0.9.13/PythonAPI/examples/data_collection'


# save = '/media/hankung/07FEBA3A204BD225/r4_4'
type_list = ['interactive', 'non-interactive', 'ap_Town01', 'ap_Town02', 
'ap_Town03', 'ap_Town04', 'ap_Town05', 'ap_Town06', 'ap_Town07', 'ap_Town10HD']


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
        if 'DS' in s:
            continue
        # a basic scenario
        scenario_id = s.split('/')[-1]
        if not os.path.isdir(os.path.join(save, scenario_id)):
            os.mkdir(os.path.join(save, scenario_id))
        save_scen = os.path.join(save, scenario_id, 'variant_scenario')
        if not os.path.isdir(save_scen):
            os.mkdir(save_scen)
        variants_path = os.path.join(s, 'variant_scenario')
        variants = [os.path.join(variants_path, v) for v in os.listdir(variants_path) if os.path.isdir(os.path.join(variants_path, v))]
       
        for v in variants:
            if 'DS' in v:
                continue
            v_id = v.split('/')[-1]
            save_v = os.path.join(save_scen, v_id)

            if not os.path.isdir(save_v):
                os.mkdir(save_v)

            # a data sample
            fronts = []


            # f_path = v+"/rgb_f/" + '_'+str(int(scale))
            # if not os.path.isdir(f_path):
            #     os.mkdir(f_path)
            if not os.path.isdir(save_v + "/r5_"+str(ds)):
                os.mkdir(save_v + "/r5_"+str(ds))
            else:
                continue

            if os.path.isdir(v+"/rgb/front/"):
                # fronts = []
                # for img in os.listdir(v+"/rgb/front/"):
                #     if os.path.isfile(v+"/rgb/front/"+ img) and \
                #     not os.path.exists(save_v + "/r5_"+str(ds)+"/front/"+ img[:9]+'npy'):
                #         fronts.append(v+"/rgb/front/"+ img)
                fronts = [v+"/rgb/front/"+ img for img in os.listdir(v+"/rgb/front/") if os.path.isfile(v+"/rgb/front/"+ img)]
                if not os.path.isdir(save_v + "/r5_"+str(ds)+"/front/"):
                    os.mkdir(save_v + "/r5_"+str(ds)+"/front/")
                # n_fronts = [save_v + "/r5_"+str(ds)+"/front/"+ img[:9]+'npy' for img in fronts]
                n_fronts = [save_v + "/r5_"+str(ds)+"/front/"+ img[:9]+'npy' for img in os.listdir(v+"/rgb/front/")]

            
            front_tensor = []
            for i in range(len(fronts)):
                # front_tensor.append(torch.from_numpy(np.array(
                #     scale_and_crop_image(Image.open(fronts[i]).convert('RGB'), scale))).float())
                x = Image.open(fronts[i]).convert('RGB')
                x = scale(x, 2.0)
                front_tensor.append(x)
                print(fronts[i])
            front_tensor = to_np(front_tensor)

            l = len(front_tensor)//4
            front_tensor_ls = []

            try:
                front_tensor_ls.append(torch.stack(front_tensor[:l]))
                front_tensor_ls.append(torch.stack(front_tensor[l:2*l]))
                front_tensor_ls.append(torch.stack(front_tensor[2*l:3*l]))
                front_tensor_ls.append(torch.stack(front_tensor[3*l:]))

            except:
                print('empty stack')
                continue

            
            with torch.no_grad():   
                for i, t in enumerate(front_tensor_ls):
                    t = t.to('cuda', dtype=torch.float32)
                    front_tensor_ls[i] = model.backbone(t)['res5'].cpu()
                f_features = torch.cat((front_tensor_ls[0],front_tensor_ls[1],front_tensor_ls[2],front_tensor_ls[3]), dim=0)
                # f_features = torch.stack(front_tensor_ls)
            t_f = threading.Thread(target=save_feature, args=(f_features, n_fronts))
            t_f.start()


            t_f.join()


            front_tensor = []



