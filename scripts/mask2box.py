import os
import json
import torch
from torchvision.ops.boxes import masks_to_boxes
import cv2
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

town_list = ['interactive', 'non-interactive', 'ap_Town01', 
        'ap_Town02','ap_Town03', 'ap_Town04', 'ap_Town05', 'ap_Town06', 'ap_Town07', 'ap_Town10HD', 
        'runner_Town03','runner_Town05', 'runner_Town10HD']

def read_json(f):
    with open(f) as json_data:
        data = json.load(json_data)
    json_data.close()
    return data

def produce_box(root, types ,write_video=False,save_bbox=False):
    instance_mode = True
    W,H = 768,256
    
    def get_ids(mask, ver, area_threshold=75):
        """
            Args:
                mask: instance image
        """
        h,w = mask.shape[1:]
        mask_2 = torch.zeros((2,h,w), device="cuda:0")
        mask_2[0] = mask[0]
        mask_2[1] = mask[1]+mask[2]*256
        
        if ver == 0 :
            condition = mask_2[0]== 4
            condition += mask_2[0]== 10
        else:
            condition = mask_2[0]== 14 # Car
            condition += mask_2[0]== 15 # Truck
            condition += mask_2[0]== 16 # Bus
            condition += mask_2[0]== 12 # Pedestrian
            condition += mask[0]== 18 # Motorcycle
            condition += mask[0]== 19 # Bicycle
        background_mask = condition.clone()
        if ver == 0 :
            background_mask += mask_2[0]==6
            background_mask += mask_2[0]==7
            background_mask += mask_2[0]==8
            background_mask += mask_2[0]==14
            background_mask += mask_2[0]==12
            background_mask += mask_2[0]==18
        else:
            background_mask += mask_2[0]==24
            background_mask += mask_2[0]==25
            background_mask += mask_2[0]==1
            background_mask += mask_2[0]==2
            # background_mask += mask_2[0]==7
            # background_mask += mask_2[0]==8
        background_mask = ~background_mask

        obj_ids = torch.unique(mask_2[1,condition])
        masks = mask_2[1] == obj_ids[:, None, None]
        masks = masks*condition
        area_condition = masks.long().sum((1,2))>=area_threshold
        masks = masks[area_condition]
        
        obj_ids = obj_ids[area_condition].type(torch.int).cpu().numpy()
        boxes = masks_to_boxes(masks).type(torch.int16).cpu().numpy()
        assert len(obj_ids) == len(boxes)

        condition = torch.nn.functional.interpolate(condition[None][None].to(float),size=(32,96))[0,0]
        background_mask = torch.nn.functional.interpolate(background_mask[None][None].to(float),size=(32,96))[0,0]
        return boxes, obj_ids, condition.to(bool).cpu().numpy(), background_mask.to(bool).cpu().numpy()

    def process_static(path, s_type):
        """
            return
                obstacle
                    dict: id -> box coords
                map static object
                ego_id
        """
        tmp_obstacles = None

        actor_attribute = read_json(os.path.join(path,"actor_attribute.json"))
        # obstacles
        if s_type == "obstacle":
            tmp_obstacles = []
            obstacle_infos = actor_attribute["obstacle"]
            for ob_id in obstacle_infos:
                tmp_obstacles.append(ob_id)

        return tmp_obstacles, actor_attribute["ego_id"]

    # for scenario types
    #types = ['collision','interactive','non-interactive','obstacle']
    for s_type in os.listdir(root):
        if s_type not in types:
            continue
        print(s_type)
        for s_id in tqdm(os.listdir(os.path.join(root,s_type)), position=0, leave=False):
            if not os.path.isdir(os.path.join(root,s_type,s_id)):
                continue
            for variant in tqdm(os.listdir(os.path.join(root,s_type,s_id,'variant_scenario')), position=1, leave=False):
                if save_bbox:
                    out_box = {}
                if write_video:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out = cv2.VideoWriter(os.path.join(root,'demo.mp4'), fourcc, 12.0, (W,  H))
                variant_path = os.path.join(root,s_type,s_id,'variant_scenario',variant)
                # bbox_path = os.path.join(root,s_type,s_id,'variant_scenario',variant)

                # print(variant_path)
                # process static objects
                # obstacle_ids, ego_id = process_static(variant_path,s_type)
                try:
                    if not os.path.exists(os.path.join(variant_path,'mask')):
                        os.makedirs(os.path.join(variant_path,'mask'))
                        os.makedirs(os.path.join(variant_path,'mask','background'))
                        os.makedirs(os.path.join(variant_path,'mask','object'))
                    rgb_imgs = sorted(os.listdir(os.path.join(variant_path,'rgb','downsampled')))
                except:
                    continue
                flag = False
                # check version beforer iter
                for rgb_file in rgb_imgs:
                    frame_id = rgb_file[:-4]
                    if not os.path.isfile(os.path.join(variant_path,'instance_segmentation','ins_front',frame_id+".png")):
                        continue
                    instance = cv2.imread(os.path.join(variant_path,'instance_segmentation','ins_front',frame_id+".png"))
                    instance = torch.flip(torch.from_numpy(instance).type(torch.int).permute(2,0,1),[0])
                    instance = instance.to("cuda:0")[0,:(H//8)*3]
                    ver = ((instance == 13).sum() - (instance == 11).sum()).cpu().numpy()
                    if ver == 0:
                        ver = ((instance == 1).sum() - (instance == 3).sum()).cpu().numpy()
                    
                    if ver == 0:
                        flag = True
                    elif ver>0:
                        ver = 0
                    else:
                        ver = 1
                    break
                if flag: 
                    print('\n',variant_path)
                    continue
                for rgb_file in rgb_imgs:
                    frame_id = rgb_file[:-4]
                    if not os.path.isfile(os.path.join(variant_path,'instance_segmentation','ins_front',frame_id+".png")):
                        continue
                    instance = cv2.imread(os.path.join(variant_path,'instance_segmentation','ins_front',frame_id+".png"))
                    if write_video:
                        img = cv2.imread(os.path.join(variant_path,'rgb','downsampled',rgb_file))
                    if save_bbox:
                        out_box[frame_id] = {}
                    instance = torch.flip(torch.from_numpy(instance).type(torch.int).permute(2,0,1),[0])
                    instance = instance.to("cuda:0")[None].to(float)
                    instance = torch.nn.functional.interpolate(instance,size=(H//2,W//2)).to(int)[0]
                    if instance_mode:
                        bboxs, obj_ids, obj_mask, background_mask = get_ids(instance, ver)
                        for box, actor_id in zip(bboxs, obj_ids):
                            box = [int(b) for b in box.astype(int)]
                            # if actor_id == ego_id:
                            #     continue
                            if write_video:
                                cv2.rectangle(img, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (255,0,0, 255), 1)  
                            if save_bbox:
                                out_box[frame_id][int(actor_id)] = box
                        background_mask = Image.fromarray(np.uint8(background_mask*255))
                        background_mask.save(os.path.join(variant_path,'mask','background',frame_id)+'.png')
                        # obj_mask = Image.fromarray(np.uint8(obj_mask*255))
                        # obj_mask.save(os.path.join(variant_path,'mask','object',frame_id)+'.png')

                    if write_video:
                        out.write(img)
                if save_bbox:
                    # with open(os.path.join(variant_path,'bbox.json'), 'w') as f:
                    #     json.dump(out_box, f)
                    with open(os.path.join(variant_path,'bbox.json'), 'w') as f:
                        json.dump(out_box, f)
                if write_video:
                    out.release()
                # raise BaseException
if __name__ == '__main__':



    parser = argparse.ArgumentParser()
    
    parser.add_argument("-r",
                        "--root",
                        default="/media/user/data/FinalFinalRiskBenchDataset/data_collection",
                        type=str,
                        )
    parser.add_argument("-s",
                        "--scenario",
                        default="interactive",
                        type=str,
                        )
    args = parser.parse_args()


    types = [f'{args.scenario}']
    
    produce_box(args.root,town_list,save_bbox=True,write_video=False)
    
    # old
        
