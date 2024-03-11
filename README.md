# Action-slot

**[CVPR 2024] Action-slot: Visual Action-centric Representations for Multi-label Atomic Activity Recognition in Traffic Scenes**

<sup>1</sup>[Chi-Hsi Kung](https://hankkung.github.io/website/),  <sup>1,</sup>[Shu-Wei Lu](https://www.linkedin.com/in/shu-wei-lu/),  <sup>2</sup>[Yi-Hsuan Tsai](https://sites.google.com/site/yihsuantsai/),  <sup>1</sup>[Yi-Ting Chen](https://sites.google.com/site/yitingchen0524)

<sup>1</sup>National Yang Ming Chiao Tung University,  <sup>2</sup>Google

[[arxiv](https://arxiv.org/abs/2311.17948)] [[Project Page](https://hcis-lab.github.io/Action-slot/)]

This repository contains code for training and evaluating baselines presented in the paper.

## 🚀 Installation
Create and activate the conda environment:
   ```
   pip install -e . 
   ```
## 📦 Datasets Download

**TACO** [[One Drive](https://nycu1-my.sharepoint.com/personal/ychen_m365_nycu_edu_tw/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fychen%5Fm365%5Fnycu%5Fedu%5Ftw%2FDocuments%2FTACO&ga=1)]
The TACO dataset consists of 13 folders of videos (scenarios), which are separated based on different maps (e.g., Town01, Town02...) in the CARLA simulator and different collecting methods (i.e., autopilot (AP), scenario runner (runner), and manual collecting [1] (i.e., interactive & non-interactive)). Please refer to the supplementary material for more dataset details.

[1] Kung et al., "RiskBench: A Scenario-based Benchmark for Risk Identification". ICRA 2024.
**OATS** [[Website](https://usa.honda-ri.com/oats)]


## 🌐 Train & Evaluation on TACO
Training
```
# Action-slot
python train_taco.py --dataset taco --root [path_to_TACO] --model_name action_slot --num_slots 64\
 --bg_slot --bg_mask --action_attn_weight 1 --allocated_slot --bg_attn_weight 0.5

# X3D
python train_taco.py --dataset taco --root [path_to_TACO] --model_name x3d 
```

Evaluation
```
# Action-slot
python eval_taco.py --cp [path_to_checkpoint] --root [path_to_TACO] --dataset taco\
 --model_name action_slot --num_slots 64 --bg_slot --allocated_slot

# X3D
python eval_taco.py --root [path_to_TACO] --cp [path_to_checkpoint] --dataset taco --model_name x3d 
```

## 🌐 Train & Evaluation on OATS
```
# Action-slot
python train_oats.py --dataset oats --oats_test_split s1 --model_name action_slot --epochs 50\
   --num_slots 35 --bg_slot --bg_mask --action_attn_weight 0.1 --allocated_slot\
   --bg_attn_weight 0.1 --ego_loss_weight 0

python eval_oats.py --cp [path_to_checkpoint] --dataset oats --oats_test_split s3  --root [path_to_dataset]\
    --model_name action_slot --allocated_slot --backbone x3d --num_slots 35 --bg_slot 
```

## 🌐 Train & Evaluation on nuScenes
```
# train from scratch
python train_nuscenes.py --dataset nuscenes --root [path]/nuscenes/trainval/samples\
   --model_name action_slot --num_slots 64 --bg_slot --bg_mask --action_attn_weight 1\
 --allocated_slot --bg_attn_weight 0.5 --bce_pos_weight 7

# transfer learning: TACO -> nuScenes
python train_nuscenes.py --pretrain taco --root [path]/nuscenes/trainval/samples --cp [path_to_checkpoint] --dataset nuscenes\
   --model_name action_slot --num_slots 64 --bg_slot --bg_mask --action_attn_weight 1\
   --allocated_slot --bg_attn_weight 0.5 --bce_pos_weight 20 --root /media/hcis-s20/SRL/nuscenes/trainval/samples

# transfer learning: OATS -> nuScenes
python train_nuscenes.py --pretrain oats --root [path]/nuscenes/trainval/samples --cp [path_to_checkpoint] --dataset nuscenes\
   --model_name action_slot--num_slots 64 --bg_slot --bg_mask --action_attn_weight 1 --allocated_slot --bg_attn_weight 0.5\
   --bce_pos_weight 15  
```

## 📊 Attention Visualization
![image](https://github.com/HCIS-Lab/Action-slot/blob/main/img/taco_attn.gif)
```
python eval_taco.py --cp [path_to_checkpoint] --plot --dataset taco --root [path]/nuscenes/trainval/samples\
    --model_name action_slot --num_slots 64 --bg_slot --allocated_slot --plot_threshold 0.5 
```

## Citation
```
@article{kung2023action,
  title={Action-slot: Visual Action-centric Representations for Multi-label Atomic Activity Recognition in Traffic Scenes},
  author={Kung, Chi-Hsi and Lu, Shu-Wei and Tsai, Yi-Hsuan and Chen, Yi-Ting},
  journal={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```

### Acknowledgement
* Slot attention is adapted from [Discovering Object that Can Move](https://github.com/zpbao/Discovery_Obj_Move)
* DeepLabV3+ is adapted from [DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch)
