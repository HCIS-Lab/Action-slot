# [CVPR 2024] Action-slot: Visual Action-centric Representations for Multi-label Atomic Activity Recognition in Traffic Scenes

<sup>1</sup>[Chi-Hsi Kung](https://hankkung.github.io/website/),  <sup>1,</sup>[Shu-Wei Lu](https://www.linkedin.com/in/shu-wei-lu/),  <sup>2</sup>[Yi-Hsuan Tsai](https://sites.google.com/site/yihsuantsai/),  <sup>1</sup>[Yi-Ting Chen](https://sites.google.com/site/yitingchen0524)

<sup>1</sup>National Yang Ming Chiao Tung University,  <sup>2</sup>Google

[[CVPR](https://openaccess.thecvf.com/content/CVPR2024/papers/Kung_Action-slot_Visual_Action-centric_Representations_for_Multi-label_Atomic_Activity_Recognition_in_CVPR_2024_paper.pdf)] [[arxiv](https://arxiv.org/abs/2311.17948)] [[Project Page](https://hcis-lab.github.io/Action-slot/)] [[Challenge ECCV'24](https://sites.google.com/view/road-eccv2024/challenge?authuser=0)]

This repository contains the official code for training and evaluating baselines presented in the paper.

## :fire: News
- **`2024/06/03`**: **Challenge** for the **Atomic Activity Recognition Challenge** is launched, [come win the prize](https://eval.ai/web/challenges/challenge-page/2301/overview)!
- **`2024/05/26`**: **Code** for the **Atomic Activity Recognition Challenge** is released!
- **`2024/04/12`**: **Challenge for Atomic Activity Recognition**  got accepted by **ECCV 2024**! Check more details [here](https://sites.google.com/view/road-eccv2024/home?authuser=0).
- **`2024/02/27`**: Action-slot got accepted by **CVPR 2024**!

## 🚀 Installation
Create and activate the conda environment:
   ```
   conda create --name action_slot python=3.7
   pip install -r requirements.txt
   ```

## 📦 Datasets Download

**TACO**: [[One-drive](https://nycu1-my.sharepoint.com/:f:/g/personal/ychen_m365_nycu_edu_tw/EnRg1zT7CeZGg3Ju2TIP1j8B0NB0fCpYsjGQBc0Tcf2H6w?e=FGJvTc)]

The TACO dataset consists of 13 folders of videos (scenarios), which are separated based on different maps (e.g., Town01, Town02...) in the CARLA simulator and different collecting methods (i.e., autopilot (AP), scenario runner (runner), and manual collecting [1] (i.e., interactive & non-interactive)). We use data collected in Town03 as val set and Town10HD as the test set. Please refer to the supplementary material for more dataset details.
Note that we use both train and val splits for training in our benchmark. Please also note that there's an updated number of videos in each split, train: 2753, val: 977, test: 1446.

**OATS** [2] [[Website](https://usa.honda-ri.com/oats)]

[1] Kung et al. "RiskBench: A Scenario-based Benchmark for Risk Identification". ICRA 2024.

[2] Agarwal and Chen "Ordered Atomic Activity for Fine-grained Interactive Traffic Scenario Understanding". ICCV 2023


## :fire::fire::fire: 3rd ROAD Challenge for Atomic Activity Recognition
We provide the script to generate .pkl prediction file for the challenge on TACO. The script takes Action-slot as an example:
```
cd scripts/
python generate_test_results.py --split [val/test] --cp path_checkpoint --model_name action_slot --backbone x3d --bg_slot --bg_mask --allocated_slot  --root path_taco 
```


## 🌐 Train & Evaluation on TACO
Training
```
# Action-slot
python train_taco.py --dataset taco --root [path_to_TACO] --model_name action_slot --num_slots 64 --bg_slot --bg_mask --action_attn_weight 1 --allocated_slot --bg_attn_weight 0.5

# X3D
python train_taco.py --dataset taco --root [path_to_TACO] --model_name x3d 
```

Evaluation
```
# Action-slot
python eval_taco.py --cp [path_to_checkpoint] --root [path_to_TACO] --dataset taco --model_name action_slot --num_slots 64 --bg_slot --allocated_slot

# X3D
python eval_taco.py --root [path_to_TACO] --cp [path_to_checkpoint] --dataset taco --model_name x3d 
```

## 🌐 Train & Evaluation on OATS
```
# Action-slot
python train_oats.py --dataset oats --oats_test_split s1 --model_name action_slot --epochs 50 --num_slots 35 --bg_slot --bg_mask --action_attn_weight 0.1 --allocated_slot --bg_attn_weight 0.1 --ego_loss_weight 0

python eval_oats.py --cp [path_to_checkpoint] --dataset oats --oats_test_split s3  --root [path_to_dataset] --model_name action_slot --allocated_slot --backbone x3d --num_slots 35 --bg_slot 
```

## 🌐 Train & Evaluation on nuScenes
```
# train from scratch
python train_nuscenes.py --dataset nuscenes --root [path]/nuscenes/trainval/samples --model_name action_slot --num_slots 64 --bg_slot --bg_mask --action_attn_weight 1 --allocated_slot --bg_attn_weight 0.5 --bce_pos_weight 7

# transfer learning: TACO -> nuScenes
python train_nuscenes.py --pretrain taco --root [path]/nuscenes/trainval/samples --cp [path_to_checkpoint] --dataset nuscenes --model_name action_slot --num_slots 64 --bg_slot --bg_mask --action_attn_weight 1 --allocated_slot --bg_attn_weight 0.5 --bce_pos_weight 20 --root /media/hcis-s20/SRL/nuscenes/trainval/samples

# transfer learning: OATS -> nuScenes
python train_nuscenes.py --pretrain oats --root [path]/nuscenes/trainval/samples --cp [path_to_checkpoint] --dataset nuscenes --model_name action_slot--num_slots 64 --bg_slot --bg_mask --action_attn_weight 1 --allocated_slot --bg_attn_weight 0.5 --bce_pos_weight 15  
```

## 📊 Attention Visualization
![image](https://github.com/HCIS-Lab/Action-slot/blob/main/img/taco_attn.gif)
```
python eval_taco.py --cp [path_to_checkpoint] --plot --dataset taco --root [path_to_TACO] --model_name action_slot --num_slots 64 --bg_slot --allocated_slot --plot_threshold 0.2
```

## Citation
```
@inproceedings{kung2023action,
  title={Action-slot: Visual Action-centric Representations for Multi-label Atomic Activity Recognition in Traffic Scenes},
  author={Kung, Chi-Hsi and Lu, Shu-Wei and Tsai, Yi-Hsuan and Chen, Yi-Ting},
  journal={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```

### Acknowledgement
* Slot attention is adapted from [Discovering Object that Can Move](https://github.com/zpbao/Discovery_Obj_Move)
* DeepLabV3+ is adapted from [DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch)
* VideoMAE is adapted from [VideoMAE](https://github.com/MCG-NJU/VideoMAE?tab=readme-ov-file)
