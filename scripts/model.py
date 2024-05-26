import torch
from torch import nn
import numpy as np

import sys
sys.path.append('../models')

import i3d_kinetics
import x3d
import csn
import mvit
import slowfast
import action_slot
import slot_vps
import slot_mo
import slot_savi
import ARG
import ORN
import action_slot_query
from classifier import Head, Allocated_Head

from timm.models import create_model
from collections import OrderedDict
# import videomae.utils as utils

# for registering timm create_model
import modeling_finetune


def generate_model(args, num_ego_class, num_actor_class):
    model_name = args.model_name
    seq_len = args.seq_len
    if model_name == 'vivit':
        model = vivit.ViViT((256, 768), 16, seq_len, num_ego_class, num_actor_class)

    elif model_name == 'i3d':
        model = i3d_kinetics.I3D_KINETICS(num_ego_class, num_actor_class)
        for t in model.model.parameters():
            t.requires_grad=False
        for t in model.model.blocks[-1].parameters():
            t.requires_grad=True
        for t in model.model.blocks[-2].parameters():
            t.requires_grad=True
        for t in model.model.blocks[-3].parameters():
            t.requires_grad=True

    elif model_name == 'x3d':
        model = x3d.X3D(num_ego_class, num_actor_class, args)
        for t in model.model.parameters():
            t.requires_grad=False

        for t in model.model.blocks[-1].parameters():
            t.requires_grad=True
        for t in model.model.blocks[-2].parameters():
            t.requires_grad=True
        for t in model.model.blocks[-3].parameters():
            t.requires_grad=True

    elif model_name == 'csn':
        model = csn.CSN(num_ego_class, num_actor_class)
        for t in model.model.parameters():
            t.requires_grad=False
        for t in model.model.blocks[-1].parameters():
            t.requires_grad=True
        for t in model.model.blocks[-2].parameters():
            t.requires_grad=True
        for t in model.model.blocks[-3].parameters():
            t.requires_grad=True

    elif model_name == 'slowfast':
        model = slowfast.SlowFast(num_ego_class, num_actor_class)
        for t in model.parameters():
            t.requires_grad=False
        for t in model.model.blocks[-1].parameters():
            t.requires_grad=True
        for t in model.model.blocks[-2].parameters():
            t.requires_grad=True
        for t in model.model.blocks[-3].parameters():
            t.requires_grad=True

    elif model_name == 'mvit':
        model = mvit.MViT(num_ego_class, num_actor_class,args.scale)
        tune_block_idx = args.tune_block_idx
        print(tune_block_idx)
        for t in model.parameters():
            t.requires_grad=False
        for t in model.head.parameters():
            t.requires_grad=True
            t = nn.init.trunc_normal_(t)
        for t in model.model.cls_positional_encoding.parameters():
            t.requires_grad=True
        for t in model.model.patch_embed.parameters():
            t.requires_grad=True
        for t in model.model.norm_embed.parameters():
            t.requires_grad=True
        # for t in model.custom_posembed.parameters():
        # 	t.requires_grad=True
        for idx in tune_block_idx:
            for t in model.model.blocks[idx].parameters():
                t.requires_grad=True

    elif model_name == 'action_slot': 
        model = action_slot.ACTION_SLOT(args, num_ego_class, num_actor_class, args.num_slots, box=args.box)
        for t in model.parameters():
            t.requires_grad=True

        if args.backbone == 'r50':
            for t in model.resnet.parameters():
                t.requires_grad=False
            for t in model.resnet.blocks[-1].parameters():
                t.requires_grad=True
        else:
            for t in model.resnet.parameters():
                t.requires_grad=False
            for t in model.resnet[-1].parameters():
                t.requires_grad=True
            for t in model.resnet[-2].parameters():
                t.requires_grad=True

    elif model_name == 'action_slot_query': 
        model = action_slot_query.ACTION_SLOT_QUERY(args, num_ego_class, num_actor_class, args.num_slots, box=args.box)
        for t in model.parameters():
            t.requires_grad=True
        for t in model.resnet.parameters():
            t.requires_grad=False
        for t in model.resnet[-1].parameters():
            t.requires_grad=True
        for t in model.resnet[-2].parameters():
            t.requires_grad=True

    elif model_name == 'slot_vps':
        model = slot_vps.SLOT_VPS(args, num_ego_class, num_actor_class, args.num_slots)
        for t in model.parameters():
            t.requires_grad=True

        for t in model.resnet.parameters():
            t.requires_grad=False
        for t in model.resnet[-1].parameters():
            t.requires_grad=True
        for t in model.resnet[-2].parameters():
            t.requires_grad=True

    elif model_name == 'slot_mo':
        model = slot_mo.SLOT_MO(args, num_ego_class, num_actor_class, args.num_slots)
        for t in model.parameters():
            t.requires_grad=True
        for t in model.resnet.parameters():
            t.requires_grad=False
        for t in model.resnet[-1].parameters():
            t.requires_grad=True
        for t in model.resnet[-2].parameters():
            t.requires_grad=True

    elif model_name == 'slot_savi':
        model = slot_savi.SLOT_SAVI(args, num_ego_class, num_actor_class, args.num_slots)
        for t in model.parameters():
            t.requires_grad=True
        for t in model.resnet.parameters():
            t.requires_grad=False
        for t in model.resnet[-1].parameters():
            t.requires_grad=True
        for t in model.resnet[-2].parameters():
            t.requires_grad=True

    elif model_name == 'ARG' or model_name == 'ORN':
        if model_name == 'ARG':
            model = ARG.ARG(args, max_N=63, num_ego_class=num_ego_class, num_actor_class=num_actor_class)
        else:
            model = ORN.ORN(args, max_N=63, num_ego_class=num_ego_class, num_actor_class=num_actor_class)
        for t in model.parameters():
            t.requires_grad=True
        for t in model.resnet.parameters():
            t.requires_grad=False
        for t in model.resnet[-1].parameters():
            t.requires_grad=True
        for t in model.resnet[-2].parameters():
            t.requires_grad=True

    elif model_name == 'videoMAE':
        tune_block_idx = args.tune_block_idx
        # print(tune_block_idx)
        model = video_mae(args.scale)
        model = mvit.VideoMAE_pretrained(model,num_ego_class, num_actor_class)
        # for t in model.named_parameters():
        #     print(t[0])
        # raise BaseException
        for t in model.parameters():
            t.requires_grad=False
        for t in model.head.parameters():
            t.requires_grad=True
            t = nn.init.trunc_normal_(t)
        for t in model.model.patch_embed.parameters():
            t.requires_grad=True
        for t in model.model.fc_norm.parameters():
            t.requires_grad=True
        for idx in tune_block_idx:
            for t in model.model.blocks[idx].parameters():
                t.requires_grad=True
    
    params = sum([np.prod(p.size()) for p in model.parameters()])
    print ('Total parameters: ', params)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print ('Total trainable parameters: ', params)
    return model

def video_mae(scale):
    model_name = "vit_base_patch16_224" # vit_large_patch16_224
    if scale == -1.0:
        input_size = (224,224)
    else:
        input_size = (512//scale,1536//scale)
    window_size, num_frames = (8, 14, 14), 16
    finetune = "./videomae/checkpoint_B.pth"
    model_key = "model|module"
    model = create_model( 
        model_name,
        pretrained=False,
        num_classes=400,
        all_frames=16,
        tubelet_size=2,
        fc_drop_rate=0.5,
        drop_rate=0.0,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        drop_block_rate=None,
        use_checkpoint=False,
        use_mean_pooling=True,
        init_scale=0.001,
    )
    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    window_size = (num_frames // 2, input_size[0] // patch_size[0], input_size[1] // patch_size[1])
        # args.patch_size = patch_size

    if finetune:
        checkpoint = torch.load(finetune, map_location='cpu')
        print("Load ckpt from %s" % finetune)
        checkpoint_model = None
        for model_key in model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        new_dict = OrderedDict()
        for key in all_keys:
            if key.startswith('backbone.'):
                new_dict[key[9:]] = checkpoint_model[key]
            elif key.startswith('encoder.'):
                new_dict[key[8:]] = checkpoint_model[key]
            elif key.startswith('head.'):
                continue
            else:
                new_dict[key] = checkpoint_model[key]
        checkpoint_model = new_dict
        # interpolate position embedding
        if 'pos_embed' in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1] # channel dim
            num_patches = model.patch_embed.num_patches # 
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches # 0/1

            # height (== width) for the checkpoint position embedding 
            orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens)//(num_frames // model.patch_embed.tubelet_size)) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int((num_patches // (num_frames // model.patch_embed.tubelet_size) )** 0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                # B, L, C -> BT, H, W, C -> BT, C, H, W
                pos_tokens = pos_tokens.reshape(-1, num_frames // model.patch_embed.tubelet_size, orig_size, orig_size, embedding_size)
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, num_frames // model.patch_embed.tubelet_size, new_size, new_size, embedding_size) 
                pos_tokens = pos_tokens.flatten(1, 3) # B, L, C
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['pos_embed'] = new_pos_embed

        utils.load_state_dict(model, checkpoint_model, prefix='')
        return model
