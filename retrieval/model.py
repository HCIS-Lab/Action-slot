import torch
from torch import nn
import numpy as np

import sys
sys.path.append('/media/hankung/ssd/retrieval/models')
sys.path.append('/work/u8526971/retrieval/models')
sys.path.append('/home/hcis-s19/Desktop/retrieval/models')
import vivit
# import cnn_gru
# import cnn_convgru
import res3d
import i3d
import i3d_kinetics
import x3d
import csn
import mvit
import slowfast
import slot_video
import slot_vps
import slot_seg
import slot_mo
import slot_savi
import ARG
import ORN

import segment
from retrieval_head import Head, Instance_Head

from timm.models import create_model
from collections import OrderedDict
import videomae.utils as utils
import modeling_finetune

def load_pretrained_model(model, num_ego_class, num_actor_class, \
    pretrain_path='/media/hankung/ssd/retrieval/models/r3d50_K_200ep.pth'):
    if pretrain_path:
        print('loading pretrained model {}'.format(pretrain_path))
        pretrain = torch.load(pretrain_path, map_location='cpu')

        model.load_state_dict(pretrain['state_dict'], strict=False)
        tmp_model = model

        tmp_model.head = Head(tmp_model.head_in_c,
                                     num_ego_class, num_actor_class)

    return model

def generate_model(args, model_name, num_ego_class, num_actor_class, seq_len):

    # if model_name == 'cnngru':
    # 	model = cnn_gru.CNNGRU(num_ego_class, num_actor_class)
    # 	for t in model.parameters():
    # 		t.requires_grad=True

    # if model_name == 'cnn_convgru':
    # 	model = cnn_convgru.CNN_CONVGRU(num_ego_class, num_actor_class)
    # 	for t in model.parameters():
    # 		t.requires_grad=True

    # elif model_name == 'cnnfc':
    # 	model = cnnfc.CNNFC(num_ego_class, num_actor_class, seq_len)
    # 	if use_backbone:
    # 		for t in model.backbone.parameters():
    # 	  		t.requires_grad=False

    if model_name == 'vivit':
        model = vivit.ViViT((256, 768), 16, seq_len, num_ego_class, num_actor_class)
    elif model_name == '3dres':
        model = res3d.ResNet3D(num_ego_class=num_ego_class, num_actor_class=num_actor_class)
        model = load_pretrained_model(model, num_ego_class, num_actor_class)
        for t in model.parameters():
              t.requires_grad=False
        for t in model.head.parameters():
            t.requires_grad=True
        for t in model.layer4.parameters():
            t.requires_grad=True
    elif model_name =='i3d':
        model = i3d.InceptionI3d(num_ego_class, num_actor_class, in_channels=3)
        model.load_state_dict(torch.load('/media/hankung/ssd/retrieval/models/rgb_charades.pt'), strict=False)
        model.replace_logits()
        for t in model.parameters():
              t.requires_grad=False
        for t in model.end_points['Mixed_3b'].parameters():
            t.requires_grad=True
        for t in model.end_points['Mixed_3c'].parameters():
            t.requires_grad=True
        for t in model.end_points['Mixed_4b'].parameters():
            t.requires_grad=True
        for t in model.end_points['Mixed_4c'].parameters():
            t.requires_grad=True
        for t in model.end_points['Mixed_4d'].parameters():
            t.requires_grad=True
        for t in model.end_points['Mixed_4e'].parameters():
            t.requires_grad=True
        for t in model.end_points['Mixed_4f'].parameters():
            t.requires_grad=True
        for t in model.end_points['Mixed_5b'].parameters():
            t.requires_grad=True
        for t in model.end_points['Mixed_5b'].parameters():
            t.requires_grad=True
        for t in model.end_points['Mixed_5c'].parameters():
            t.requires_grad=True
        for t in model.logits.parameters():
            t.requires_grad=True
    elif model_name == 'i3d_kinetics':
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
        model = x3d.X3D(num_ego_class, num_actor_class)
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

    elif model_name == 'segment':
        model = segment.SEGMENT(args, num_ego_class, num_actor_class)
        for t in model.parameters():
            t.requires_grad=True
        for t in model.resnet.parameters():
            t.requires_grad=False
        for t in model.resnet[-1].parameters():
            t.requires_grad=True

    elif model_name == 'slot':
        model = slot_video.SLOT_VIDEO(args, num_ego_class, num_actor_class, args.num_slots, box=args.box)
        for t in model.parameters():
            t.requires_grad=True
        for t in model.resnet.parameters():
            t.requires_grad=False
        for t in model.resnet[-1].parameters():
            t.requires_grad=True
        for t in model.resnet[-2].parameters():
            t.requires_grad=True

    elif model_name == 'slot_seg':	
        model = slot_seg.SLOT_SEG(args, num_ego_class, num_actor_class, args.num_slots, box=args.box)
        for t in model.parameters():
            t.requires_grad=True
        for t in model.resnet.parameters():
            t.requires_grad=False
        for t in model.resnet[-1].parameters():
            t.requires_grad=True
        for t in model.resnet[-2].parameters():
            t.requires_grad=True

    elif model_name == 'slot_video_fc':
        model = slot_video_fc.SLOT_VIDEO_FC(args, num_ego_class, num_actor_class, args.num_slots, box=args.box)
        for t in model.parameters():
            t.requires_grad=True
        for t in model.resnet.parameters():
            t.requires_grad=False
        for t in model.resnet[-1].parameters():
            t.requires_grad=True

    elif model_name == 'slot_collapse':
        model = slot_collapse_time.SLOT_COLLAPSE(args, num_ego_class, num_actor_class, args.num_slots)
        for t in model.parameters():
            t.requires_grad=True
        for t in model.resnet.parameters():
            t.requires_grad=False
        for t in model.resnet[-1].parameters():
            t.requires_grad=True

    elif model_name == 'slot_3d':
        model = slot_3d.SLOT_3D(args, num_ego_class, num_actor_class, args.num_slots)
        for t in model.parameters():
            t.requires_grad=True
        for t in model.resnet.parameters():
            t.requires_grad=False
        for t in model.resnet[-1].parameters():
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
        print(tune_block_idx)
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
    finetune = "/home/hcis-s19/Desktop/VideoMAE/checkpoint_B.pth"
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
    
# @register_model
# def vit_large_patch16_224(pretrained=False, **kwargs):
#     model = VisionTransformer(
#         patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     return model
