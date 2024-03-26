import argparse
import os

def get_parser():
    parser = argparse.ArgumentParser()

    #dataset
    parser.add_argument('--dataset', type=str, default='taco', choices=['taco', 'oats', 'nuscenes'])
    parser.add_argument('--oats_test_split', type=str, default='0', choices=['s1', 's2', 's3'])
    parser.add_argument('--root', type=str, help='dataset path')

    
    # model
    parser.add_argument('--model_name', type=str, help='Unique experiment identifier.')
    parser.add_argument('--backbone', type=str, default="x3d")
    parser.add_argument('--num_slots', type=int, default=64, help='')
    parser.add_argument('--seq_len', type=int, default=16, help='')
    parser.add_argument('--allocated_slot', help="", action="store_true")
    parser.add_argument('--channel', type=int, default=256, help='')
    parser.add_argument('--box', help="", action="store_true")

    
    # attention
    parser.add_argument('--bg_slot', help="", action="store_true")
    parser.add_argument('--action_attn_weight', type=float, default=0., help='')
    parser.add_argument('--bg_attn_weight', type=float, default=0., help='')
    parser.add_argument('--bg_mask', help="", action="store_true")
    parser.add_argument('--mask_every_frame', type=int, default=4, help='')
    parser.add_argument('--bg_upsample', type=int, default=4, help='')
    parser.add_argument('--obj_mask', help="", action="store_true")
    parser.add_argument('--flow', help="", action="store_true")

    # action loss
    parser.add_argument('--bce_pos_weight', type=float, default=10, help='')
    parser.add_argument('--ce_pos_weight', type=float, default=1, help='')
    parser.add_argument('--ce_neg_weight', type=float, default=0.05, help='')
    parser.add_argument('--ego_loss_weight', type=float, default=0.03, help='')

    
    # training
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--pretrain', type=str, default='', choices=['taco', 'oats'])
    parser.add_argument('--epochs', type=int, default=100, help='Number of train epochs.')
    parser.add_argument('--wd', type=float, default=0.07, help='')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate.')
    parser.add_argument('--scheduler', help="", action="store_true")
    
    parser.add_argument('--val_every', type=int, default=10, help='Validation frequency (epochs).')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of train epochs.')
    parser.add_argument('--parallel', help="", action="store_true")
    parser.add_argument('--tune_block_idx', type=int, default=[0,1,2,-3,-2,-1],nargs='+')
    

    # eval
    parser.add_argument('--model_index', type=int, default=-1)
    parser.add_argument('--cp', type=str, default='best_model.pth')
    parser.add_argument('--plot', help="", action="store_true")
    parser.add_argument('--plot_threshold', type=float, default=0, help='')
    parser.add_argument('--plot_mode', type=str, default='both')
    parser.add_argument('--val_confusion', help="", action="store_true")
    parser.add_argument('--ego_motion', type=int, default=-1)
    parser.add_argument('--scale', type=float, default=-1.0)
    
    # others
    parser.add_argument('--test', help="", action="store_true")
    parser.add_argument('--gt', help="", action="store_true")


    args = parser.parse_args()

    if not args.bg_mask:
        args.bg_attn_weight = 0.

    if args.pretrain == '':
        if args.dataset == 'oats' and args.oats_test_split != '0':
            based_log = args.dataset + '_' + args.oats_test_split + '_log'
        else:
            based_log = args.dataset + '_log'
    else:
        if args.dataset == 'oats':
            if args.oats_test_split == '0':
                return
            else:
                based_log = args.dataset + '_' + '_pretrained_' + args.pretrain 
                + args.oats_test_split + '_log'
        else:
            based_log = args.dataset + '_pretrained_' + args.pretrain 
    if not os.path.isdir(based_log):
        os.makedirs(based_log)
    based_log = os.path.join(based_log, args.model_name)
    if not os.path.isdir(based_log):
        os.makedirs(based_log)

    if args.model_name in ['action_slot', 'slot_savi', 'slot_mo', 'slot_vps', 'action_slot_query']:
        logdir = os.path.join(
            based_log,
            'num_slots: ' + str(args.num_slots) + '\n'
            + 'allocated :' + str(args.allocated_slot) + '\n'
            + args.backbone + '\n'
            + 'channel :' + str(args.channel) + '\n'
            +'bg_slot: ' + str(args.bg_slot) + '\n'
            +'bg_mask: ' + str(args.bg_mask) + '\n'
            +'action_attn_w: ' + str(args.action_attn_weight) + '\n'
            +'bg_attn_w: ' + str(args.bg_attn_weight) + '\n'
            +'obj_mask: ' + str(args.obj_mask) + '\n'
            +'epoch: ' + str(args.epochs) + '\n'
            +'lr: ' + str(args.lr) + '\n'
            +'wd: '+ str(args.wd) + '\n'
            +'bce_pos_weight: ' + str(args.bce_pos_weight) + '\n'
            +'bg_upsample: ' + str(args.bg_upsample) + '\n'
            +'ego_loss_weight: ' + str(args.ego_loss_weight)
            )

    elif args.model_name in ['action-slot', 'slot_savi', 'slot_mo', 'slot_vps'] and not args.allocated_slot:
        logdir = os.path.join(
            based_log,
            'num_slots: ' + str(args.num_slots) + '\n'
            + 'allocated_slot: ' + str(args.allocated_slot) + '\n'
            + args.backbone + '\n'
            +'obj_mask: ' + str(args.obj_mask) + '\n'
            +'epoch: ' + str(args.epochs) + '\n'
            +'lr: ' + str(args.lr) + '\n'
            +'wd: '+ str(args.wd) + '\n'
            +'ce_pos_weight: ' + str(args.ce_pos_weight) + '\n'
            +'ce_neg_weight: ' + str(args.ce_neg_weight)
            )
    elif args.model_name in ['i3d', 'x3d', 'csn', 'slowfast']:
        logdir = os.path.join(
            based_log,
            'channel :' + str(args.channel) + '\n'
            +'epoch: ' + str(args.epochs) + '\n'
            +'lr: ' + str(args.lr) + '\n'
            +'wd: '+ str(args.wd) + '\n'
            +'bce_pos_weight: ' + str(args.bce_pos_weight) + '\n'
            +'ego_loss_weight: ' + str(args.ego_loss_weight)
            )
    elif args.model_name in ['mvit', 'videoMAE']:
        logdir = os.path.join(
            based_log,
            'tune_block_idx: ' + str(args.tune_block_idx) + '\n'
            + 'channel :' + str(args.channel) + '\n'
            +'epoch: ' + str(args.epochs) + '\n'
            +'lr: ' + str(args.lr) + '\n'
            +'wd: '+ str(args.wd) + '\n'
            +'bce_pos_weight: ' + str(args.bce_pos_weight) + '\n'
            +'ego_loss_weight: ' + str(args.ego_loss_weight)
            )
    elif args.box:
        # object-based
        logdir = os.path.join( 
            based_log,
            args.backbone + '\n'
            +'gt_box: ' + str(args.gt) + '\n'
            + 'channel :' + str(args.channel) + '\n'
            +'epoch: ' + str(args.epochs) + '\n'
            +'lr: ' + str(args.lr) + '\n'
            +'wd:'+ str(args.wd) + '\n'
            +'ce_pos_weight: ' + str(args.ce_pos_weight) + '\n'
            +'ce_neg_weight: ' + str(args.ce_neg_weight) + '\n'
            +'ego_loss_weight: ' + str(args.ego_loss_weight)
            )

    if args.model_index != -1:
        logdir = logdir + '\n' + 'idx: ' + str(args.model_index)
    return args, logdir
