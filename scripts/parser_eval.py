import argparse
import os

def get_eval_parser():
    parser = argparse.ArgumentParser()

    #dataset
    parser.add_argument('--dataset', type=str, default='taco', choices=['taco', 'oats', 'nuscenes'])
    parser.add_argument('--oats_test_split', type=str, default='0', choices=['s1', 's2', 's3'])
    parser.add_argument('--root', type=str, help='dataset path')
    parser.add_argument('--nuscenes_test_split', type=str, default='0', choices=['boston', 'singapore'])

    
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
    parser.add_argument('--bg_mask', help="", action="store_true")

    parser.add_argument('--obj_mask', help="", action="store_true")

    # training
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    parser.add_argument('--num_workers', type=int, default=2, help='Number of train epochs.')    
    # eval
    parser.add_argument('--pretrain', type=str, default='', choices=['taco', 'oats'])
    parser.add_argument('--model_index', type=int, default=-1)
    parser.add_argument('--cp', type=str, default='best_model.pth')
    parser.add_argument('--plot', help="", action="store_true")
    parser.add_argument('--plot_threshold', type=float, default=0, help='')
    parser.add_argument('--plot_mode', type=str, default='')
    parser.add_argument('--val_confusion', help="", action="store_true")
    parser.add_argument('--scale', type=float, default=-1.0)
    parser.add_argument('--ego_motion', type=int, default=-1)
    # others
    parser.add_argument('--test', help="", action="store_true")
    parser.add_argument('--gt', help="", action="store_true")
    parser.add_argument('--num_objects', type=int, default=-1)


    args = parser.parse_args()


    if args.dataset == 'oats' and args.oats_test_split != '0':
        based_log = args.dataset + '_' + args.oats_test_split + '_eval'
    else:
        based_log = args.dataset + '_eval'
    if not os.path.isdir(based_log):
        os.makedirs(based_log)
    based_log = os.path.join(based_log, args.model_name)
    if not os.path.isdir(based_log):
        os.makedirs(based_log)

    if args.model_name in ['action_slot', 'slot_savi', 'slot_mo', 'slot_vps']:
        logdir = os.path.join(
            based_log,
            'num_slots: ' + str(args.num_slots) + '\n'
            +'obj_mask: ' + str(args.obj_mask) 
            )
    else:
        logdir = based_log

    if args.model_index != -1:
        logdir = logdir + '\n' + 'idx: ' + str(args.model_index)
    return args, logdir
