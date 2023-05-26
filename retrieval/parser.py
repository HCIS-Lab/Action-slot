import argparse
import os

def get_parser():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--id', type=str, default='cnnlstm_imagenet', help='Unique experiment identifier.')
    parser.add_argument('--backbone', type=str, help="x3d-2")
    parser.add_argument('--num_slots', type=int, default=20, help='')
    parser.add_argument('--seq_len', type=int, default=16, help='')
    parser.add_argument('--fix_slot', help="", action="store_true")
    parser.add_argument('--channel', type=int, default=128, help='')
    parser.add_argument('--box', help="", action="store_true")

    parser.add_argument('--cp', type=str, default='best_model.pth')

    # attention
    parser.add_argument('--seg', help="", action="store_true")
    parser.add_argument('--action_attn_weight', type=float, default=0., help='')
    parser.add_argument('--bg_attn_weight', type=float, default=0., help='')
    parser.add_argument('--mask', help="", action="store_true")
    parser.add_argument('--upsample', type=int, default=1, help='')
    parser.add_argument('--obj_mask', help="", action="store_true")

    # action loss
    parser.add_argument('--bce', type=float, default=1, help='')
    parser.add_argument('--weight', type=float, default=5, help='')
    parser.add_argument('--ce_weight', type=float, default=1, help='')
    parser.add_argument('--empty_weight', type=float, default=0.05, help='')
    
    
    # training
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--epochs', type=int, default=100, help='Number of train epochs.')
    parser.add_argument('--ego_weight', type=float, default=0.05, help='')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate.')
    parser.add_argument('--val_every', type=int, default=10, help='Validation frequency (epochs).')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of train epochs.')
    parser.add_argument('--parallel', help="", action="store_true")
    parser.add_argument('--model_index', type=int, default=-1)
    parser.add_argument('--tune_block_idx', type=int, default=[0,1,2,-3,-2,-1],nargs='+')
    parser.add_argument('--scheduler', help="", action="store_true")

    # eval
    parser.add_argument('--plot', help="", action="store_true")
    parser.add_argument('--plot_threshold', type=float, default=0, help='')
    parser.add_argument('--plot_mode', type=str, default='both')
    parser.add_argument('--val_confusion', help="", action="store_true")
    parser.add_argument('--ego_motion', type=int, default=-1)
    # others
    parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')
    parser.add_argument('--test', help="", action="store_true")
    parser.add_argument('--scale', type=float, default=2, help='')
    parser.add_argument('--gt', help="", action="store_true")

    args = parser.parse_args()

    # if args.bg_attn_weight > 0.:
    #     args.upsample = 4

    if 'slot' in args.id:
        args.logdir = os.path.join(args.logdir, 
            args.id+'_'+args.backbone+'_wd:'+str(args.weight_decay)+
            '_atw:'+str(args.action_attn_weight) + '_btw'+str(args.bg_attn_weight)+
            '_epoch:'+str(args.epochs)+'_lr:'+str(args.lr)+
            '_mask:'+str(args.mask) + '_bgseg'+str(args.seg)+
            '_upsample:'+str(args.upsample))
    else:
        args.logdir = os.path.join(args.logdir, 
            args.id+
            '_wd:'+str(args.weight_decay)+
            '_epoch:'+str(args.epochs))
    if args.num_slots != 20 and 'slot' in args.id:
        args.logdir = args.logdir + 'num_slots:'+str(args.num_slots)
    if args.channel != 128:
        args.logdir = args.logdir + '_channel:'+str(args.channel)
    if 'slot' in args.id and not args.fix_slot:
        args.logdir = args.logdir +'_no_fix'
    if args.obj_mask:
        args.logdir = args.logdir +'_obj_mask'
        
    if args.gt:
        args.logdir = args.logdir + '_gt'
    if args.scale != 2:
        args.logdir = args.logdir + '_scale:' +str(args.scale)
    if args.model_index != -1:
        args.logdir = args.logdir + '_'+str(args.model_index)
    print(f'Model path: {args.logdir}')
    return args
