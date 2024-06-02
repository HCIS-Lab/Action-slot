import argparse
import os

def get_test_parser():
    parser = argparse.ArgumentParser()

    #dataset
    parser.add_argument('--dataset', type=str, default='taco', choices=['taco'])
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'])
    parser.add_argument('--root', type=str, help='dataset path')
    
    # model
    parser.add_argument('--model_name', type=str, help='Unique experiment identifier.')
    parser.add_argument('--backbone', type=str, help="x3d-2")
    parser.add_argument('--num_slots', type=int, default=64, help='')
    parser.add_argument('--seq_len', type=int, default=16, help='')
    parser.add_argument('--allocated_slot', help="", action="store_true")
    parser.add_argument('--channel', type=int, default=256, help='')
    parser.add_argument('--box', help="", action="store_true")

    # slot attention
    parser.add_argument('--bg_slot', help="", action="store_true")
    parser.add_argument('--bg_mask', help="", action="store_true")
    parser.add_argument('--obj_mask', help="", action="store_true")

    # training
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    parser.add_argument('--num_workers', type=int, default=2, help='Number of train epochs.')    
    
    # eval
    parser.add_argument('--pretrain', type=str, default='', choices=['taco'])
    parser.add_argument('--model_index', type=int, default=-1)
    parser.add_argument('--cp', type=str, default='best_model.pth')
    parser.add_argument('--scale', type=float, default=-1.0)
    
    # others
    parser.add_argument('--gt', help="", action="store_true")


    args = parser.parse_args()

    return args
