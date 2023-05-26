import os
from unittest import skip
import numpy as np
import matplotlib.pyplot as plt 

def data_analysis(gt, 
                    ego_4, a_4, 
                    ego_3_1, a_3_1,
                    ego_3_2, a_3_2,
                    ego_3_3, a_3_3,
                    ego_s, a_s,
                    trainig=True):
    
    
    for k, v in gt.iter():

        map = k.split('_')[0]
        if trainig and map == '10':
            continue
        if not training and map != '10':
            continue

        road_type = k.split('_')[1][:2]
        ego_class, actor_class = gt_interactive.split(',')[0], gt_interactive.split(',')[1]
        if 

            
    
    y=np.arange(len(actions))
    plt.figure(figsize=(12,8))
    #plt.ylim(0, max(num_of_actions) +1)
    #plt.ylim(-10,20)
    plt.bar(y, num_of_actions, tick_label=actions)
    #plt.ylim(-10,20)
    for index,data in enumerate(num_of_actions):
        plt.text(x=index-0.05 , y =data+0.2 , s=f"{data}" , fontdict=dict(fontsize=10))
    plt.title('Interactive actor types ('+ data_type + " : " + str(total_count_actor) +  ')')
    #plt.xlabel('Actors types')
    plt.ylabel('Instances')
    #plt.set_ylim([0, max(y)])
    #plt.ylim(0, max(num_of_actions) +1)
    plt.savefig("./interactive/"+ data_type +"/Actor_types.png", facecolor='white')

    y=np.arange(len(actions))
    plt.figure(figsize=(16,8))
    plt.bar(y, num_of_vio_actions, color = "darkorange")
    plt.bar(y, num_of_actions_vio,bottom=num_of_vio_actions, tick_label=actions, color = "royalblue")
    #plt.bar(y, num_of_actions, tick_label=actions)

    plt.xticks(rotation=50)
    for index,data in enumerate(num_of_action):
        if data != 0:
            plt.text(x=index-0.1 , y =data+0.05 , s=f"{data}" , fontdict=dict(fontsize=10))

    for index,data in enumerate(num_of_vio_actions):
        if data != 0:
            plt.text(x=index-0.1 , y =data+0.05 , s=f"{data}" , fontdict=dict(fontsize=10))
    plt.title('Action type combinations ('+ data_type + " : " + str(total_count_action) + ')')
    plt.ylim(0,max(num_of_action)+1)
    plt.xlabel('Action type')
    plt.ylabel('Instances')
    plt.legend(['Illegal event','Normal event'],loc=2)
    plt.savefig("./interactive/"+ data_type +"/Action_type_combination.png",  facecolor='white')  

    
if __name__ == "__main__":

    ego_4way = {'z1-z1': 0, 'z1-z2': 0, 'z1-z3':0, 'z1-z4': 0}
    4way_label = {'z1-z1': 0, 'z1-z2': 0, 'z1-z3':0, 'z1-z4':0,
                    'z2-z1': 0, 'z2-z2': 0, 'z2-z3': 0, 'z2-z4': 0,
                    'z3-z1': 0, 'z3-z2': 0, 'z3-z3': 0, 'z3-z4': 0,
                    'z4-z1': 0, 'z4-z2': 0, 'z4-z3': 0, 'z4-z4': 0,
                    'c1-c2': 0, 'c1-c4': 0,
                    'c2-c1': 0, 'c2-c3': 0,
                    'c3-c2': 0, 'c3-c4': 0,
                    'c4-c1': 0, 'c4-c3': 0,
                    '': 0}

    ego_3way_1 = {'t1-t1': 0, 't1-t2': 0, 't1-t4': 0}
    3way_1_label = {'t1-t1': 0, 't1-t2': 0, 't1-t4': 0, 
                    't2-t1': 0, 't2-t2': 0, 't2-t4': 0,
                    't4-t1': 0, 't4-t2': 0, 't4-t4': 0,
                    'c1-cf': 0, 'c1-c4': 0,
                    'cf-c1': 0, 'cf-c4': 0,
                    'c4-c1': 0, 'c4-cf': 0,
                    '': 0}

    ego_3way_2 = {'t1-t1': 0, 't1-t2': 0, 't1-t3': 0}
    3way_2_label = {'t1-t1': 0, 't1-t2': 0, 't1-t3': 0,
                    't2-t1': 0, 't2-t2': 0, 't2-t3': 0,
                    't3-t1': 0, 't3-t2': 0, 't3-t3': 0,
                    'c1-c2': 0, 'c1-cl': 0,
                    'c2-c1': 0, 'c2-cl': 0,
                    'cl-c1': 0, 'cl-c2': 0,
                    '': 0}

    ego_3way_3 = {'t1-t1': 0, 't1-t3': 0, 't1-t4': 0}
    3way_3_label = {'t1-t1': 0, 't1-t3': 0, 't1-t4': 0,
                    't3-t1': 0, 't3-t3': 0, 't3-t4': 0,
                    't4-t1': 0, 't4-t3': 0, 't4-t4': 0,
                    'c3-c4': 0, 'c3-cr': 0,
                    'c4-c3': 0, 'c4-cr': 0,
                    'cr-c3': 0, 'cr-c4': 0,
                    '': 0}

    ego_straight = {'s-s': 0, 's-sl': 0, 's-sr': 0}
    straight_label = {'s-s': 0, 's-sl': 0, 's-sr': 0,
                        'sl-s': 0,
                        'sr-s': 0,
                        'jl-s': 0, 'jl-sl': 0, 'jl-jr': 0,
                        'jr-s': 0, 'jr-sr': 0, 'jr-jl': 0,
                        '': 0}
    with open('retrieval_interactive_gt.json') as f:
        gt_interactive = json.load(f)
    with open('retrieval_non-interactive_gt.json') as f:
        gt_non_interactive = json.load(f)
    with open('retrieval_collision_gt.json') as f:
        gt_collision = json.load(f)
    with open('retrieval_obstacle_gt.json') as f:
        gt_obstacle = json.load(f)



    ego_4way, 4way_label, 
    ego_3way_1, 3way_3_label,
    ego_3way_2, 3way_2_label,
    ego_3way_3, 3way_3_label,
    ego_straight, straight_label = data_analysis(gt_interactive, 
                                                ego_4way, 4way_label, 
                                                ego_3way_1, 3way_3_label,
                                                ego_3way_2, 3way_2_label,
                                                ego_3way_3, 3way_3_label,
                                                ego_straight, straight_label, training=True)

    ego_4way, 4way_label, 
    ego_3way_1, 3way_3_label,
    ego_3way_2, 3way_2_label,
    ego_3way_3, 3way_3_label,
    ego_straight, straight_label = data_analysis(gt_non_interactive, 
                                                ego_4way, 4way_label, 
                                                ego_3way_1, 3way_3_label,
                                                ego_3way_2, 3way_2_label,
                                                ego_3way_3, 3way_3_label,
                                                ego_straight, straight_label, training=True)

    ego_4way, 4way_label, 
    ego_3way_1, 3way_3_label,
    ego_3way_2, 3way_2_label,
    ego_3way_3, 3way_3_label,
    ego_straight, straight_label = data_analysis(gt_collision, 
                                                ego_4way, 4way_label, 
                                                ego_3way_1, 3way_3_label,
                                                ego_3way_2, 3way_2_label,
                                                ego_3way_3, 3way_3_label,
                                                ego_straight, straight_label, training=True)

    ego_4way, 4way_label, 
    ego_3way_1, 3way_3_label,
    ego_3way_2, 3way_2_label,
    ego_3way_3, 3way_3_label,
    ego_straight, straight_label = data_analysis(gt_obstacle, 
                                                ego_4way, 4way_label, 
                                                ego_3way_1, 3way_3_label,
                                                ego_3way_2, 3way_2_label,
                                                ego_3way_3, 3way_3_label,
                                                ego_straight, straight_label, training=True)








