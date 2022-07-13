#!/usr/bin/env python
# coding: utf-8



import os
import Hashing_KNN as HK


os.chdir('C:\\Master_Code_92589')



import pandas as pd
from collections import Counter
import statistics
import random
import csv
import os
import numpy as np
import pickle



position_list = [[1,1],[1,2],[1,3],[1,4],[1,5],[1,6],[1,7],[1,8],[1,9],[1,10],
                    [6,1],[6,2],[6,3],[6,4],[6,5],[6,6],[6,7],[6,8],[6,9],[6,10],
                    [11,1],[11,2],[11,3],[11,4],[11,5],[11,6],[11,7],[11,8],[11,9],[11,10],
                    [11,11],[10,11],[9,11],[8,11],[7,11],[6,11],[5,11],[4,11],[3,11],[2,11],[1,11]]

table_class = np.array([\
                        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                        [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,31],\
                        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,32],\
                        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,33],\
                        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,34],\
                        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,35],\
                        [ 0,11,12,13,14,15,16,17,18,19,20,36],\
                        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,37],\
                        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,38],\
                        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,39],\
                        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,40],\
                        [ 0,21,22,23,24,25,26,27,28,29,30,41],\
                        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

def class_to_coordinate(a):
    table = np.array([\
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
        [ 0, 1, 0, 0, 0, 0,11, 0, 0, 0, 0,21],\
        [ 0, 2, 0, 0, 0, 0,12, 0, 0, 0, 0,22],\
        [ 0, 3, 0, 0, 0, 0,13, 0, 0, 0, 0,23],\
        [ 0, 4, 0, 0, 0, 0,14, 0, 0, 0, 0,24],\
        [ 0, 5, 0, 0, 0, 0,15, 0, 0, 0, 0,25],\
        [ 0, 6, 0, 0, 0, 0,16, 0, 0, 0, 0,26],\
        [ 0, 7, 0, 0, 0, 0,17, 0, 0, 0, 0,27],\
        [ 0, 8, 0, 0, 0, 0,18, 0, 0, 0, 0,28],\
        [ 0, 9, 0, 0, 0, 0,19, 0, 0, 0, 0,29],\
        [ 0,10, 0, 0, 0, 0,20, 0, 0, 0, 0,30],\
        [ 0,31,32,33,34,35,36,37,38,39,40,41]])
    
    x = np.argwhere(table == a)[0][1]
    y = np.argwhere(table == a)[0][0]
    
    coordinate = [x,y]
    
    return coordinate

def add_weight(particle_weight_map, pos, particle_num, alpha, c, spread = 3):
    
    # add weight to target position, and spread to neighbor
    alpha_1 = alpha*c
    alpha_2 = alpha_1*c
    alpha_3 = alpha_2*c
    
    for direction in range(4):
        particle_weight_map = add_weight_dir(particle_weight_map, pos, direction, alpha * particle_num)
    
    x = position_list[pos][0]
    y = position_list[pos][1]
    
    if spread > 0:
        
        neighbor_1 = [[x+1, y], [x,y+1], [x-1,y], [x,y-1]]
        neighbor_2 = [[x+1,y+1], [x-1,y+1], [x+1,y-1], [x-1,y-1], [x+2,y], [x,y+2], [x-2,y], [x,y-2]]
        neighbor_3 = [[x+1,y+2], [x+2,y+1], [x+2,y-1], [x+1,y-2], [x-1,y-2], [x-2,y-1], [x-2,y+1], [x-1,y+2],\
                      [x+3, y], [x,y+3], [x-3,y], [x,y-3]]
        
        neighbor_list = [neighbor_1, neighbor_2, neighbor_3]
        parameter_list = [alpha_1, alpha_2, alpha_3]
                
        for i in range(spread): 
            for neighbor_pos in neighbor_list[i]:
                if neighbor_pos in position_list:
                    np = position_list.index(neighbor_pos)
                    for direction in range(4):
                        particle_weight_map = add_weight_dir(particle_weight_map, np, direction, parameter_list[i] * particle_num)
                    
    return particle_weight_map

def add_weight_dir(particle_weight_map, pos, direction, particle_num):
    coor = position_list[pos]
    particle_weight_map[coor[0]][coor[1]][direction] += particle_num
    return particle_weight_map

def proportion_init(particle_weight_map, p_count_knn):
    
    # spread particles
    w_sum = 0
    for i in range(int(p_count_knn/4)):
        for d in range(4):
            pos = position_list[ random.randint(0,33) ]
            particle_weight_map[pos[0]][pos[1]][d] += (1/p_count_knn)
            w_sum += (1/p_count_knn)
    
    return particle_weight_map

def euclidean_distance(p1,p2):
    return np.sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1]))

def avg_normalization(particle_weight_map_i):
    w_sum = 0
    for pos in position_list:
        for d in range(4):
            w_sum += particle_weight_map_i[pos[0]][pos[1]][d]
            
    for pos in position_list:
        for d in range(4):
            particle_weight_map_i[pos[0]][pos[1]][d] /= w_sum
            
    return particle_weight_map_i

def min_max_normalization(particle_weight_map_i):
    Min = 9999999
    Max = 0
    for pos in position_list:
        for d in range(4):
            if particle_weight_map_i[pos[0]][pos[1]][d] > Max:
                Max = particle_weight_map_i[pos[0]][pos[1]][d]
            if particle_weight_map_i[pos[0]][pos[1]][d] < Min:
                Min = particle_weight_map_i[pos[0]][pos[1]][d]
    for pos in position_list:
        for d in range(4):
            particle_weight_map_i[pos[0]][pos[1]][d] = (particle_weight_map_i[pos[0]][pos[1]][d]-Min)/Max-Min
            
    return particle_weight_map_i

def normalization(particle_weight_map_i):
    return avg_normalization(particle_weight_map_i)

def particle_move(particle_weight_map):
    new_map = np.zeros((13,13,4))
    for pos in position_list: # for every particle
        
        if [pos[0] - 1, pos[1]] in position_list:
            new_map[pos[0] - 1][pos[1]][0] += particle_weight_map[pos[0]][pos[1]][0]*1 # 0: x-1
        if [pos[0], pos[1] + 1] in position_list:
            new_map[pos[0]][pos[1] + 1][1] += particle_weight_map[pos[0]][pos[1]][1]*1 # 1: y+1
        if [pos[0] + 1, pos[1]] in position_list:
            new_map[pos[0] + 1][pos[1]][2] += particle_weight_map[pos[0]][pos[1]][2]*1 # 2: x+1
        if [pos[0], pos[1] - 1] in position_list:
            new_map[pos[0]][pos[1] - 1][3] += particle_weight_map[pos[0]][pos[1]][3]*1 # 3: y+1
            
    return new_map



#HK.max(voter, key=voter.count)
#HK.dict
#HK.particle_weight_map









def calculate(knn_candidates,i):
    p_count_knn = 2000



    alpha = 1
    c = 0.5
    particle_prev_count = 20000
    knn_candidates = []
    spread=3

    
#map的部分用np save 和np load 讀寫就好 不用傳入




    





    #for i in range(len(knn_candidates)): # for 每個 KNN 預測的位置

    max_weight = 0
    max_pos = [0, 0]







    if i == 0: # if 第一個位置 (是特例，沒有前一個位置)
        # 將該位置候選人以票數做 weight, 更新 partcle weight
        particle_weight_map = np.zeros((13,13,4))
        particle_weight_map = proportion_init(particle_weight_map, p_count_knn)
        candidate_list = list(knn_candidates.keys()) # 每一輪預測KNN的參考位置 ( all 候選人 )----->[i]放 一個list("1":"5票" "2":6票).....?

        candidate_vote_list = list(knn_candidates.values()) # 每一輪預測 KNN 的參考位置的次數 ( all 候選人票數 )

        all_vote = sum(candidate_vote_list) # 總票數

        # 由得票數設定 weight
        for k in range(len(candidate_vote_list)): # 每一個參考位置的比例分配
            candidate_vote_list[k] /= all_vote

        # 把 weight佈到 map上
        for j, element in enumerate(candidate_list): #每一輪KNN的每一個參考位置
            if class_to_coordinate(element) in position_list:
                position_list_index = position_list.index(class_to_coordinate(element))
                particle_weight_map = add_weight(particle_weight_map, position_list_index, 1*candidate_vote_list[j], alpha, c, spread)

        # 找地圖重心
        weight_sum = 0
        for pos in position_list:
            for d in range(4):
                if max_weight < particle_weight_map[pos[0]][pos[1]][d]:
                    max_weight = particle_weight_map[pos[0]][pos[1]][d]
                    max_pos = pos

        particle_weight_map = normalization(particle_weight_map)
        #想辦法把map存起來
        np.save('particle_weight_map',particle_weight_map)
        return max_pos

        

    else:
        particle_weight_map=np.load('particle_weight_map.npy')
        for pos in position_list:
            for d in range(4):
                particle_weight_map[pos[0]][pos[1]][d] *= particle_prev_count

        candidate_list = list(knn_candidates.keys()) # 每一輪預測KNN的參考位置 ( all 候選人 )

        candidate_vote_list = list(knn_candidates.values()) # 每一輪預測 KNN 的參考位置的次數 ( all 候選人票數 )

        all_vote = sum(candidate_vote_list) # 總票數

        # 由得票數設定 weight
        for k in range(len(candidate_vote_list)): # 每一個參考位置的比例分配
            candidate_vote_list[k] /= all_vote

        for j,element in enumerate(candidate_list): #每一輪 KNN 的每一個參考位置
            if class_to_coordinate(element) in position_list:
                position_list_index = position_list.index(class_to_coordinate(element))
                particle_weight_map = add_weight(particle_weight_map, position_list_index, \
                                                    p_count_knn*candidate_vote_list[j], alpha, c, spread)

        # 找地圖重心
        weight_sum = 0
        for pos in position_list:
            for d in range(4):
                if max_weight < particle_weight_map[pos[0]][pos[1]][d]:
                    max_weight = particle_weight_map[pos[0]][pos[1]][d]
                    max_pos = pos

        particle_weight_map = normalization(particle_weight_map)
        #想辦法把map存起來
        np.save('particle_weight_map',particle_weight_map)
        return max_pos


#給 max_pos 和 particle_weight_map回去

