# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 19:09:28 2019

@author: bian
"""
import os,shutil
small_pads=os.listdir('./imgsave/0small_pad')
for pad in small_pads:
    path00='./imgsave/0small_pad/'+pad
    path01='./imgsave/pure/0small_pad/'+pad
    
    
    path10='./imgsave/1origin/'+pad
    path11='./imgsave/pure/1origin/'+pad
    
    path20='./imgsave/2target/'+pad
    path21='./imgsave/pure/2target/'+pad
    
    path30='./imgsave/3human/'+pad
    path31='./imgsave/pure/3human/'+pad
    
    path40='./imgsave/4human_target/'+pad
    path41='./imgsave/pure/4human_target/'+pad

    shutil.copyfile(path00, path01)
    shutil.copyfile(path10, path11)
    shutil.copyfile(path20, path21)
    shutil.copyfile(path30, path31)
    shutil.copyfile(path40, path41)
    
    print(pad + '    copied...')