# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 11:10:23 2019

@author: bian
"""
import os,torch,cv2
import matplotlib.image as mpimg 
import numpy as np
import torch.utils.data.dataloader as DataLoader
import matplotlib.pyplot as plt

class DealDataset():
    """
        下载数据、初始化数据，都可以在这里完成
    """
    def __init__(self):
        
        self.NUM=self.readfile()
        self.ximages=np.asarray(self.NUM[0])
        self.yimages=np.asarray(self.NUM[1])
        self.len = len(self.ximages)
    
    def __getitem__(self, index):
        
        imgx=torch.Tensor(self.ximages[index])
        imgy=torch.Tensor(self.yimages[index])
        return imgx,imgy

    def __len__(self):
        return self.len


    """
    读取图片文件夹中的所有图片
    """    
    def readfile(self):
        path_images='./imgsave/pure/3human'
        path_results='./imgsave/pure/4human_target'
        files=os.listdir(path_images)
        NUM,himgs,rimgs=[],[],[]
        for file in files:
            human=path_images+'/'+file
            results= path_results+'/'+file
            
            human_img = mpimg.imread(human)
            results_img = mpimg.imread(results)
            
            img_shape_x=int(human_img.shape[1]*200/human_img.shape[0])
            img_human_test = cv2.resize(human_img, (img_shape_x, 200))
            img_zeros=np.zeros([200,150,3])
            #result_zeros=img_zeros([200,150])
            img_zeros[:,0:img_shape_x,:]=img_human_test/255
            
            result_zeros=cv2.resize(results_img[:,:,0], (img_shape_x, 200))
            result_nonzeros=np.asarray(np.nonzero(result_zeros))
            result_nonzeros_x=result_nonzeros[0,:]
            result_nonzeros_y=result_nonzeros[1,:]
            x_begin=np.min(result_nonzeros_x)
            x_end=np.max(result_nonzeros_x)
            y_begin=np.min(result_nonzeros_y)
            y_end=np.max(result_nonzeros_y)
            
            
            himgs.append(img_zeros)
            rimgs.append([x_begin,x_end,y_begin,y_end,1])
        NUM.append(himgs)
        NUM.append(rimgs)
        return NUM
        
    

##########################################################
"""
Debug
"""
dataset=DealDataset()
dataloader = DataLoader.DataLoader(dataset,batch_size= 1, shuffle = True, num_workers= 0)
for i, item in enumerate(dataloader):
    print('i:', i)
    data, label = item
    print('data:', data.shape)
    print('label:', label)

#############################################################