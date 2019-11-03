# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 11:10:23 2019

@author: bian
"""
import os,torch,cv2
import matplotlib.image as mpimg 
import numpy as np
import torch.utils.data.dataloader as DataLoader


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
        imgx_trans=imgx.permute(2,0,1)
        imgy=torch.Tensor(self.yimages[index])
        
        #ori_rlt=torch.Tensor(self.ori_result[index])
        return imgx_trans,imgy

    def __len__(self):
        return self.len


    """
    读取图片文件夹中的所有图片
    """    
    def readfile(self):
        path_images='./imgsave/pure/3human'
        path_results='./imgsave/pure/4human_target'
        files=os.listdir(path_images)
        NUM,himgs,rimgs,ori_result=[],[],[],[]
        for file in files:
            human=path_images+'/'+file
            results= path_results+'/'+file            
            human_img = mpimg.imread(human)
            results_img0 = mpimg.imread(results)
            results_img=np.zeros(results_img0.shape)
            results_img[0:-200,:]=results_img0[200:,:]
            #human_img 人原图
            #results_img 框原图
            
            img_shape_x=int(human_img.shape[1]*200/human_img.shape[0])
            img_human_test = cv2.resize(human_img, (img_shape_x, 200))
            img_zeros=np.zeros([250,250,3])
            result_zeros=np.zeros([250,250])
            img_zeros[0:200,0:img_shape_x,:]=img_human_test/255
            
            result_zeros[0:200,0:img_shape_x]=cv2.resize(results_img[:,:,0]/255, (img_shape_x, 200))
            
            confidence=np.zeros([25,25,5])
            
            sum_result=sum(sum(result_zeros))+1
            
            for x_cont0 in range(20):
                for y_cont0 in range(20):
                    x_cont=int(x_cont0*10)
                    y_cont=int(y_cont0*10)
                    confidence[x_cont0,y_cont0,0]=(2*sum(sum(result_zeros[x_cont:x_cont+21,y_cont:y_cont+28]))-sum_result)/sum_result
                    confidence[x_cont0,y_cont0,1]=(2*sum(sum(result_zeros[x_cont:x_cont+24,y_cont:y_cont+32]))-sum_result)/sum_result
                    confidence[x_cont0,y_cont0,2]=(2*sum(sum(result_zeros[x_cont:x_cont+27,y_cont:y_cont+36]))-sum_result)/sum_result
                    confidence[x_cont0,y_cont0,3]=(2*sum(sum(result_zeros[x_cont:x_cont+30,y_cont:y_cont+40]))-sum_result)/sum_result
                    confidence[x_cont0,y_cont0,4]=(2*sum(sum(result_zeros[x_cont:x_cont+33,y_cont:y_cont+44]))-sum_result)/sum_result                    
                    confidence[confidence<0]=0
            
            himgs.append(img_zeros)
            rimgs.append(confidence)
            #ori_result.append(result_zeros)
        NUM.append(himgs)
        NUM.append(rimgs)
        return NUM
        
    

##########################################################
"""
Debug
"""
"""
import matplotlib.pyplot as plt
dataset=DealDataset()
dataloader = DataLoader.DataLoader(dataset,batch_size= 5, shuffle = True, num_workers= 0)
for i, item in enumerate(dataloader):
    print('i:', i)
    data, label = item
    print('data:', data.shape)
    print('label:', label.shape)
"""