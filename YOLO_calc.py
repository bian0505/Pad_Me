i# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 23:38:36 2019

@author: bian
"""
import torch,os,cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
from modelclass import Region_Mask

human_path="D://Users/bian/Desktop/testimage/1273.jpg"

DEVICE = torch.device("cpu")

model_cpu = torch.load("model_Large561.bian")
model = model_cpu.to(DEVICE)
model.eval()
human_img = mpimg.imread(human_path)
img_shape_x=int(human_img.shape[1]*200/human_img.shape[0])
img_human_test = cv2.resize(human_img, (img_shape_x, 200))
img_zeros=np.zeros([1,250,250,3])
result_zeros=np.zeros([250,250])
img_zeros[0,0:200,0:img_shape_x,:]=img_human_test/255


output=model.forward(torch.tensor(img_zeros).permute(0,3,1,2).float().to(DEVICE))
out=output.cpu().detach().numpy()
#out[out<0]=0
out_3d=out[0,:,:,:]
#out_3d_sum=out_3d[:,:,0]+out_3d[:,:,1]+out_3d[:,:,2]+out_3d[:,:,3]+out_3d[:,:,4]
#out_3d_sum[out_3d_sum<0]=0
#out_3d_250=cv2.resize(out_3d_sum, (250, 250))
#
#plt.figure()
#plt.subplot(1,2,1)
#plt.imshow(img_zeros[0,:,:,:])
#plt.subplot(1,2,2)
#plt.imshow(out_3d_250)
#plt.show()
index=np.where(out_3d==np.max(out_3d))
#index=out_3d.argmax(out_3d)
x_loc=index[0][0]*10
x_end=index[0][0]*10+(index[2][0]+7)*3
y_loc=index[1][0]*10
y_end=index[1][0]*10+(index[2][0]+7)*4

img=img_zeros[0,x_loc:x_end,y_loc:y_end,:]
imgx=cv2.rectangle(img_zeros[0,:,:,:], (y_loc, x_loc), (y_end, x_end), (0, 0, 255), 2)
plt.figure()
plt.imshow(imgx)