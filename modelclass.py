#https://arxiv.org/pdf/1506.02640.pdf
#import torch
import torch.nn as nn
import torch.nn.functional as F


class Region_Mask(nn.Module):
    def __init__(self):
        super().__init__()
        # input,250*250*3
        self.conv1=nn.Conv2d(3,64,7)    
        self.bt1 = nn.BatchNorm2d(64)
        #-->245*245
        #-->122*122
        self.conv2=nn.Conv2d(64,192,3)   
        self.bt2 = nn.BatchNorm2d(192)
        #-->121,121
        #-->60*60
        self.conv30 = nn.Conv2d(192,128,1)    
        self.conv31 = nn.Conv2d(128,256,3)
        self.conv32 = nn.Conv2d(256,256,1)
        self.conv33 = nn.Conv2d(256,512,3)
        self.bt3 = nn.BatchNorm2d(512)
        #-->28*28
        self.conv400 = nn.Conv2d(512,256,1)    
        self.conv410 = nn.Conv2d(256,512,3)
        self.conv401 = nn.Conv2d(512,256,1)    
        self.conv411 = nn.Conv2d(256,512,3)
        self.conv402 = nn.Conv2d(512,256,1)    
        self.conv412 = nn.Conv2d(256,512,3)
        self.conv403 = nn.Conv2d(512,256,1)    
        self.conv413 = nn.Conv2d(256,512,3)
        self.conv42 = nn.Conv2d(512,512,1)
        self.conv43 = nn.Conv2d(512,1024,3)
        self.bt4 = nn.BatchNorm2d(1024)
        #-->47*47
        #-->24*24
        self.conv500 = nn.Conv2d(1024,512,1)   
        self.conv510 = nn.Conv2d(512,1024,3)  
        self.conv501 = nn.Conv2d(1024,512,1)   
        self.conv511 = nn.Conv2d(512,1024,3)
        self.conv52  = nn.Conv2d(1024,1024,3)
        self.bt5 = nn.BatchNorm2d(1024)
        self.dp5 = nn.Dropout(0.2)
        #-->18*18
        #-->9*9
        
        self.conv60=nn.Conv2d(1024,1024,3)
        self.conv61=nn.Conv2d(1024,1024,3)
        self.bt6 = nn.BatchNorm2d(1024)
        self.dp6 = nn.Dropout(0.2)
        #-->8*8
               
        self.fc1 = nn.Linear(4096,4096)
        self.fc2 = nn.Linear(4096,3125)
        #-->25*25*5=3125
    def forward(self,x):
        #########1##################
        #in_size = x.size(0)
        out = self.conv1(x)
        out = self.bt1(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2)
        #print("Level1")
        #print(out.shape)
        #########2#####################
        out = self.conv2(out)
        out = self.bt2(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2)
        #print("Level2")
        #print(out.shape)
        #########3######################
        out = self.conv30(out)
        out = self.conv31(out)
        out = self.conv32(out)
        out = self.conv33(out)
        out = self.bt3(out)
        out = F.relu(out)   
        out = F.max_pool2d(out, 2, 2) 
        #print("Level3")
        #print(out.shape)
        ########4#######################
        out = self.conv400(out)
        out = self.conv410(out)
        out = self.conv401(out)
        out = self.conv411(out)
        out = self.conv402(out)
        out = self.conv412(out)
        out = self.conv403(out)
        out = self.conv413(out)
        out = self.conv42(out)
        out = self.conv43(out)
        out = self.bt4(out)
        out = F.relu(out)   
        #out = F.max_pool2d(out, 2, 2) 
        #print("Level4")
        #print(out.shape)
        ########5########################
        out = self.conv500(out)
        out = self.conv510(out)
        out = self.conv501(out)
        out = self.conv511(out)
        out = self.conv52(out)
        out = self.bt5(out)
        out = self.dp5(out)
        out = F.relu(out) 
        out = F.max_pool2d(out, 2, 2)
        #print("Level5")
        #print(out.shape)
        ##################################
        out = self.conv60(out)
        out = self.conv61(out)
        out = self.bt6(out)
        out = self.dp6(out)
        out = F.relu(out) 
#        print("Level6")
#        print(out.shape)
        ########FC########################
        out = out.view(out.size(0),-1)
        out = self.fc1(out)        
        out = self.fc2(out)
        #print("Level_fc2")
        #print(out.shape)
        out_final = out.view([-1,25,25,5])
        return out_final.float()
    
#############################################
"""""""""""""""""""""""""""""""""
debug
"""""""""""""""""""""""""""""""""

#"""
import torch
import numpy as np
rn=Region_Mask()
a=torch.tensor(np.zeros([1,3,250,250]),dtype=torch.float32)
a_torch=torch.tensor(a,dtype=torch.float32)
output=rn.forward(a)
#"""





















