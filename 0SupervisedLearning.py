# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 13:56:11 2019

@author: bian
"""
import matplotlib.pyplot as plt
import re,cv2,time,os
import eventlet
import people0,people
import find_pad


eventlet.monkey_patch()
f = open("datasheet.csv")             # 返回一个文件对象
line = f.readline()             # 调用文件的 readline()方法
ssd_model,utils=people.people_detec_ini()

######Set Begin Line##############
begin=3472
for cont in range(begin):
    time.sleep(0.01)
    line=f.readline()    
    print(cont)
####################################
print(line)
while line:
    url_read=re.match('SZLT2018/.*?,', line, flags=0)
    try:
        url1_0=url_read.group(0)
    except:
        print('! read url failed !')
        time.sleep(0.01)
        line = f.readline()
        time.sleep(0.01)
        continue  
    url1=url1_0[:-1]
    url2="http://cdn.ptbchina.com/";
    url=url2+url1;
    
    #find the numbers
    #--->'[""1234","5678""]
    regex=re.compile(r'\[.*\]')
    num_read=regex.findall(line)
    #--->['1234','5678']
    regex=re.compile(r'\d{4}')
    nums=regex.findall(num_read[0])
    
    
    if(len(nums)==1):
        #download picture & find people
        downloaded=0
        with eventlet.Timeout(10,False):
            print('Image Downloading......')
            cap=cv2.VideoCapture(url)
            img0=cap.read()
            downloaded=1
            print('Image Downloaded......')
        if(downloaded==0):
            continue
        try:
            #result_test=people.people_detec(url,ssd_model,utils)
            result_test=[]
            with eventlet.Timeout(10,False):
                print("Wei  is fucking YYJ..........")
                result_test=people0.people_detec(img0[1],url,ssd_model,utils)
                print("Wei has fucked  YYJ..........")
                
            img=img0[1]
#            if (img.shape[0]>img.shape[1]):
#                x0=int(img.shape[0]*result_test[1])+240
#                y0=int(img.shape[1]*result_test[0])
#                x1=int(img.shape[0]*result_test[3])+240
#                y1=int(img.shape[1]*result_test[2])
#            else:
#                x0=int(img.shape[0]*result_test[1])
#                y0=int(img.shape[1]*result_test[0])+240
#                x1=int(img.shape[0]*result_test[3])
#                y1=int(img.shape[1]*result_test[2])+240
            if (img.shape[0]>img.shape[1]):
                x0=max(int(result_test[1])+240,0)
                y0=max(int(result_test[0]),0)
                x1=max(int(result_test[3])+240,0)
                y1=max(int(result_test[2]),0)
            else:
                x0=max(int(result_test[1]),0)
                y0=max(int(result_test[0])+240,0)
                x1=max(int(result_test[3]),0)
                y1=max(int(result_test[2])+240,0)
            human=img[x0:x1,y0:y1]
            human_tot_0=img[x0:x1,y0:y1]
            human_target_0=human_tot_0-human_tot_0
            human_tot=img
            human_target=human_tot-human_tot
            
            if(human.shape[0]*human.shape[1]>0):
                print("Pad Searching ...........")
                pad=find_pad.find_pad(human)
                print("Pad Detected......")
            else:
                print("Fucking the 'people0.py'")
                continue
            
            
            ################################
            #
            #plt.imshow(pad[0])
            #plt.pause(0.1)
            
            #print('Correct? Press "0"')
            #print('Otherwise? Press "1"')
            #
            num_check='0'
            if(num_check=='0'):
######################Debug-----------> Check pads##########################                
#                human_tot[x0+pad[1]+200:x0+pad[2]+200,y0+pad[3]:y0+pad[4]]=255
#                plt.imshow(human_tot)
#                plt.pause(0.1)
#                a=input()
############################################################################                
                path_write="./imgsave/0small_pad/" + url1[9:-4] + "[" + nums[0] + "].jpg"
                cv2.imwrite(path_write, pad[0])
                print('Image0 Saved')
                
                path_write="./imgsave/1origin/" + url1[9:-4] + "[" + nums[0] + "].jpg"
                cv2.imwrite(path_write, human_tot)
                print('Image1 Saved')
                
                human_target[x0+pad[1]+200:x0+pad[2]+200,y0+pad[3]:y0+pad[4]]=255
                path_write="./imgsave/2target/" + url1[9:-4] + "[" + nums[0] + "].jpg"
                cv2.imwrite(path_write, human_target)
                print('Image2 Saved')
                
                path_write="./imgsave/3human/" + url1[9:-4] + "[" + nums[0] + "].jpg"
                cv2.imwrite(path_write, human_tot_0)
                print('Image3 Saved')
                
                human_target_0[pad[1]:pad[2],pad[3]:pad[4]]=255
                path_write="./imgsave/4human_target/" + url1[9:-4] + "[" + nums[0] + "].jpg"
                cv2.imwrite(path_write, human_target_0)
                print('Image4 Saved')
                

        except:
            print('Someting Wrong, Next Image')
        ################################## 
        
        
    time.sleep(0.01)
    line = f.readline()
    time.sleep(0.01)
    begin=begin+1
    print('"begin" is:   ',begin)

f.close()