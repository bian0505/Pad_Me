import torch,time
import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.dataloader as DataLoader
#from torchvision import datasets, transforms
#
import numpy as np
#import matplotlib.pyplot as plt


from YOLO_data_dataset import DealDataset
from modelclass import Region_Mask

torch.__version__

BATCH_SIZE=30 #256大概需要2G的显存
EPOCHS=3000 # 总共训练批次
DEVICE = torch.device("cuda")


dataset=DealDataset()
dataloader = DataLoader.DataLoader(dataset,batch_size= 1, shuffle = True, num_workers= 0)





def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.float().to(device)
        optimizer.zero_grad()
        output = model(data)
        #lossFunc = nn.MSELoss(reduction='mean')
        lossFunc = nn.MSELoss(reduction='sum')
        loss = lossFunc(output, target)
        loss.backward()
        optimizer.step()
        if(batch_idx+1)%10 == 0: 
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        return loss.item()




loss_array=np.zeros([EPOCHS,2])
try:
    model_cpu = torch.load("model_Large561.bian")
    model = model_cpu.to(DEVICE)
except:
    model = Region_Mask().to(DEVICE)
    print("model recreated")
     
for epoch in range(EPOCHS+1):
    #optimizer = optim.Adam(model.parameters())
    #optimizer = optim.Adam(model.parameters(),lr=0.001)
    LearnRate=0.001+0.01*(10**(-epoch/300))
    optimizer = optim.SGD(model.parameters(),lr=LearnRate,momentum=0.9)   
    loss=train(model, DEVICE, dataloader,optimizer, epoch)
    loss_array[epoch,0]=epoch
    loss_array[epoch,1]=loss
    if(epoch+1)%50 == 0:
        
        filedate="day"+str(time.localtime().tm_mday)+"hour"+str(time.localtime().tm_hour)+"min"+str(time.localtime().tm_min)
        model_filename="./model_ep"+str(epoch)+"_"+filedate+".bian"
        csv_filename="./loss_log.csv"
        torch.save(model,model_filename)
        np.savetxt(csv_filename, loss_array, delimiter = ',')  
    #test(model, DEVICE,NUMS)    
np.savetxt("./loss_Last.csv", loss_array, delimiter = ',') 
torch.save(model,"./model_Last.bian")
