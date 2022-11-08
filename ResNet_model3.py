from mimetypes import init
from typing_extensions import dataclass_transform
from matplotlib.pyplot import axes
from matplotlib.style import available
from torchvision.models import resnet50
import torchvision
import torchvision.transforms as tf 
import glob
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader 
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau as opt
from torch import nn
from sklearn.manifold import TSNE
from matplotlib import cm
from torch.nn.functional import interpolate 
from torch.utils.tensorboard import SummaryWriter
import glob
import torchmetrics
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.regularizers import LpRegularizer
from pytorch_metric_learning import miners,losses
import numpy as np
import cv2

writer=SummaryWriter('model_checkpoints/model_writer_mul')
img_train=glob.glob('data_ResNet/data_att_amsk/train/img/*.png')
label_train=glob.glob('data_ResNet/data_att_amsk/train/label/*.png')
img_val=glob.glob('data_ResNet/data_att_amsk/val/img/*.png')
label_val=glob.glob('data_ResNet/data_att_amsk/val/label/*.png')
img_test=glob.glob('data_ResNet/data_att_mask2/val/img/*.png')
label_test=glob.glob('data_ResNet/data_att_mask2/val/label/*.png')
colors=[[255,255,255],[0,0,255],[0,255,255],[0,255,0],[255,255,0]]
def tSNE(classes,code,i):
    color=['red','green','blue','yellow','black']
    labels =       {0: "Gebäude",
                        1: "Siedlung",
                        2: "Verkehr",
                        3: "Vegetation",
                        4: "Gewässer",
                        }
    all_classes=np.array(classes).astype(np.uint8)
    all_codes=np.array(code)
    #X=np.array(torch.stack(all_codes[0:500],dim=0).cpu())
    X_emb = TSNE(random_state=0).fit_transform(all_codes)
    
    cmap = cm.get_cmap('tab20')
    fig, ax = plt.subplots(figsize=(8,8))
    num_categories = 5
    for lab in range(num_categories):
        indices = all_classes==lab
        ax.scatter(X_emb[indices,0],X_emb[indices,1], c=color[lab], label = labels[lab] ,alpha=0.5)
    ax.legend(fontsize='large', markerscale=2)
    plt.savefig('tSNE{}.png'.format(i))
    plt.show()
def visualize(img,mask):
    mask_out=np.zeros(shape=(256,256,3))
    mask_out=np.where((np.expand_dims(mask,axis=-1)==1),[255,130,200],mask_out).astype(np.uint8) # mask in shape (256,256)
    cv2.imwrite('test2.png',cv2.cvtColor(mask_out,cv2.COLOR_RGB2BGR))
    cv2.imwrite('test.png',img)
class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x
def init_weights(m):
    torch.nn.init.constant_(m.mask1[1].weight.data, 0)
    torch.nn.init.constant_(m.mask1[2].bias.data, 1)
    torch.nn.init.constant_(m.mask2[1].weight.data, 0)
    torch.nn.init.constant_(m.mask2[2].bias.data, 1)
    torch.nn.init.constant_(m.mask3[1].weight.data, 0)
    torch.nn.init.constant_(m.mask3[2].bias.data, 1)
    torch.nn.init.constant_(m.mask4[1].weight.data, 0)
    torch.nn.init.constant_(m.mask4[2].bias.data, 1)
    torch.nn.init.constant_(m.mask5[1].weight.data, 0)
    torch.nn.init.constant_(m.mask5[2].bias.data, 1)
class model2(torch.nn.Module):
       # define model elements
    def __init__(self,ResNet):
        super(model2, self).__init__()
        self.layer0=nn.Sequential(nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        )
        self.mask1=nn.Sequential(Interpolate(size=(64,64), mode='bilinear'),
        nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
        nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )
        self.mask2=nn.Sequential(Interpolate(size=(64,64), mode='bilinear'),
        nn.Conv2d(1, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
        nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )
        self.mask3=nn.Sequential(Interpolate(size=(32,32), mode='bilinear'),
        nn.Conv2d(1, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
        nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )
        self.mask4=nn.Sequential(Interpolate(size=(16,16), mode='bilinear'),
        nn.Conv2d(1, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
        nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )
        self.mask5=nn.Sequential(Interpolate(size=(8,8), mode='bilinear'),
        nn.Conv2d(1, 2048, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True),
        nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )
        self.fc=nn.Linear(2048,128)
        self.mul=torch.mul
        self.ResNet=ResNet
        self.Relu=nn.ReLU(inplace=True)
        self.dropout=nn.Dropout2d(p=0.1)
    # forward propagate input
    def forward(self, x,mask):
        x=self.layer0(x)
        mask1=self.mask1(mask)
        mask1=self.Relu(mask1)
        x=self.mul(mask1,x)
        x=self.dropout(x)
        x=self.ResNet.layer1(x)
        mask2=self.mask2(mask)
        mask2=self.Relu(mask2)
        x=torch.mul(x,mask2)
        x=self.dropout(x)
        x=self.ResNet.layer2(x)
        mask3=self.mask3(mask)
        mask3=self.Relu(mask3)
        x=torch.mul(x,mask3)
        x=self.dropout(x)
        x=self.ResNet.layer3(x)
        mask4=self.mask4(mask)
        mask4=self.Relu(mask4)
        x=torch.mul(x,mask4)
        x=self.dropout(x)
        x=self.ResNet.layer4(x)
        mask5=self.mask5(mask)
        mask5=self.Relu(mask5)
        x=torch.mul(x,mask5) 
        x=self.ResNet.avgpool(x)
        x = x.view(x.size(0), -1)
        x=self.fc(x)
        x=self.ResNet.fc(x)
        return x
smooth_factor=0.00001
n_images=np.zeros(shape=(5,))

class dataset(Dataset):
    def __init__(self,img_id,label_id) :
        self.img_id=img_id
        self.label_id=label_id
        self.trans=tf.Compose([
            tf.ToTensor(),
            tf.Normalize(mean=[0.2130, 0.2283, 0.2079],std=[0.0764, 0.0741, 0.0638]),
        ])
        self.len_data=len(self.img_id)

    def __getitem__(self, index):
        example={}
        img=cv2.imread(self.img_id[index])
        img=self.trans(img)
        example['img']=img
        label=np.array(cv2.imread(self.label_id[index],-1)).astype(np.uint8)
        example['mask']=torch.unsqueeze(torch.from_numpy(label),axis=0).float()
        example['label']=torch.tensor(int(self.img_id[index].split('/')[-1].split('_')[1].split('.')[0]))
        return example
        
    def __len__(self):
        return self.len_data

class test(Dataset):
    def __init__(self,img_id,label_id) :
        self.img_id=img_id
        self.label_id=label_id
        self.trans=tf.Compose([
            tf.ToTensor(),
            tf.Normalize(mean=[0.2130, 0.2283, 0.2079],std=[0.0764, 0.0741, 0.0638]),
        ])
        self.len_data=len(self.img_id)

    def __getitem__(self, index):
        example={}
        img=cv2.imread(self.img_id[index])
        img=self.trans(img)
        example['img']=img
        label=np.array(cv2.imread(self.label_id[index],-1)).astype(np.uint8)
        example['mask']=torch.unsqueeze(torch.from_numpy(label),axis=0).float()
        example['label']=torch.tensor(int(self.img_id[index].split('/')[-1].split('_')[-1].split('.')[0]))
        example['img_id']=torch.tensor(int(self.img_id[index].split('/')[-1].split('_')[1]))
        example['path']=self.img_id[index]
        
        return example
    def __len__(self):
        return self.len_data

'''
list=[]
for data in dataset(img_train,label_train):
    list.append(data['label'])
list=torch.tensor(list).float()
hist=torch.histogram(list,bins=torch.tensor([0,1,2,3,4,5]).float())[0]
weight=torch.from_numpy(np.array([(sum(hist)/x) for x in hist])).float()
weight/=weight.max()
weight[0:3]+=0.08
weight[-2]-=0.07
#print (weight)
'''
train_loader=DataLoader(dataset(img_train,label_train),batch_size=64,shuffle=True)
val_loader=DataLoader(dataset(img_val,label_val),batch_size=300,shuffle=True)
#test_loader=DataLoader(test(img_test,label_test),batch_size=1,shuffle=True)

'''
# data test
data=next(iter(val_loader))
img,mask=np.array(data['img'][5].permute(1,2,0)*255).astype(np.uint8),np.squeeze(np.array(data['mask'][5].permute(1,2,0)).astype(np.uint8),axis=-1)
print(data['label'][5])
visualize(img,mask)
'''

model=resnet50(pretrained=True)
model.fc=torch.nn.Linear(in_features=128, out_features=5, bias=True)
model=model2(model.cuda()).cuda()
init_weights(model)
#new_model=torch.load('model_checkpoints/ResNet_last.pth')
#new_model.eval()

extracted_nodes=['fc', 'ResNet.fc']
new_model=create_feature_extractor(model,extracted_nodes).cuda()
new_model.train()


#print(out)

distance = CosineSimilarity()
reducer = ThresholdReducer(low=0)
loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
miner = miners.TripletMarginMiner(
    margin=0.2, distance=distance, type_of_triplets="semihard"
)
#optimizer=torch.optim.SGD(params=new_model.parameters(),lr=1e-1, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
s1=0
s2=0
lr_opt=opt(optimizer,mode='min',factor=0.1,patience=5,threshold=0.0001,min_lr=0)
for i in range(100):
    losses=[]
    losses_val=[]
    
    for data in train_loader:
        optimizer.zero_grad()
        result=new_model(data['img'].cuda(),data['mask'].cuda())['fc']
        hard_pairs = miner(result,data['label'].cuda())
        loss=loss_func(result,data['label'].cuda(),hard_pairs)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    
        writer.add_scalar('res/train_loss', loss, s1)
        s1+=1
    for data in val_loader:
        with torch.no_grad():
            result=new_model(data['img'].cuda(),data['mask'].cuda())['fc']
            loss=loss_func(result,data['label'].cuda())
            losses_val.append(loss.item())
            writer.add_scalar('res/val_loss', loss, s2)
            s2+=1
    loss_avg=torch.tensor(losses).mean().item()
    loss_val_avg=torch.tensor(losses_val).mean().item()
    lr_opt.step(loss_val_avg)
    writer.add_scalar('res/val_avg_loss', loss_val_avg, i)
    if (i+1)%5==0:
        torch.save(new_model,'model_checkpoints/ResNet_last.pth')
        print("epochs  = {} Average epoch loss: {} ".format(i+1,loss_avg))
        print("epochs  = {} Average epoch val_loss: {} ".format(i+1,loss_val_avg))
    


# build of t-SNE
'''

for i,data in enumerate(val_loader):
    with torch.no_grad():
        result=new_model(data['img'].cuda(),data['mask'].cuda())['fc'] # hier i get code of size (64,2048)
        #print(data['label'],'.....',torch.argmax(result,dim=1))
        #loss = loss_func(result, data['label'])
        #tSNE(data['label'],result.cpu(),i)
        #print(loss)
'''