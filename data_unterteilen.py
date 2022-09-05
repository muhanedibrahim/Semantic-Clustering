from cProfile import label
import torch
import cv2
import rasterio as ra
from rasterio.plot import reshape_as_raster, reshape_as_image
from omegaconf import OmegaConf
from taming.models.cond_transformer import Net2NetTransformer
import yaml
import albumentations
import images_id
import torch.nn as nn
from PIL import Image
import numpy as np
import glob
from matplotlib import cm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from taming.models import vqgan
from torch.utils.data import DataLoader,Dataset
from taming.data.custom3 import CustomTrain,CustomTest
from torch.autograd import Variable
#from tsne_torch import TorchTSNE as TSNE

labels =       {0: "Gebäude",
                        1: "Siedlung",
                        2: "Verkehr",
                        3: "Vegetation",
                        4: "Gewässer",
                        }
def tSNE(classes,code):
    all_classes=[]
    all_codes=[]
    for l,list in enumerate(classes):
        for i in list:
            all_classes.append(i)
            all_codes.append(code[l])
    X=np.array(torch.stack(all_codes[0:500],dim=0).cpu())
    X_emb = TSNE(random_state=0).fit_transform(X)
    
    cmap = cm.get_cmap('tab20')
    fig, ax = plt.subplots(figsize=(8,8))
    num_categories = 5
    for lab in range(num_categories):
        indices = np.array(all_classes[0:500])==lab
        ax.scatter(X_emb[indices,0],X_emb[indices,1], c=np.array(cmap(lab)).reshape(1,4), label = labels[lab] ,alpha=0.5)
    ax.legend(fontsize='large', markerscale=2)
    plt.savefig('tSNE.png')
    plt.show()
    

data=CustomTrain() # bilder und Masken (5 klassen), die aus (Dortmund,Remscheid..) sind, wurden in einem ordner gespeichert
val=CustomTest()
dataloader=DataLoader(data,batch_size=32)
dataloader_val=DataLoader(val,batch_size=12)
config_path = "logs/2022-07-17T12-13-05_VQmodel/configs/2022-07-19T12-50-31-project.yaml"
config = OmegaConf.load(config_path)
model = vqgan.VQModel(**config.model.params)
ckpt_path = "logs/2022-07-17T12-13-05_VQmodel/checkpoints/last.ckpt"
sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
missing, unexpected = model.load_state_dict(sd, strict=False)
model.cuda().eval()
torch.set_grad_enabled(False)

epochs=1
classes=[]
codes=[]
batches_data={}
batches_data_val={}

class MLP(nn.Module):
    # define model elements
    def __init__(self,):
        super(MLP, self).__init__()
        self.layer1=nn.Linear(256,200)
        self.activation1=nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.layer2=nn.Linear(200,200)
        self.layer3=nn.Linear(200,150)
        self.layer4=nn.Linear(150,50)
        self.layer5=nn.Linear(50,5)
        self.activation2=nn.Sigmoid()
    # forward propagate input
    def forward(self, x):
        x=self.layer1(x)
        x=self.activation1(x)
        x = self.dropout(x)
        x=self.layer2(x)
        x=self.activation1(x)
        x = self.dropout(x)
        x=self.layer3(x)
        x=self.activation1(x)
        x = self.dropout(x)
        x=self.layer4(x)
        x=self.activation1(x)
        x=self.layer5(x)
        x=self.activation2(x)
        return x
class dataset2(Dataset):
    def __init__(self,codes,classes):
        self.codes=codes
        self.classes=classes
        self.length=len(codes)

    def onehot_encoding(self,list):
        encoded_classes=np.zeros((5,))
        for i in list:
            encoded_classes[i]=1    
        return torch.from_numpy(encoded_classes).float()
    def __getitem__(self, index):
        classe=self.onehot_encoding(self.classes[index])
        code=self.codes[index].float()
        return code,classe
    def __len__(self):
        return self.length
for i in [dataloader,dataloader_val]:
    for input in i:
        mask=input['segmentation'].cuda() 
        image= model.get_input(input, 'image').cuda()
        quant, diff,_, _ = model.encode(image)        # code-book für die mini-batches mit der Große [1,256,16,16], weil attention resolution = 16
        
        # hier werden die mini_batches aus Masken den Embeddingvektoren zugeordnet
        
        for l in range(quant.shape[0]):  
            for k in range(16):
                for j in range(16):
                    mini_Batch_code=quant[l,:,k,j]
                    mask_batch=mask[l,k*16:(k+1)*16,j*16:(j+1)*16]
                    classes_in_batch=torch.unique(mask_batch)
                    classes.append(classes_in_batch.cpu().tolist())
                    codes.append(mini_Batch_code.cpu())
    if i == dataloader:  
        batches_data['classes']=classes
        batches_data['code']=codes
    else:
        batches_data_val['classes']=classes
        batches_data_val['code']=codes
'''
Histogram=np.zeros(shape=(5,))
for i in range(5):
    for list in batches_data['classes']:
        Histogram[i]+=list.count(i)
print(Histogram)
tSNE(batches_data['classes'],batches_data['code'])
'''


dataset_final=dataset2(batches_data['code'],batches_data['classes'])
dataset_final_val=dataset2(batches_data_val['code'],batches_data_val['classes'])
dataloader2=DataLoader(dataset_final,batch_size=32)
dataloader2_val=DataLoader(dataset_final_val,batch_size=32)
model2=MLP().cuda()
#model2=torch.load('ALKIS_dataset_master/saved_models/MLP_model.pth').cuda()
criterion = nn.BCELoss().cuda()
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.0005)
model2.train()


for i in range(600):
    losses=[]
    losses_val=[]
    for data in dataloader2:
        with torch.set_grad_enabled(True):
            optimizer2.zero_grad()
            result=model2(data[0].cuda())
            loss=criterion(result,data[1].cuda())
            losses.append(loss.item())
            loss.backward()
            optimizer2.step()
    for data in dataloader2_val:
        with torch.no_grad():
            result=model2(data[0].cuda())
            loss=criterion(result,data[1].cuda())
            losses_val.append(loss.item())
    torch.save(model2,'ALKIS_dataset_master/saved_models/MLP_model_new.pth')
    loss_avg=torch.tensor(losses).mean().item()
    loss_val_avg=torch.tensor(losses_val).mean().item()
    print("epochs  = {} Average epoch loss: {} ".format(i+1,loss_avg))
    print("epochs  = {} Average epoch val_loss: {} ".format(i+1,loss_val_avg))
