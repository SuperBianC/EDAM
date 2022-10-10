import torch.nn as nn
import torch
import sys
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,precision_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.utils.data
import cv2_util
import cv2
import skimage.transform as trans


def transNumToStandardName(numSeq):
    names = {0:'Norm', 1:'CE',2:'AE',3:'HC'}
    return [names[num] for num in numSeq]

def transNameToNum(nameSeq):
    nums = {'norm': 0, 'CE': 1,'AE':2,'HC':3}
    return [nums[name] for name in nameSeq]

def getDiseaseTypeFromName(file):
    if file.startswith('CE'):
        return(1)
    elif file.startswith('AE'):
        return(2)
    elif file.startswith('HC'):
        return(3)
    elif file.startswith('norm'):
        return(0)
    else:return(3)
    
def dealimg(img):
    img2 = (img+200)/(200+200)
    img2[img2>1] = 1
    img2[img2<0] = 0
    img2 = img2*255
    img2 = trans.resize(img2,(img.shape[0],256,256))
    return img2

def hex2rgb(hexcolor):
    rgb = ((hexcolor >> 16) & 0xff,(hexcolor >> 8) & 0xff,hexcolor & 0xff)
    return rgb
           
def np2cv2(img):
    img = np.expand_dims(img, axis = 2)
    img = np.concatenate((img, img, img), axis = 2)
    return img

def getContour(mask):
    mask = np.array(mask,dtype='uint8')
    thresh = mask
    contours,_ = cv2_util.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    return contours


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(param)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

def trainGRUattn(dataloader, model, criterion, optimizer, device):
    model.train()
    epoch_losses = []
    epoch_accs = []
    for batch in tqdm(dataloader, desc='training...', file=sys.stdout):
        ids = batch['ids'].to(device)
        length = batch['length']
        label = batch['label'].to(device)
        _,prediction = model(ids, length)
        loss = criterion(prediction, label)
        accuracy,_,_ = get_accuracy(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy.item())
    return epoch_losses, epoch_accs

def evaluate(dataloader, model, criterion, device):
    model.eval()
    epoch_losses = []
    predicted_classes = []
    predicted_scores = []
    groundLabels = []
    totalName = []
    seqAttn = []
    seq = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='evaluating...', file=sys.stdout):
            ids = batch['ids'].to(device)
            seq.append(ids)
            length = batch['length']
            label = batch['label'].to(device)
            attenW, prediction = model(ids, length)
            attenW = attenW.squeeze(2).squeeze(0)
            seqAttn.append(attenW)
            name = batch['name']
            loss = criterion(prediction, label)    
            predicted_score = prediction.cpu().squeeze(0).numpy()
            predicted_scores.append(predicted_score)
            predicted_class = int(prediction.argmax(dim=-1).squeeze(0).cpu())
            groundLabel = int(label.squeeze(0).cpu())
            groundLabels.append(groundLabel)
            predicted_classes.append(predicted_class)
            epoch_losses.append(loss.item())
            totalName.append(name)
    f1score = f1_score(groundLabels,predicted_classes,average='macro')
    precision = precision_score(groundLabels,predicted_classes,average='macro',zero_division=0)
    epoch_accs = accuracy_score(groundLabels,predicted_classes)
    return seq,seqAttn,epoch_losses, epoch_accs,f1score,precision,groundLabels,predicted_scores,predicted_classes,totalName

def oneHot(a):
    b = np.zeros((a.size, a.max()+1))
    b[np.arange(a.size),a] = 1
    return b

def plot_confusion_matrix(cm, labels_name, title,cutcm = None):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]   
    cm = np.around(cm,2)
    ax = plt.subplot(1,1,1)
    if cutcm:
        cm = cm[:cutcm]
    plt.imshow(cm, interpolation='nearest',cmap='Blues')    
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            if cm[i,j]>0.5:
                cc = 'w'
            else:
                cc = 'black'
            text = plt.text(j, i, cm[i, j],
                           ha="center", va="center",color=cc)
    plt.title(title)    
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, )    
    if cutcm:
        plt.yticks(num_local[:cutcm], labels_name[:cutcm],rotation=90)    
        
    else:
        plt.yticks(num_local, labels_name,rotation=90)
 
    plt.ylabel('True label')    
    plt.xlabel('Predicted label')

    plt.tick_params(bottom=False, top=False, left=False, right=False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    
def get_accuracy(prediction, label):
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    accuracy_tmp = accuracy_score(label.cpu().numpy(),predicted_classes.cpu().numpy())
    f1score = f1_score(label.cpu().numpy(),predicted_classes.cpu().numpy(),average='macro')
    precisionScore = precision_score(label.cpu().numpy(),predicted_classes.cpu().numpy(),average='macro',zero_division=0)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy,f1score,precisionScore