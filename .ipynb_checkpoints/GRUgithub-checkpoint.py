import seaborn as sns
from sklearn.metrics import recall_score
import functools
import sys
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score,f1_score,precision_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import pickle

with open("testDick0685_1norm.pkl", "rb") as tf:
    testWithNorm = pickle.load(tf)

def transNumToStandardName(numSeq):
    names = {0:'Norm', 1:'CE',2:'AE',3:'HC'}
    return [names[num] for num in numSeq]

def transNameToNum(nameSeq):
    nums = {'norm': 0, 'nang': 1,'pao':2,'gnz':3}
    return [nums[name] for name in nameSeq]

def getDiseaseTypeFromName(file):
    if file.startswith('nang'):
        return(1)
    elif file.startswith('pao'):
        return(2)
    elif file.startswith('gnz'):
        return(3)
    elif file.startswith('xgl'):
        return(4)
    elif file.startswith('norm'):
        return(0)
    else:
        return(3)

def numericalize_data(targetDict, max_length = 350,dataPading = False, padidx = 4):
    DicList = []
    for key in list(targetDict.keys()):
        first = getDiseaseTypeFromName(key)
        numSeq = transNameToNum(targetDict[key])
        cl = len(numSeq)
        if cl >max_length:
            cutting_len = int((cl - max_length)/2)
            numSeq = numSeq[cutting_len:cl - cutting_len]
        lengthFlag = len(numSeq)
        if dataPading and len(numSeq)<max_length:
            numSeq += [padidx]*(max_length-len(numSeq))
        DicList.append({'name':key,'ids':numSeq,'label':first,'length':lengthFlag})
    return DicList

class SliceDataset(Dataset):
    def __init__(self, DictList, transform=None, target_transform=None):
        self.slices = DictList
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        SliceDict = {}
        tmp = self.slices[index]
        SliceDict['ids'] = torch.LongTensor(tmp['ids'])
        SliceDict['label'] = torch.scalar_tensor(tmp['label']).long()
        SliceDict['length'] = torch.scalar_tensor(tmp['length']).long()
        SliceDict['name'] = tmp['name']
        return SliceDict

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

def collate(batch, pad_index):
    batch_ids = [i['ids'] for i in batch]
    batch_ids = nn.utils.rnn.pad_sequence(batch_ids, padding_value=pad_index, batch_first=True)
    batch_length = [i['length'] for i in batch]
    batch_length = torch.stack(batch_length)
    batch_label = [i['label'] for i in batch]
    batch_label = torch.stack(batch_label)
    batch_name = [i['name'] for i in batch]
    batch = {'ids': batch_ids,
             'length': batch_length,
             'label': batch_label,
            'name':batch_name}
    return batch

def trainGRUattn(dataloader, model, criterion, optimizer, device):

    model.train()
    epoch_losses = []
    epoch_accs = []
    for batch in tqdm.tqdm(dataloader, desc='training...', file=sys.stdout):
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
        for batch in tqdm.tqdm(dataloader, desc='evaluating...', file=sys.stdout):
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

def get_accuracy(prediction, label):
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    accuracy_tmp = accuracy_score(label.cpu().numpy(),predicted_classes.cpu().numpy())
    f1score = f1_score(label.cpu().numpy(),predicted_classes.cpu().numpy(),average='macro')
    precisionScore = precision_score(label.cpu().numpy(),predicted_classes.cpu().numpy(),average='macro',zero_division=0)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy,f1score,precisionScore

def metricsOFdisease(ddList,groundLabels,predicted_classes): # dd in [0,1,2,3]
    metricTable = pd.DataFrame()
    for dd in ddList:
        namesDic = {'1':'nang','2':'pao','3':'gnz','0':'norm'}
        ddvsother_gt = []
        ddvsother_pred = []
        for gg in groundLabels:
            if gg in [dd]:
                ddvsother_gt.append(1)
            else:
                ddvsother_gt.append(0)
        for gg in predicted_classes:
            if gg in [dd]:
                ddvsother_pred.append(1)
            else:
                ddvsother_pred.append(0)
        f1ss = f1_score(ddvsother_gt,ddvsother_pred)
        accss = accuracy_score(ddvsother_gt,ddvsother_pred)
        press = precision_score(ddvsother_gt,ddvsother_pred)
        senss = recall_score(ddvsother_gt,ddvsother_pred)
        tn, fp, fn, tp = confusion_matrix(ddvsother_gt,ddvsother_pred).ravel()
        specificity = tn / (tn+fp)
        print(namesDic[str(dd)])
        print('acc:',accss,'sensitivity:',senss,'precision:',press,'specificity:',specificity,'f1:',f1ss,)
        metr = [accss,senss,specificity,f1ss]
        metricTable[namesDic[str(dd)]] = metr
    return metricTable

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
    
def test_data_cm(model):
    seq,seqAttn,_,_,f1score,precision,gl,ps,pc,tn = evaluate(test_dataloader, model, criterion, device)
    ps = np.array(ps)
    pred = transNumToStandardName(pc)
    groundTruth = transNumToStandardName(gl)
    labels=["CE", "AE", "HC",'Norm']
    cm = confusion_matrix(groundTruth, pred,labels=labels)
    plot_confusion_matrix(cm,labels_name=labels,title = 'cm')
    return seq,seqAttn

def metricsOFdisease(ddList,groundLabels,predicted_classes): 
    metricTable = pd.DataFrame()
    for dd in ddList:
        namesDic = {'1':'nang','2':'pao','3':'gnz','0':'norm'}
        ddvsother_gt = []
        ddvsother_pred = []
        for gg in groundLabels:
            if gg in [dd]:
                ddvsother_gt.append(1)
            else:
                ddvsother_gt.append(0)
        for gg in predicted_classes:
            if gg in [dd]:
                ddvsother_pred.append(1)
            else:
                ddvsother_pred.append(0)
        f1ss = f1_score(ddvsother_gt,ddvsother_pred)
        accss = accuracy_score(ddvsother_gt,ddvsother_pred)
        press = precision_score(ddvsother_gt,ddvsother_pred)
        senss = recall_score(ddvsother_gt,ddvsother_pred)
        tn, fp, fn, tp = confusion_matrix(ddvsother_gt,ddvsother_pred).ravel()
        specificity = tn / (tn+fp)
        print(namesDic[str(dd)])
        print('acc:',accss,'sensitivity:',senss,'precision:',press,'specificity:',specificity,'f1:',f1ss,)
        metr = [accss,senss,specificity,f1ss]
        metricTable[namesDic[str(dd)]] = metr
    return metricTable

### define model
class GRU_with_Attention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional,
                 dropout_rate, pad_index):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers, bidirectional=bidirectional,
                            dropout=dropout_rate, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
    def attention(self, lstm_output, final_state):
        lstm_output = lstm_output.permute(1, 0, 2)
        merged_state = torch.cat([s for s in final_state], 1)
        merged_state = merged_state.unsqueeze(2)
        weights = torch.bmm(lstm_output, merged_state)
        weights = F.softmax(weights.squeeze(2), dim=1).unsqueeze(2)
        return weights, torch.bmm(torch.transpose(lstm_output, 1, 2), weights).squeeze(2)
        
    def forward(self, ids, length):
        embedded = self.embedding(ids)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, length, batch_first=True, 
                                                            enforce_sorted=False)
        packed_output, hidden = self.gru(packed_embedded)
        output, output_length = nn.utils.rnn.pad_packed_sequence(packed_output)
        if self.gru.bidirectional:
            hidden = self.dropout(torch.cat([hidden[-1], hidden[-2]], dim=-1))
        else:
            hidden = self.dropout(hidden[-1])
        attenW, attn_output = self.attention(output, hidden.unsqueeze(0))
        prediction = self.fc(attn_output)
        return attenW,prediction

### load model
maxLen = 230
vocab_size = 5
embedding_dim = 5 
hidden_dim = 50
output_dim = 4
n_layers = 1
bidirectional = True
dropout_rate = 0
pad_index = 4

modelGRU_atten = GRU_with_Attention(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout_rate, 
             pad_index = pad_index)
save = torch.load('pretrained/patient-level-GRU-attention.pt',map_location=torch.device('cpu')) 
device = torch.device('cpu')
modelGRU_atten.load_state_dict(save)
modelGRU_atten.to(device)
modelGRU_atten.eval()
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)
collate = functools.partial(collate, pad_index=pad_index)

### test data
testDicList = numericalize_data(testWithNorm,max_length=maxLen,dataPading=False)
test_data = SliceDataset(testDicList)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, collate_fn=collate,shuffle = False)
seq,seqAttn,epoch_losses, epoch_accs,f1score,precision,groundLabels,predicted_scores,predicted_classes,totalName = evaluate(test_dataloader,modelGRU_atten,criterion,device)
predScores = torch.softmax(torch.tensor(predicted_scores),1)
predScores = np.around(np.array(predScores.tolist()),3)
for pss,pcs in zip(predScores, predicted_classes):
    print('the patient is predicted as {} with confidenct {}'.format(transNumToStandardName([pcs])[0],pss[pcs]))

### slice-attention map
tnames = []
for batch in test_dataloader:
    tnames.append(batch['name'][0])
num = 0
print(tnames[num])
plt.figure(figsize=[20,2])
plt.subplot(2,1,1)
ax = sns.heatmap([seq[num].detach().cpu()[0].numpy()],square=False,cbar = False,linewidths=0.1,cmap = 'Accent',vmax=4,vmin=0)
ax.xaxis.tick_top()
plt.xticks(rotation=90)
plt.yticks([])
plt.subplot(2,1,2)
sns.heatmap([seqAttn[num].detach().cpu().numpy()],square=False,linewidths=0,cmap = 'Reds',cbar = False,)
plt.yticks([])
plt.xticks([])