import os
from tqdm import tqdm
import numpy as np
import os                
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import os
import torch
import numpy as np
import torch.utils.data
from PIL import Image
import transforms as T
import re

def get_instance_segmentation_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model

def getSegmentationOnePic(image_path, masklabel_confidence = 0.6):
    imm = Image.open(image_path)
    imm = np.array(imm)
    transforms = []
    transforms.append(T.ToTensor())
    immmm = T.Compose(transforms)(imm,target=False)
    model.eval()
    prediction = model([immmm[0].to(device)])
    labels = prediction[0]['labels']
    labels = labels.cpu().detach().numpy()
    scores = prediction[0]['scores']
    scores = scores.cpu().detach().numpy()
    masks = prediction[0]['masks']
    m1 = masks[0][0].detach().cpu()
    m1[m1>=masklabel_confidence]= int(labels[0])
    m1[m1<masklabel_confidence] = 0
    return m1

def getDiagnosisOnePic(image_path,masklabel_confidence = 0.6):
        imm = Image.open(image_path)
        imm = np.array(imm)
        transforms = []
        transforms.append(T.ToTensor())
        immmm = T.Compose(transforms)(imm,target=False)
        prediction = model([immmm[0].to(device)])
        labels = prediction[0]['labels']
        labels = labels.cpu().detach().numpy()
        scores = prediction[0]['scores']
        scores = scores.cpu().detach().numpy()
        label_idx = [i for i in range(len(scores)) if scores[i] > masklabel_confidence]
        if len(label_idx) == 0:
            name = 'norm'
        else:
            scs = scores.tolist()
            name = names.get(str(labels[scs.index(max(scs))].item()))
        return name

def diagnose_one_person(root1,masklabel_confidence):
    lst = os.listdir(root1)
    print(root1)
    lst.sort(key=lambda l: int(re.findall('\d+',l.split('_')[-1])[0]))
    imgss = lst
    people_sli = []
    for pickOne in imgss:
        imm = Image.open(os.path.join(root1,pickOne))
        imm = np.array(imm)
        transforms = []
        transforms.append(T.ToTensor())
        immmm = T.Compose(transforms)(imm,target=False)
        prediction = model([immmm[0].to(device)])
        labels = prediction[0]['labels']
        labels = labels.cpu().detach().numpy()
        scores = prediction[0]['scores']
        scores = scores.cpu().detach().numpy()
        label_idx = [i for i in range(len(scores)) if scores[i] > masklabel_confidence]
        if len(label_idx) == 0:
            name = 'norm'
        else:
            scs = scores.tolist()
            name = names.get(str(labels[scs.index(max(scs))].item()))
        people_sli.append(name)
    return people_sli

def diagnose_all_people(rootAll,masklabel_confidence = 0.8):
    rootList = os.listdir(rootAll)
    peopleDiagSequence = {}
    for rootp in tqdm(rootList):
        personRoot = rootAll + rootp
        people_slis = diagnose_one_person(os.path.join(personRoot,'livertumorData'),masklabel_confidence)
        peopleDiagSequence[rootp] = people_slis
        print(people_slis)
    return peopleDiagSequence


names = {'0': 'norm', '1': 'nang','2':'pao','3':'gnz'}
device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 4
model = get_instance_segmentation_model(num_classes)
save = torch.load('pretrained/slice-level-prediction.pth') 
model.load_state_dict(save['model'])
model.to(device)
model.eval()