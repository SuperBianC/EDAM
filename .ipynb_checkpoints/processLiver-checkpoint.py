import os
import SimpleITK as sitk
import numpy as np
import skimage.transform as trans
import imageio

def dealimg(img):
    img2 = (img+200)/(200+200)
    img2[img2>1] = 1
    img2[img2<0] = 0
    img2 = img2*255
    img2 = trans.resize(img2,(img.shape[0],256,256))
    return img2

def getDiseaseTypeFromName(file):
    if file.startswith('nang'):
        return('nang')
    elif file.startswith('pao'):
        return('pao')
    elif file.startswith('gnz'):
        return('gnz')
    elif file.startswith('xgl'):
        return('xgl')
    else:
        print('Never know,',file)

def extractLiver_OnePerson(data_path,seg_path,liverSeg_path, save_dir,dataName,label = 1):
    personPath = os.path.join(save_dir,dataName)
    if not os.path.exists(personPath):
        os.mkdir(personPath)
    if not os.path.exists(os.path.join(personPath,'livertumorData')):
        os.mkdir(os.path.join(personPath,'livertumorData'))
    if not os.path.exists(os.path.join(personPath,'livertumorMask')):
        os.mkdir(os.path.join(personPath,'livertumorMask'))
    dsd = os.path.join(personPath,'livertumorData')
    msd = os.path.join(personPath,'livertumorMask')

    sitk_liver = sitk.ReadImage(liverSeg_path)
    sitk_data = sitk.ReadImage(data_path)
    sitk_seg = sitk.ReadImage(seg_path)

    mask = sitk.GetArrayFromImage(sitk_seg)
    data = sitk.GetArrayFromImage(sitk_data)
    liverMask = sitk.GetArrayFromImage(sitk_liver)
    
    liver_top = min(np.where(liverMask == label)[0])
    liver_bottom = max(np.where(liverMask == label)[0])
    
    diseaseType = getDiseaseTypeFromName(dataName)
    
    ## nang,pao,gnz  1,2,3
    if diseaseType == 'nang':
        j = 1
    elif diseaseType=='pao':
        j = 2
    elif diseaseType=='gnz':
        j = 3
    else:
        print('warning, unkown disease type')
    data = dealimg(data)
    
    for i in range(liver_top,liver_bottom):
        maski = mask[i].astype(float)
        maski = trans.resize(maski,(256,256))
        maski[maski>=0.5] = j
        maski[maski<0.5] = 0
        datai = data[i]
        imageio.imwrite(os.path.join(dsd,'{0}{1}{2}.png'.format(dataName,'_',i)), datai.astype('uint8'))
        imageio.imwrite(os.path.join(msd,'{0}{1}{2}.png'.format(dataName,'_',i)), maski.astype('uint8'))
    return dataName