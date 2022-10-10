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

def extractLiver_OnePerson(data_path,liverSeg_path,save_dir,dataName,label = 1):
    personPath = os.path.join(save_dir,dataName)
    if not os.path.exists(personPath):
        os.mkdir(personPath)
    if not os.path.exists(os.path.join(personPath,'liverData')):
        os.mkdir(os.path.join(personPath,'liverData'))
    if not os.path.exists(os.path.join(personPath,'liverPredMask')):
        os.mkdir(os.path.join(personPath,'liverPredMask'))
    dsd = os.path.join(personPath,'liverData')
    msd = os.path.join(personPath,'liverPredMask')

    sitk_liver = sitk.ReadImage(liverSeg_path)
    sitk_data = sitk.ReadImage(data_path)
    data = sitk.GetArrayFromImage(sitk_data)
    liverMask = sitk.GetArrayFromImage(sitk_liver)
    liver_top = min(np.where(liverMask == label)[0])
    liver_bottom = max(np.where(liverMask == label)[0])
    
    data = dealimg(data)
    
    for i in range(liver_top,liver_bottom):
        datai = data[i]
        maski = liverMask[i].astype(float)
        maski = trans.resize(maski,(256,256))
        maski[maski>=0.5] = 1
        maski[maski<0.5] = 0
        imageio.imwrite(os.path.join(dsd,'{0}{1}{2}.png'.format(dataName,'_',i)), datai.astype('uint8'))
        imageio.imwrite(os.path.join(msd,'{0}{1}{2}.png'.format(dataName,'_',i)), maski.astype('uint8'))
    return dataName