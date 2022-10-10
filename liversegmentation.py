import os
import torch
import random
import collections
from time import time
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage
import torch.backends.cudnn as cudnn
from net.RUNET import ResUNet
import warnings
import skimage.morphology as morphology
import skimage.measure as measure
import copy
warnings.filterwarnings("ignore")

# set seed
seed = 1029
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# set parameter
test_ct_path = 'data-input/patient-example/'
model_path = "module/pretrained/liver-segmentation.pth"
pred_path = 'results/liver-segmentation/'

liver_score = collections.OrderedDict()
liver_score['dice'] = []
liver_score['time'] = []
file_name = []
size = 48
down_scale =0.5
upper, lower = 157, -126  # cut off the CTS
threshold = 0.5
maximum_hole = 5e4
stride = 12

# load net
use_gpu = 1
if use_gpu:

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    net = torch.nn.DataParallel(ResUNet(training=False)).cuda()
    cudnn.benchmark = False
else:
    net = torch.nn.DataParallel(ResUNet(training=False))
net.load_state_dict(torch.load(model_path))
net = net.eval()


for file_index, file in enumerate(os.listdir(test_ct_path)):
    start = time()
    file_name.append(file)
    ct = sitk.ReadImage(os.path.join(test_ct_path, file), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)
    origin_shape = ct_array.shape
    ct_array[ct_array > upper] = upper
    ct_array[ct_array < lower] = lower

    ct_array = ct_array.astype(np.float32)
    ct_array = ct_array / 200

    ct_array = ndimage.zoom(ct_array, (1, 0.5,0.5), order=3)

    print(ct_array.shape)
    too_small = False
    if ct_array.shape[0] < size:
        depth = ct_array.shape[0]
        temp = np.ones((size, int(512 * 0.5), int(512 * 0.5))) * lower
        temp[0: depth] = ct_array
        ct_array = temp 
        too_small = True
    start_slice = 0
    end_slice = start_slice + size - 1
    count = np.zeros((ct_array.shape[0], 512, 512), dtype=np.int16)
    probability_map = np.zeros((ct_array.shape[0], 512, 512), dtype=np.float32)

    with torch.no_grad():
        while end_slice < ct_array.shape[0]:
            ct_tensor = torch.FloatTensor(ct_array[start_slice: end_slice + 1]).cuda()
            ct_tensor = ct_tensor.unsqueeze(dim=0).unsqueeze(dim=0)
            outputs = net(ct_tensor)

            count[start_slice: end_slice + 1] += 1
            probability_map[start_slice: end_slice + 1] += np.squeeze(outputs.cpu().detach().numpy())
            del outputs      

            start_slice += stride
            end_slice = start_slice + size - 1

        if end_slice != ct_array.shape[0] - 1:
            end_slice = ct_array.shape[0] - 1
            start_slice = end_slice - size + 1

            ct_tensor = torch.FloatTensor(ct_array[start_slice: end_slice + 1]).cuda()
            ct_tensor = ct_tensor.unsqueeze(dim=0).unsqueeze(dim=0)
            outputs = net(ct_tensor)

            count[start_slice: end_slice + 1] += 1
            probability_map[start_slice: end_slice + 1] += np.squeeze(outputs.cpu().detach().numpy())

            del outputs

        pred_seg = np.zeros_like(probability_map)
        pred_seg[probability_map >= (threshold * count)] = 1


        if too_small:
            temp = np.zeros((depth, 512, 512), dtype=np.float32)
            temp += pred_seg[0: depth]
            pred_seg = temp

    pred_seg = pred_seg.astype(np.uint8)
    liver_seg = copy.deepcopy(pred_seg)
    liver_seg = measure.label(liver_seg, connectivity=2)
    props = measure.regionprops(liver_seg)
    max_area = 0
    max_index = 0
    for index, prop in enumerate(props, start=1):
        if prop.area > max_area:
            max_area = prop.area
            max_index = index

    liver_seg[liver_seg != max_index] = 0
    liver_seg[liver_seg == max_index] = 1

    liver_seg = liver_seg.astype(bool)
    morphology.remove_small_holes(liver_seg, maximum_hole, connectivity=2, in_place=True)
    liver_seg = liver_seg.astype(np.uint8)
    pred_seg = sitk.GetImageFromArray(liver_seg)
    pred_seg.SetDirection(ct.GetDirection())
    pred_seg.SetOrigin(ct.GetOrigin())
    pred_seg.SetSpacing(ct.GetSpacing())
    sitk.WriteImage(pred_seg, os.path.join(pred_path, file.replace('.nii', '_pred.nii.gz')))
    
    speed = time() - start

    print(file_index, 'this case use {:.3f} s'.format(speed))
    print('-----------------------')