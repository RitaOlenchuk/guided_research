from mmseg.apis import inference_segmentor, init_segmentor
import os
import numpy as np
from os import listdir
from os.path import isfile, join

def get_patiens(file_list):
    return np.unique([file_prefix.split('_')[0]+'_'+file_prefix.split('_')[1] for file_prefix in file_list])

output_pred = '/media/data_4T/margaryta/base/center_level/Singapore/upernet_swin_tiny_patch4_80k_wmh_dice/pred'
truth = '/media/data_4T/bran/WMH_dataset/raw/training/singapore_flair_png/check/'
if not os.path.exists(output_pred):
    os.makedirs(output_pred)

if not os.path.exists(output_pred):
    os.makedirs(output_pred)

config_file = '/media/data_4T/margaryta/base/center_level/Singapore/upernet_swin_tiny_patch4_80k_wmh_dice/upernet_swin_tiny_patch4_80k_wmh_dice_si.py'
checkpoint_file = '/media/data_4T/margaryta/base/center_level/Singapore/upernet_swin_tiny_patch4_80k_wmh_dice/latest.pth'

to_test = '/media/data_4T/bran/WMH_dataset/raw/training/singapore_flair_png/val/images'

imgs = [f for f in listdir(to_test) if isfile(join(to_test, f))]
patients = get_patiens(imgs)

# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

for patient in patients:
    print(patient)
    files = [img for img in imgs if img.startswith(patient)]
    check = np.load(join(truth, patient+'.npy'))
    pred = np.zeros(check.shape)
    for file_ in files:
        path = join(to_test, file_)
        result = inference_segmentor(model, path)
        slice = file_.split('.')[0].split('_')[-1]
        pred[:,:,int(slice)] = result[0]
        np.save(join(output_pred, patient), pred)