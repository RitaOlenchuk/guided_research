import os
import csv
from PIL import Image
from os.path import join
import numpy as np
import nibabel as nib
#from numpngw import write_png
from PIL import Image

output_train = '/media/data_4T/bran/WMH_dataset/raw/training/patient_specific_flair_png3/train'
output_test = '/media/data_4T/bran/WMH_dataset/raw/training/patient_specific_flair_png3/val'
output_check = '/media/data_4T/bran/WMH_dataset/raw/training/patient_specific_flair_png3/check'
SLICE_THRSHL = 0.16

palette = [[0, 0, 0], [255, 255, 255]]

def generate_images(scans_T1, scans_FLAIR, masks):
    print(str(len(scans_FLAIR)))
    train_patients_ids = np.random.choice(a=np.arange(len(scans_FLAIR)), size=int(len(scans_FLAIR)*(4/5)), replace=False)
    test_pations_ids = list(set(range(len(scans_FLAIR))) - set(train_patients_ids))

    train_masks =  np.array(masks)[train_patients_ids]

    train_scans_T1 = [path for path in scans_T1 if any(path.startswith(patient_path[:-10]) for patient_path in train_masks)]
    train_scans_FLAIR = [path for path in scans_FLAIR if any(path.startswith(patient_path[:-10]) for patient_path in train_masks)]

    test_masks = np.array(masks)[test_pations_ids]
    test_scans_T1 = [path for path in scans_T1 if not path in train_scans_T1]
    test_scans_FLAIR = [path for path in scans_FLAIR if not path in train_scans_FLAIR]
    
    train_masks.sort()
    train_scans_T1.sort()
    train_scans_FLAIR.sort()

    test_masks.sort()
    test_scans_T1.sort()
    test_scans_FLAIR.sort()
    split_train = dict()
    print(str(len(train_scans_FLAIR)), str(len(test_scans_FLAIR)))
    for i in range(len(train_masks)): 
        patient = train_scans_FLAIR[i].split('/')[-3]
        medc = train_scans_FLAIR[i].split('/')[-4]
        
        nii_img_FLAIR  = nib.load(train_scans_FLAIR[i])
        nii_idata_FLAIR = nii_img_FLAIR.get_fdata()  
        
        nii_img_T1  = nib.load(train_scans_T1[i])
        nii_idata_T1 = nii_img_T1.get_fdata()  
        
        nii_msk  = nib.load(train_masks[i])
        nii_mdata = nii_msk.get_fdata()

        bottom_limit = nii_mdata.shape[2]*SLICE_THRSHL
        upper_limit = nii_mdata.shape[2] - bottom_limit
        
        for slice in range(nii_mdata.shape[2]):
            if slice>bottom_limit and slice<upper_limit:
                max_value = max(nii_msk.shape[0], nii_msk.shape[1])
                img = np.zeros((max_value, max_value, 2))
                
                img[:,:,0] = np.min(normalize(nii_idata_FLAIR[:,:,slice]))
                img[:,:,1] = np.min(normalize(nii_idata_T1[:,:,slice]))

                min0 = (max_value-nii_idata_FLAIR[:,:,slice].shape[0])//2
                min1 = (max_value-nii_idata_FLAIR[:,:,slice].shape[1])//2

                img[min0:max_value-min0,min1:max_value-min1,0] = normalize(nii_idata_FLAIR[:,:,slice])
                img[min0:max_value-min0,min1:max_value-min1,1] = normalize(nii_idata_T1[:,:,slice])
                
                normed_mask = np.zeros((max_value, max_value))
                normed_mask[min0:max_value-min0,min1:max_value-min1] =  normalize(nii_mdata[:,:,slice])
                
                img = Image.fromarray(normalize(img[:,:,0]))
                img.save(join(output_train, 'images', medc+'_'+patient+'_'+str(slice)+'.png'), format='PNG')
                normed_mask[normed_mask<255] = 0
                
                seg = Image.fromarray(normed_mask).convert('L')
                seg.putpalette(np.array(palette, dtype=np.uint8))
                seg.save(join(output_train, 'masks', medc+'_'+patient+'_'+str(slice)+'.png'), format='PNG')
                
                split_train[str(join(output_train, 'images', medc+'_'+patient+'_'+str(slice)+'.npy'))] = str(join(output_train, 'masks', medc+'_'+patient+'_'+str(slice)+'.npy'),)
                
    split_test = dict()            
                
    for i in range(len(test_masks)): 
        patient = test_scans_FLAIR[i].split('/')[-3]
        medc = test_scans_FLAIR[i].split('/')[-4]
        
        nii_img_FLAIR  = nib.load(test_scans_FLAIR[i])
        nii_idata_FLAIR = nii_img_FLAIR.get_fdata()  
        
        nii_img_T1  = nib.load(test_scans_T1[i])
        nii_idata_T1 = nii_img_T1.get_fdata()  
        
        nii_msk  = nib.load(test_masks[i])
        nii_mdata = nii_msk.get_fdata()
        
        bottom_limit = nii_mdata.shape[2]*SLICE_THRSHL
        upper_limit = nii_mdata.shape[2] - bottom_limit
        
        mask_npy = dict()
        for slice in range(nii_mdata.shape[2]):
            if slice>bottom_limit and slice<upper_limit:
                max_value = max(nii_msk.shape[0], nii_msk.shape[1])
                img = np.zeros((max_value, max_value, 2))
                
                img[:,:,0] = np.min(normalize(nii_idata_FLAIR[:,:,slice]))
                img[:,:,1] = np.min(normalize(nii_idata_T1[:,:,slice]))

                min0 = (max_value-nii_idata_FLAIR[:,:,slice].shape[0])//2
                min1 = (max_value-nii_idata_FLAIR[:,:,slice].shape[1])//2

                img[min0:max_value-min0,min1:max_value-min1,0] = normalize(nii_idata_FLAIR[:,:,slice])
                img[min0:max_value-min0,min1:max_value-min1,1] = normalize(nii_idata_T1[:,:,slice])
                
                normed_mask = np.zeros((max_value, max_value))
                normed_mask[min0:max_value-min0,min1:max_value-min1] =  normalize(nii_mdata[:,:,slice])
                img = Image.fromarray(normalize(img[:,:,0]))
                img.save(join(output_test, 'images', medc+'_'+patient+'_'+str(slice)+'.png'), format='PNG')
                normed_mask[normed_mask<255] = 0
                mask_npy[slice] = normed_mask
                seg = Image.fromarray(normed_mask).convert('L')
                seg.putpalette(np.array(palette, dtype=np.uint8))
                seg.save(join(output_test, 'masks', medc+'_'+patient+'_'+str(slice)+'.png'), format='PNG')
                
                split_test[str(join(output_test, 'images', medc+'_'+patient+'_'+str(slice)+'.npy'))] = str(join(output_test, 'masks', medc+'_'+patient+'_'+str(slice)+'.npy'))
                
        array_mask = np.zeros((list(mask_npy.values())[0].shape[0], list(mask_npy.values())[0].shape[1], nii_mdata.shape[2]))        
        for elem in mask_npy:
            array_mask[:,:,int(elem)] = mask_npy[elem]
        np.save(join(output_check, medc+'_'+patient), array_mask) 
    return split_train, split_test

if not os.path.exists(output_train):
    os.makedirs(join(output_train, 'images'))
    os.makedirs(join(output_train, 'masks'))
if not os.path.exists(output_test):
    os.makedirs(join(output_test, 'images'))
    os.makedirs(join(output_test, 'masks'))
if not os.path.exists(output_check):
    os.makedirs(output_check)

def get_paths(path):
    folders = [join(path, elem, 'pre') for elem in os.listdir(path)]
    scans_T1 = [join(folder, file_) for folder in folders for file_ in os.listdir(folder) if not '3D' in file_ and 'T1' in file_]
    scans_FLAIR = [join(folder, file_) for folder in folders for file_ in os.listdir(folder) if 'FLAIR' in file_]
    masks = [join(folder, file_) for folder in [join(path, elem) for elem in os.listdir(path)] for file_ in os.listdir(folder) if file_.endswith('gz')] 
    return scans_FLAIR, scans_T1, masks

def normalize(I):
    I8 = (((I - I.min()) / (I.max() - I.min())) * 255).astype(np.uint8)
    return I8

s_scans_FLAIR, s_scans_T1, s_masks = get_paths(path = '/media/data_4T/bran/WMH_dataset/raw/Singapore/')
u_scans_FLAIR, u_scans_T1, u_masks = get_paths(path = '/media/data_4T/bran/WMH_dataset/raw/Utrecht/')
g_scans_FLAIR, g_scans_T1, g_masks = get_paths(path = '/media/data_4T/bran/WMH_dataset/raw/GE3T/')

s_split_train, s_split_test = generate_images(scans_T1=s_scans_T1, scans_FLAIR=s_scans_FLAIR, masks=s_masks)
u_split_train, u_split_test = generate_images(scans_T1=u_scans_T1, scans_FLAIR=u_scans_FLAIR, masks=u_masks)
g_split_train, g_split_test = generate_images(scans_T1=g_scans_T1, scans_FLAIR=g_scans_FLAIR, masks=g_masks)

merged_split_train = {**s_split_train, **u_split_train}
merged_split_train = {**merged_split_train, **g_split_train}

merged_split_test = {**s_split_test, **u_split_train}
merged_split_test = {**merged_split_test, **g_split_test}

with open('/media/data_4T/bran/WMH_dataset/raw/training/patient_specific_flair_png2/split_train.tsv', 'w') as f:
    for key in merged_split_train.keys():
        f.write("%s\t%s\n"%(key,merged_split_train[key]))     
with open('/media/data_4T/bran/WMH_dataset/raw/training/patient_specific_flair_png2/split_test.tsv', 'w') as f:
    for key in merged_split_test.keys():
        f.write("%s\t%s\n"%(key,merged_split_test[key]))          