import os
from PIL import Image
from os.path import join
import numpy as np
import nibabel as nib
#from numpngw import write_png
from PIL import Image

output_train = '/media/data_4T/bran/WMH_dataset/raw/training/patient_specific_flair_png/train'
output_test = '/media/data_4T/bran/WMH_dataset/raw/training/patient_specific_flair_png/val'
output_look = '/media/data_4T/bran/WMH_dataset/raw/training/patient_specific_flair_png/check'
SLICE_THRSHL = 0.16
CHECK = False
palette = [[0, 0, 0], [255, 255, 255]]

if not os.path.exists(output_train):
    os.makedirs(join(output_train, 'images'))
    os.makedirs(join(output_train, 'masks'))
if not os.path.exists(output_test):
    os.makedirs(join(output_test, 'images'))
    os.makedirs(join(output_test, 'masks'))
if not os.path.exists(output_look):
    os.makedirs(join(output_look, 'images'))
    os.makedirs(join(output_look, 'masks'))

def get_paths(path):
    folders = [join(path, elem, 'pre') for elem in os.listdir(path)]
    scans_T1 = [join(folder, file_) for folder in folders for file_ in os.listdir(folder) if not '3D' in file_ and 'T1' in file_]
    scans_FLAIR = [join(folder, file_) for folder in folders for file_ in os.listdir(folder) if 'FLAIR' in file_]
    masks = [join(folder, file_) for folder in [join(path, elem) for elem in os.listdir(path)] for file_ in os.listdir(folder) if file_.endswith('gz')] 
    return scans_FLAIR, scans_T1, masks

#(x, y, 2)  0-Flair 1-T1 Scan
#(x, y, 1) Mask

def normalize(I):
    I8 = (((I - I.min()) / (I.max() - I.min())) * 255).astype(np.uint8)
    return I8

s_scans_FLAIR, s_scans_T1, s_masks = get_paths(path = '/media/data_4T/bran/WMH_dataset/raw/Singapore/')
u_scans_FLAIR, u_scans_T1, u_masks = get_paths(path = '/media/data_4T/bran/WMH_dataset/raw/Utrecht/')
g_scans_FLAIR, g_scans_T1, g_masks = get_paths(path = '/media/data_4T/bran/WMH_dataset/raw/GE3T/')

all_patients = s_masks + u_masks + g_masks

train_patients_ids = np.random.choice(a=np.arange(len(all_patients)), size=int(len(all_patients)*(4/5)), replace=False)
test_pations_ids = list(set(range(len(all_patients))) - set(train_patients_ids))

train_masks =  np.array(all_patients)[train_patients_ids]
train_scans_T1 = [path for path in s_scans_T1+u_scans_T1+g_scans_T1 if any(path.startswith(patient_path[:-10]) for patient_path in train_masks)]
train_scans_FLAIR = [path for path in s_scans_FLAIR+u_scans_FLAIR+g_scans_FLAIR if any(path.startswith(patient_path[:-10]) for patient_path in train_masks)]

test_masks = np.array(all_patients)[test_pations_ids]
test_scans_T1 = [path for path in s_scans_T1+u_scans_T1+g_scans_T1 if not path in train_scans_T1]
test_scans_FLAIR = [path for path in s_scans_FLAIR+u_scans_FLAIR+g_scans_FLAIR if not path in train_scans_FLAIR]

train_masks.sort()
train_scans_T1.sort()
train_scans_FLAIR.sort()

test_masks.sort()
test_scans_T1.sort()
test_scans_FLAIR.sort()

split_dict = dict()

for i in range(len(train_masks)): 
    patient = train_scans_FLAIR[i].split('/')[-3]
    medc = train_scans_FLAIR[i].split('/')[-4]
    
    nii_img_FLAIR  = nib.load(train_scans_FLAIR[i])
    nii_idata_FLAIR = nii_img_FLAIR.get_fdata()  
    
    nii_img_T1  = nib.load(train_scans_T1[i])
    nii_idata_T1= nii_img_T1.get_fdata()  
    
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
            
            msk = np.zeros((max_value, max_value))
            msk[min0:max_value-min0,min1:max_value-min1] = normalize(nii_mdata[:,:,slice])
            
            if CHECK:
                t1_img = Image.fromarray(img[:,:,1]).convert('RGB')
                flair_img = Image.fromarray(img[:,:,0]).convert('RGB')
                mask_img = Image.fromarray(msk).convert('RGB')
                t1_img.save(join(output_look, 'images', medc+'_'+patient+'_t1_'+str(slice)+'.png'))
                flair_img.save(join(output_look, 'images', medc+'_'+patient+'_flair_'+str(slice)+'.png'))
                mask_img.save(join(output_look, 'masks', medc+'_'+patient+'_'+str(slice)+'.png'))
            
            img = Image.fromarray(normalize(img[:,:,0]))
            img.save(join(output_train, 'images', medc+'_'+patient+'_'+str(slice)+'.png'), format='PNG')
            #write_png(join(output_train, 'images', medc+'_'+patient+'_'+str(slice)+'.png'), normalize(img[:,:,0]))
            #np.save(join(output_train, 'images', medc+'_'+patient+'_'+str(slice)+'.npy'), normalize(img))
            normed_mask = normalize(msk)
            normed_mask[normed_mask<255] = 0
            seg = Image.fromarray(normed_mask).convert('L')
            seg.putpalette(np.array(palette, dtype=np.uint8))
            seg.save(join(output_train, 'masks', medc+'_'+patient+'_'+str(slice)+'.png'), format='PNG')
            #write_png(join(output_train, 'masks', medc+'_'+patient+'_'+str(slice)+'.png'), normed_mask)
            #np.save(join(output_train, 'masks', medc+'_'+patient+'_'+str(slice)+'.npy'), normed_mask)
            
            split_dict[str(join(output_train, 'images', medc+'_'+patient+'_'+str(slice)+'.npy'))] = str(join(output_train, 'masks', medc+'_'+patient+'_'+str(slice)+'.npy'),)

with open('/media/data_4T/bran/WMH_dataset/raw/training/patient_specific_flair_png/split_train.tsv', 'w') as f:
    for key in split_dict.keys():
        f.write("%s\t%s\n"%(key,split_dict[key]))            
            
split_dict = dict()            
            
for i in range(len(test_masks)): 
    patient = test_scans_FLAIR[i].split('/')[-3]
    medc = test_scans_FLAIR[i].split('/')[-4]
    
    nii_img_FLAIR  = nib.load(test_scans_FLAIR[i])
    nii_idata_FLAIR = nii_img_FLAIR.get_fdata()  
    
    nii_img_T1  = nib.load(test_scans_T1[i])
    nii_idata_T1= nii_img_T1.get_fdata()  
    
    nii_msk  = nib.load(test_masks[i])
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
            
            msk = np.zeros((max_value, max_value))
            msk[min0:max_value-min0,min1:max_value-min1] = normalize(nii_mdata[:,:,slice])
            
            if CHECK:
                t1_img = Image.fromarray(img[:,:,1]).convert('RGB')
                flair_img = Image.fromarray(img[:,:,0]).convert('RGB')
                mask_img = Image.fromarray(msk).convert('RGB')
                t1_img.save(join(output_look, 'images', medc+'_'+patient+'_t1_'+str(slice)+'.png'))
                flair_img.save(join(output_look, 'images', medc+'_'+patient+'_flair_'+str(slice)+'.png'))
                mask_img.save(join(output_look, 'masks', medc+'_'+patient+'_'+str(slice)+'.png'))
            
            img = Image.fromarray(normalize(img[:,:,0]))
            img.save(join(output_test, 'images', medc+'_'+patient+'_'+str(slice)+'.png'), format='PNG')
            #write_png(join(output_test, 'images', medc+'_'+patient+'_'+str(slice)+'.png'), normalize(img[:,:,0]))
            #np.save(join(output_test, 'images', medc+'_'+patient+'_'+str(slice)+'.npy'), normalize(img))
            normed_mask = normalize(msk)
            normed_mask[normed_mask<255] = 0
            seg = Image.fromarray(normed_mask).convert('L')
            seg.putpalette(np.array(palette, dtype=np.uint8))
            seg.save(join(output_test, 'masks', medc+'_'+patient+'_'+str(slice)+'.png'), format='PNG')
            #write_png(join(output_test, 'masks', medc+'_'+patient+'_'+str(slice)+'.png'), normed_mask)
            #np.save(join(output_test, 'masks', medc+'_'+patient+'_'+str(slice)+'.npy'), normed_mask)
            
            split_dict[str(join(output_test, 'images', medc+'_'+patient+'_'+str(slice)+'.npy'))] = str(join(output_test, 'masks', medc+'_'+patient+'_'+str(slice)+'.npy'))
            
with open('/media/data_4T/bran/WMH_dataset/raw/training/patient_specific_flair_png/split_test.tsv', 'w') as f:
    for key in split_dict.keys():
        f.write("%s\t%s\n"%(key,split_dict[key]))
