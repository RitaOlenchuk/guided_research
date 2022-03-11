import torch
import mmseg
import mmcv
import os
from mmcv import Config
import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
from PIL import Image
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor
from mmseg.apis import set_random_seed

# convert dataset annotation to semantic segmentation map
print(os.getcwd())
data_root = '/media/data_4T/bran/WMH_dataset/pre_processed/'
img_obj = 'images_three_datasets_sorted.npy'
ann_obj = 'masks_three_datasets_sorted.npy'
img_dir = 'images'
ann_dir = 'labels'
# define class and plaette for better visualization
classes = ('wmh', 'non_wmh')
palette = [[128, 128, 128], [241, 134, 51]]

preprocessed_data = np.load(osp.join(data_root, img_obj))
masks = np.load(osp.join(data_root, ann_obj))

def normalize8(I):
  mn = I.min()
  mx = I.max()

  mx -= mn

  I = ((I - mn)/mx) * 255
  return I.astype(np.uint8)


for i in range(preprocessed_data.shape[0]):
    normed_flair = normalize8(preprocessed_data[i,:,:,0])
    flair_img = Image.fromarray(normed_flair)
    flair_img.save(osp.join(data_root, img_dir, str(i)+'_flair.png'), format='PNG')
    
    normed_t1 = normalize8(preprocessed_data[i,:,:,1])
    t1_img = Image.fromarray(normed_t1)
    t1_img.save(osp.join(data_root, img_dir, str(i)+'_t1.png'), format='PNG')
    
    seg_img = Image.fromarray(masks[i,:,:,0]).convert('P')
    seg_img.putpalette(np.array(palette, dtype=np.uint8))
    seg_img.save(osp.join(data_root, ann_dir, str(i)+'_flair.png'), format='PNG')
    seg_img.save(osp.join(data_root, ann_dir, str(i)+'_t1.png'), format='PNG')

# split train/val set randomly
split_dir = 'splits'
mmcv.mkdir_or_exist(osp.join(data_root, split_dir))
filename_list = [osp.splitext(filename)[0] for filename in mmcv.scandir(
    osp.join(data_root, ann_dir), suffix='.png')]
with open(osp.join(data_root, split_dir, 'train.txt'), 'w') as f:
  # select first 3/5 as train set
  train_length = int(len(filename_list)*3/5)
  f.writelines(line + '\n' for line in filename_list[:train_length])
with open(osp.join(data_root, split_dir, 'val.txt'), 'w') as f:
  # select next 1/5 as val set
  val_length = int(len(filename_list)*4/5)
  f.writelines(line + '\n' for line in filename_list[train_length:val_length])
with open(osp.join(data_root, split_dir, 'test.txt'), 'w') as f:
  # select last 1/5 as test set
  f.writelines(line + '\n' for line in filename_list[val_length:])
  

@DATASETS.register_module()
class MICCAI_WMH_Dataset(CustomDataset):
  CLASSES = classes
  PALETTE = palette
  def __init__(self, split, **kwargs):
    super().__init__(img_suffix='.png', seg_map_suffix='.png', 
                     split=split, **kwargs)
    assert osp.exists(self.img_dir) and self.split is not None

   
cfg = Config.fromfile('/home/margaryta/my_training/MICCAI-WMH-2017/config/config.py')

# Since we use ony one GPU, BN is used instead of SyncBN
cfg.norm_cfg = dict(type='BN', requires_grad=True)
cfg.model.backbone.norm_cfg = cfg.norm_cfg
cfg.model.decode_head.norm_cfg = cfg.norm_cfg
cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg
# modify num classes of the model in decode/auxiliary head
cfg.model.decode_head.num_classes = 8
cfg.model.auxiliary_head.num_classes = 8

# Modify dataset type and path
cfg.dataset_type = 'MICCAI_WMH_Dataset'
cfg.data_root = data_root

cfg.data.samples_per_gpu = 8
cfg.data.workers_per_gpu=8

cfg.img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
cfg.crop_size = (256, 256)
cfg.train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(320, 240), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=cfg.crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **cfg.img_norm_cfg),
    dict(type='Pad', size=cfg.crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

cfg.test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(320, 240),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **cfg.img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


cfg.data.train.type = cfg.dataset_type
cfg.data.train.data_root = cfg.data_root
cfg.data.train.img_dir = img_dir
cfg.data.train.ann_dir = ann_dir
cfg.data.train.pipeline = cfg.train_pipeline
cfg.data.train.split = 'splits/train.txt'

cfg.data.val.type = cfg.dataset_type
cfg.data.val.data_root = cfg.data_root
cfg.data.val.img_dir = img_dir
cfg.data.val.ann_dir = ann_dir
cfg.data.val.pipeline = cfg.test_pipeline
cfg.data.val.split = 'splits/val.txt'

cfg.data.test.type = cfg.dataset_type
cfg.data.test.data_root = cfg.data_root
cfg.data.test.img_dir = img_dir
cfg.data.test.ann_dir = ann_dir
cfg.data.test.pipeline = cfg.test_pipeline
cfg.data.test.split = 'splits/val.txt'

# We can still use the pre-trained Mask RCNN model though we do not need to
# use the mask branch
#cfg.load_from = 'checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

# Set up working dir to save files and logs.
cfg.work_dir = './work_dirs/psp_r50_dice_40k_MICCAI_WMH/'
os.mkdir(cfg.work_dir)
#cfg.checkpoint_config.meta = dict(CLASSES=class, PALETTE=palette)

cfg.total_iters = 150
cfg.log_config.interval = 10
cfg.evaluation.interval = 200
cfg.checkpoint_config.interval = 10000

# Set seed to facitate reproducing the result
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)
cfg.checkpoint_config.meta = dict(CLASSES=classes, PALETTE=palette)
cfg.dump(cfg.work_dir+'test_config.py')

# Let's have a look at the final config used for training
print(f'Config:\n{cfg.pretty_text}')


# Build the dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_segmentor(
    cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_segmentor(model, datasets, cfg, distributed=False, validate=True, meta=dict())
#[1] 26157