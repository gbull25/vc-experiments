import os.path as osp
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc

import pickle
import os
import hashlib

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from timesformer_pytorch import TimeSformer

import random
import threading
import glob

import numpy as np
import wandb
from torch.utils.data import DataLoader
import os
import random
import cv2
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
import segmentation_models_pytorch as smp
import numpy as np
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
from warmup_scheduler import GradualWarmupScheduler
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 933120000

import psutil


class CFG:
    # ============== comp exp name =============
    comp_name = 'vesuvius'
    comp_dir_path = './'
    comp_folder_name = './'
    comp_dataset_path = f'./'
    exp_name = 'pretraining_all'
    # ============== model cfg =============
    in_chans = 26 # 
    # ============== training cfg =============
    size = 64
    tile_size = 256
    stride = tile_size // 8
    train_batch_size = 400 # 32
    valid_batch_size = train_batch_size

    scheduler = 'GradualWarmupSchedulerV2'
    epochs = 30 # 30
    warmup_factor = 10
    lr = 3e-5
    # ============== fold =============
    valid_id = None
    # ============== fixed =============

    min_lr = 1e-6
    weight_decay = 1e-6
    max_grad_norm = 100

    num_workers = 4 #16

    seed = 0

    # ============== set dataset path =============
    print('set dataset path')

    outputs_path = f'./outputs/{comp_name}/{exp_name}/'
    model_dir = outputs_path + \
        f'{comp_name}-models/'
    # ============== augmentation =============
    train_aug_list = [
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.75),
        A.ShiftScaleRotate(rotate_limit=360,shift_limit=0.15,scale_limit=0.15,p=0.75),
        A.OneOf([
                A.GaussNoise(var_limit=[10, 50]),
                A.GaussianBlur(),
                A.MotionBlur(),
                ], p=0.4),
        A.CoarseDropout(max_holes=2, max_width=int(size * 0.2), max_height=int(size * 0.2), 
                        mask_fill_value=0, p=0.5),
        A.Normalize(
            mean= [0] * in_chans,
            std= [1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]

    valid_aug_list = [
        A.Resize(size, size),
        A.Normalize(
            mean= [0] * in_chans,
            std= [1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]
    rotate = A.Compose([A.Rotate(5,p=1)])


def monitor_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / (1024 ** 2):.2f} MB")

def set_seed(seed=None, cudnn_deterministic=True):
    if seed is None:
        seed = 42
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = False

def make_dirs(cfg):
    for dir in [cfg.model_dir]:
        os.makedirs(dir, exist_ok=True)

def cfg_init(cfg, mode='train'):
    set_seed(cfg.seed)
    if mode == 'train':
        make_dirs(cfg)

cfg_init(CFG)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_image_mask(fragment_id,start_idx=17,end_idx=43, CFG=CFG):
    fragment_id_ = fragment_id.split("_")[0]
    images = []
    idxs = range(start_idx, end_idx)

    for i in idxs:
        if os.path.exists(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/layers/{i:02}.tif"):
            image = cv2.imread(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/layers/{i:02}.tif", 0)
        else:
            image = cv2.imread(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/layers/{i:02}.jpg", 0)
        pad0 = (CFG.tile_size - image.shape[0] % CFG.tile_size)
        pad1 = (CFG.tile_size - image.shape[1] % CFG.tile_size)
        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)        
        image=np.clip(image,0,200)
        images.append(image)
    images = np.stack(images, axis=2)
    if any(id_ in fragment_id_ for id_ in ['20230701020044','verso','20230901184804','20230901234823','20230531193658','20231007101615','20231005123333','20231011144857','20230522215721', '20230919113918', '20230625171244','20231022170900','20231012173610','20231016151000']):
        images=images[:,:,::-1]
    # Get the list of files that match the pattern
    inklabel_files = glob.glob(f"train_scrolls/{fragment_id}/*inklabels.*")
    if len(inklabel_files) > 0:
        mask = cv2.imread( inklabel_files[0], 0)
    else:
        print(f"Creating empty mask for {fragment_id}")
        mask = np.zeroes(images[0].shape)
    fragment_mask=cv2.imread(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/{fragment_id_}_mask.png", 0)
    fragment_mask = np.pad(fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)
    mask = mask.astype('float32')
    mask/=255
    return images, mask,fragment_mask

def worker_function(fragment_id, CFG):
    train_images = []
    train_masks = []
    valid_images = []
    valid_masks = []
    valid_xyxys = []

    if not os.path.exists(f"train_scrolls/{fragment_id}"):
        fragment_id = fragment_id + "_superseded"
    print('reading ',fragment_id)
    try:
        image, mask, fragment_mask = read_image_mask(fragment_id, CFG=CFG)
    except:
        print("aborted reading fragment", fragment_id)
        return None
    x1_list = list(range(0, image.shape[1]-CFG.tile_size+1, CFG.stride))
    y1_list = list(range(0, image.shape[0]-CFG.tile_size+1, CFG.stride))
    windows_dict={}

    for a in y1_list:
        for b in x1_list:
            if not np.any(fragment_mask[a:a + CFG.tile_size, b:b + CFG.tile_size]==0):
                if (fragment_id==CFG.valid_id) or (not np.all(mask[a:a + CFG.tile_size, b:b + CFG.tile_size]<0.05)):
                    for yi in range(0,CFG.tile_size,CFG.size):
                        for xi in range(0,CFG.tile_size,CFG.size):
                            y1=a+yi
                            x1=b+xi
                            y2=y1+CFG.size
                            x2=x1+CFG.size
                            if fragment_id!=CFG.valid_id:
                                train_images.append(image[y1:y2, x1:x2])
                                train_masks.append(mask[y1:y2, x1:x2, None])
                                assert image[y1:y2, x1:x2].shape==(CFG.size,CFG.size,CFG.in_chans)
                            if fragment_id==CFG.valid_id:
                                if (y1,y2,x1,x2) not in windows_dict:
                                    valid_images.append(image[y1:y2, x1:x2])
                                    valid_masks.append(mask[y1:y2, x1:x2, None])
                                    valid_xyxys.append([x1, y1, x2, y2])
                                    assert image[y1:y2, x1:x2].shape==(CFG.size,CFG.size,CFG.in_chans)
                                    windows_dict[(y1,y2,x1,x2)]='1'

    print("finished reading fragment", fragment_id)

    return train_images, train_masks, valid_images, valid_masks, valid_xyxys

def get_train_valid_dataset(fragment_ids=[
                         # scroll 1
                        "20231022170901",
                        "20231210121321",
                        "20231221180251",
                        "20230530164535",
                        "20231007101615",
                        "20230530172803",
                        "20230522215721",
                        "20230620230617",
                        "20230902141231",
                        "20231016151000",
                        "20230520175435",
                        "20230531121653",
                        "20230826170124",
                        "20230901184804",
                        "20230813_real_1",
                        "20230531193658",
                        "20230904020426",
                        "20230601193301",
                        "20231012184420",
                        "20230903193206",
                        "20230929220924",
                        "20230820203112",
                        "20231005123333",
                        "20230620230619",
                        "20230530212931",
                        "20231012173610",
                        "20230702185753",
                        "20231106155350",
                        "20231031143850",
                        "20231012184421",
                        "20231012184423",
                        "20231106155351",
                        "20231031143852",
                        "20231005123336",
                        "20230929220926",
                        "verso",
                        "recto",
                        # added 01.04.25
                        '20230522181603',
                        '20230611014200',
                        '20230701020044',
                        '20230827161847',
                        '20230904135535',
                        '20230905134255',
                        '20230909121925',
                        '20231001164029',
                        '20231004222109',
                        '20231012085431',
                        '20231022170900_superseded',
                        # были в пятом свитке почему-то
                        "20231007101619",
                        "20231016151002",
                        "20240218140920",
                        "20240222111510",
                        "20240301161650",
                        "20240227085920",
                        "20240221073650",
                        # added 06.04
                        "20240223130140",
                        # fragments
                        ###
                        #"frag1_surface_volume",
                        #"frag2_surface_volume",
                        #"frag3_surface_volume",
                        #"frag4_surface_volume",
                        #"frag5_surface_volume",
                        #"frag6_surface_volume",
                        ###
                        # Rescaled
                        #"Frag1",
                        #"Frag2",
                        #"Frag3",
                        #"Frag4",
                        #"Frag5",
                        # scroll5
                        "20241025145341",
                        "20241025145701",
                        "20241025150211",
                        "20241028121111",
                        "20241030152031",
                        "20241102125650",
                        "20241102160330",
                        "20241108111522",
                        "20241108115232",
                        "20241108120732",
                        "20241113070770",
                        "20241113080880",
                        "20241113090990",]):
    threads = []
    results = [None] * len(fragment_ids)

    # Function to run in each thread
    def thread_target(idx, fragment_id):
        results[idx] = worker_function(fragment_id, CFG)

    # Create and start threads
    for idx, fragment_id in enumerate(fragment_ids):
        thread = threading.Thread(target=thread_target, args=(idx, fragment_id))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    train_images = []
    train_masks = []
    valid_images = []
    valid_masks = []
    valid_xyxys = []
    print("Aggregating results")
    for r in results:
        if r is None:
            continue
        train_images += r[0]
        train_masks += r[1]
        valid_images += r[2]
        valid_masks += r[3]
        valid_xyxys += r[4]

    return train_images, train_masks, valid_images, valid_masks, valid_xyxys

def get_transforms(data, cfg):
    if data == 'train':
        aug = A.Compose(cfg.train_aug_list)
    elif data == 'valid':
        aug = A.Compose(cfg.valid_aug_list)
    return aug


class StreamingFragmentDataset(Dataset):
    def __init__(self, fragment_ids, cfg, transform=None, is_train=True, use_cached_indices=True):
        self.fragment_ids = fragment_ids
        self.cfg = cfg
        self.transform = transform
        self.is_train = is_train
        self.cache = {}
        self.max_cache_size = 3
        self.sample_indices = []
        self.fragment_index = []
        self.rotate = CFG.rotate
        
        # Создаем уникальный идентификатор для набора фрагментов
        self.dataset_hash = self._get_dataset_hash()
        self.indices_cache_path = f"./dataset_indices_{self.dataset_hash}_{is_train}.pkl"
        
        # Загружаем индексы или строим новые
        if use_cached_indices and os.path.exists(self.indices_cache_path):
            self._load_indices()
        else:
            self.build_indices()
            if use_cached_indices:
                self._save_indices()
    
    def _get_dataset_hash(self):
        """Создаем уникальный хэш для набора фрагментов и конфигурации"""
        # Включаем важные параметры в хэш
        hash_input = "_".join([
            ",".join(sorted(self.fragment_ids)),
            str(self.cfg.tile_size),
            str(self.cfg.stride),
            str(self.cfg.size),
            str(self.cfg.valid_id),
            "train" if self.is_train else "valid"
        ])
        return hashlib.md5(hash_input.encode()).hexdigest()[:10]
    
    def _save_indices(self):
        """Сохраняем индексы в файл"""
        data_to_save = {
            'sample_indices': self.sample_indices,
            'fragment_ids': self.fragment_ids,
            'cfg_params': {
                'tile_size': self.cfg.tile_size,
                'stride': self.cfg.stride,
                'size': self.cfg.size,
                'valid_id': self.cfg.valid_id
            }
        }
        
        os.makedirs(os.path.dirname(self.indices_cache_path), exist_ok=True)
        with open(self.indices_cache_path, 'wb') as f:
            pickle.dump(data_to_save, f)
        print(f"Индексы сохранены в {self.indices_cache_path}")
    
    def _load_indices(self):
        """Загружаем индексы из файла"""
        with open(self.indices_cache_path, 'rb') as f:
            data = pickle.load(f)
        
        # Проверяем соответствие конфигурации
        cfg_params = data['cfg_params']
        if (cfg_params['tile_size'] != self.cfg.tile_size or
            cfg_params['stride'] != self.cfg.stride or
            cfg_params['size'] != self.cfg.size or
            cfg_params['valid_id'] != self.cfg.valid_id):
            print("Предупреждение: параметры кэшированных индексов не соответствуют текущей конфигурации")
            print("Перестраиваем индексы...")
            self.build_indices()
            self._save_indices()
            return
        
        self.sample_indices = data['sample_indices']
        print(f"Загружено {len(self.sample_indices)} индексов из {self.indices_cache_path}")
        
        
    def build_indices(self):
        print("Building dataset indices...")
        self.sample_indices = []
        self.fragment_index = []
        
        for fragment_idx, fragment_id in enumerate(self.fragment_ids):
            if not os.path.exists(f"train_scrolls/{fragment_id}"):
                fragment_id = fragment_id + "_superseded"
                if not os.path.exists(f"train_scrolls/{fragment_id}"):
                    continue
                    
            try:
                # Just get the image dimensions and mask, without storing the full data
                image, mask, fragment_mask = self._load_fragment(fragment_id)
                
                # Calculate indices without actually storing all the images
                indices = self._calculate_indices(fragment_id, fragment_idx, image, mask, fragment_mask)
                
                self.sample_indices.extend(indices)
                
                # Clean up to save memory
                del image, mask, fragment_mask
                gc.collect()
                
            except Exception as e:
                print(f"Error processing fragment {fragment_id}: {e}")
                continue
                
        print(f"Built index with {len(self.sample_indices)} samples from {len(self.fragment_ids)} fragments")
        
    def _calculate_indices(self, fragment_id, fragment_idx, image, mask, fragment_mask):
        indices = []
        x1_list = list(range(0, image.shape[1]-self.cfg.tile_size+1, self.cfg.stride))
        y1_list = list(range(0, image.shape[0]-self.cfg.tile_size+1, self.cfg.stride))
        
        for a in y1_list:
            for b in x1_list:
                if not np.any(fragment_mask[a:a + self.cfg.tile_size, b:b + self.cfg.tile_size]==0):
                    is_valid_fragment = (fragment_id==self.cfg.valid_id)
                    has_ink = not np.all(mask[a:a + self.cfg.tile_size, b:b + self.cfg.tile_size]<0.05)
                    
                    if (is_valid_fragment and self.is_train is False) or (not is_valid_fragment and has_ink and self.is_train):
                        for yi in range(0,self.cfg.tile_size,self.cfg.size):
                            for xi in range(0,self.cfg.tile_size,self.cfg.size):
                                y1=a+yi
                                x1=b+xi
                                y2=y1+self.cfg.size
                                x2=x1+self.cfg.size
                                
                                # Store just the coordinates and fragment information
                                indices.append((fragment_idx, y1, y2, x1, x2))
        
        return indices
        
    def _load_fragment(self, fragment_id):
        """Load a fragment only when needed"""
        if fragment_id in self.cache:
            return self.cache[fragment_id]
        
        # If cache is full, remove least recently used item
        if len(self.cache) >= self.max_cache_size:
            # Remove first item (least recently used)
            self.cache.pop(next(iter(self.cache)))
            gc.collect()  # Force garbage collection
        
        # Load the fragment
        image, mask, fragment_mask = read_image_mask(fragment_id, CFG=self.cfg)
        
        # Add to cache
        self.cache[fragment_id] = (image, mask, fragment_mask)
        return image, mask, fragment_mask
            
    def __len__(self):
        return len(self.sample_indices)
    
    def fourth_augment(self, image):
        image_tmp = np.zeros_like(image)
        cropping_num = random.randint(18, 26)

        start_idx = random.randint(0, self.cfg.in_chans - cropping_num)
        crop_indices = np.arange(start_idx, start_idx + cropping_num)

        start_paste_idx = random.randint(0, self.cfg.in_chans - cropping_num)

        tmp = np.arange(start_paste_idx, cropping_num)
        np.random.shuffle(tmp)

        cutout_idx = random.randint(0, 2)
        temporal_random_cutout_idx = tmp[:cutout_idx]

        image_tmp[..., start_paste_idx : start_paste_idx + cropping_num] = image[..., crop_indices]

        if random.random() > 0.4:
            image_tmp[..., temporal_random_cutout_idx] = 0
        image = image_tmp
        return image

    def __getitem__(self, idx):
        fragment_idx, y1, y2, x1, x2 = self.sample_indices[idx]
        fragment_id = self.fragment_ids[fragment_idx]
        
        if not os.path.exists(f"train_scrolls/{fragment_id}"):
            fragment_id = fragment_id + "_superseded"
        
        image, mask, _ = self._load_fragment(fragment_id)
        
        # Extract the specific patch
        image_patch = image[y1:y2, x1:x2].copy()  # Create a copy to avoid memory leaks
        mask_patch = mask[y1:y2, x1:x2, None].copy()
        
        if self.is_train:
            # Apply augmentations for training
            image_patch = image_patch.transpose(2,1,0)  # (c,w,h)
            image_patch = self.rotate(image=image_patch)['image']
            image_patch = image_patch.transpose(0,2,1)  # (c,h,w)
            image_patch = self.rotate(image=image_patch)['image']
            image_patch = image_patch.transpose(0,2,1)  # (c,w,h)
            image_patch = image_patch.transpose(2,1,0)  # (h,w,c)
            
            image_patch = self.fourth_augment(image_patch)
        
        if self.transform:
            data = self.transform(image=image_patch, mask=mask_patch)
            image_patch = data['image'].unsqueeze(0)
            mask_patch = data['mask']
            mask_patch = F.interpolate(mask_patch.unsqueeze(0), (self.cfg.size//16, self.cfg.size//16)).squeeze(0)
        
        if not self.is_train:
            return image_patch, mask_patch, np.array([x1, y1, x2, y2])
        else:
            return image_patch, mask_patch

class CustomDataset(Dataset):
    def __init__(self, images ,cfg,xyxys=None, labels=None, transform=None):
        self.images = images
        self.cfg = cfg
        self.labels = labels
        
        self.transform = transform
        self.xyxys=xyxys
        self.rotate=CFG.rotate
    def __len__(self):
        return len(self.images)
    def fourth_augment(self,image):
        image_tmp = np.zeros_like(image)
        cropping_num = random.randint(18, 26)

        start_idx = random.randint(0, self.cfg.in_chans - cropping_num)
        crop_indices = np.arange(start_idx, start_idx + cropping_num)

        start_paste_idx = random.randint(0, self.cfg.in_chans - cropping_num)

        tmp = np.arange(start_paste_idx, cropping_num)
        np.random.shuffle(tmp)

        cutout_idx = random.randint(0, 2)
        temporal_random_cutout_idx = tmp[:cutout_idx]

        image_tmp[..., start_paste_idx : start_paste_idx + cropping_num] = image[..., crop_indices]

        if random.random() > 0.4:
            image_tmp[..., temporal_random_cutout_idx] = 0
        image = image_tmp
        return image

    def __getitem__(self, idx):
        if self.xyxys is not None:
            image = self.images[idx]
            label = self.labels[idx]
            xy=self.xyxys[idx]
            if self.transform:
                data = self.transform(image=image, mask=label)
                image = data['image'].unsqueeze(0)
                label = data['mask']
                label=F.interpolate(label.unsqueeze(0),(self.cfg.size//16,self.cfg.size//16)).squeeze(0)
            return image, label,xy
        else:
            image = self.images[idx]
            label = self.labels[idx]
            #3d rotate
            image=image.transpose(2,1,0)#(c,w,h)
            image=self.rotate(image=image)['image']
            image=image.transpose(0,2,1)#(c,h,w)
            image=self.rotate(image=image)['image']
            image=image.transpose(0,2,1)#(c,w,h)
            image=image.transpose(2,1,0)#(h,w,c)

            image=self.fourth_augment(image)
            
            if self.transform:
                data = self.transform(image=image, mask=label)
                image = data['image'].unsqueeze(0)
                label = data['mask']
                label=F.interpolate(label.unsqueeze(0),(self.cfg.size//16,self.cfg.size//16)).squeeze(0)
            return image, label
        

class RegressionPLModel(pl.LightningModule):
    def __init__(self,pred_shape,size=256,with_norm=False):
        super(RegressionPLModel, self).__init__()

        self.save_hyperparameters()
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)

        self.loss_func1 = smp.losses.DiceLoss(mode='binary')
        self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25)
        self.loss_func= lambda x,y:0.5 * self.loss_func1(x,y)+0.5*self.loss_func2(x,y)

        self.backbone=TimeSformer(
                dim = 512,
                image_size = 64,
                patch_size = 8, #16,
                num_frames = 26,
                num_classes = 16,
                channels=1,
                depth = 8,
                heads = 6,
                dim_head =  64,
                attn_dropout = 0.1,
                ff_dropout = 0.1
            )
        if self.hparams.with_norm:
            self.normalization=nn.BatchNorm3d(num_features=1)
    def forward(self, x):
        if x.ndim==4:
            x=x[:,None]
        if self.hparams.with_norm:
            x=self.normalization(x)
        x = self.backbone(torch.permute(x, (0, 2, 1,3,4)))
        x=x.view(-1,1,4,4)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss1 = self.loss_func(outputs, y)
        if torch.isnan(loss1):
            print("Loss nan encountered")
        self.log("train/total_loss", loss1.item(),on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss1}

    def validation_step(self, batch, batch_idx):
        x,y,xyxys= batch
        batch_size = x.size(0)
        outputs = self(x)
        loss1 = self.loss_func(outputs, y)
        y_preds = torch.sigmoid(outputs).to('cpu')
        for i, (x1, y1, x2, y2) in enumerate(xyxys):
            self.mask_pred[y1:y2, x1:x2] += F.interpolate(y_preds[i].unsqueeze(0).float(),scale_factor=16,mode='bilinear').squeeze(0).squeeze(0).numpy()
            self.mask_count[y1:y2, x1:x2] += np.ones((self.hparams.size, self.hparams.size))

        self.log("val/total_loss", loss1.item(),on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss1}
    
    def on_validation_epoch_end(self):
        self.mask_pred = np.divide(self.mask_pred, self.mask_count, out=np.zeros_like(self.mask_pred), where=self.mask_count!=0)
        wandb_logger.log_image(key="masks", images=[np.clip(self.mask_pred,0,1)], caption=["probs"])

        #reset mask
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)
    def configure_optimizers(self):

        optimizer = AdamW(self.parameters(), lr=CFG.lr)
    
        scheduler = get_scheduler(CFG, optimizer)
        return [optimizer],[scheduler]



class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    """
    https://www.kaggle.com/code/underwearfitting/single-fold-training-of-resnet200d-lb0-965
    """
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(
            optimizer, multiplier, total_epoch, after_scheduler)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

def get_scheduler(cfg, optimizer):
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 10, eta_min=1e-6)
    scheduler = GradualWarmupSchedulerV2(
        optimizer, multiplier=1.0, total_epoch=1, after_scheduler=scheduler_cosine)

    return scheduler

def scheduler_step(scheduler, avg_val_loss, epoch):
    scheduler.step(epoch)

def get_streaming_dataloaders(fragment_ids, CFG, use_cached_indices=True):
    """Create streaming dataloaders that load data incrementally"""
    # Split ids for validation if needed
    train_fragment_ids = [id for id in fragment_ids if id != CFG.valid_id]
    valid_fragment_ids = [CFG.valid_id] if CFG.valid_id in fragment_ids else []
    
    # Create datasets with caching enabled
    train_dataset = StreamingFragmentDataset(
        train_fragment_ids, 
        CFG, 
        transform=get_transforms(data='train', cfg=CFG),
        is_train=True,
        use_cached_indices=use_cached_indices
    )
    
    valid_dataset = None
    if valid_fragment_ids:
        valid_dataset = StreamingFragmentDataset(
            valid_fragment_ids, 
            CFG, 
            transform=get_transforms(data='valid', cfg=CFG),
            is_train=False,
            use_cached_indices=use_cached_indices
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.train_batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=False,
        drop_last=True,
    )
    
    valid_loader = None
    if valid_dataset:
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=CFG.valid_batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=False,
            drop_last=True
        )
    
    # Get validation mask for visualization
    pred_shape = None
    if valid_fragment_ids:
        try:
            fragment_id = valid_fragment_ids[0]
            valid_mask_gt = cv2.imread(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/{fragment_id}_inklabels.png", 0)
            pred_shape = valid_mask_gt.shape
        except:
            print(f"Could not load validation mask for {valid_fragment_ids[0]}")
            pred_shape = (1000, 1000)  # Default size
    
    return train_loader, valid_loader, pred_shape


# Modify the training loop to use the new dataloaders
torch.set_float32_matmul_precision('medium')
# Add all of the validation segments into the array to run multiple validation folds
fragments = ['20231210121321']
for fid in fragments:
    CFG.valid_id = fid
    fragment_id = CFG.valid_id
    run_slug = f'alldata'

    # Use the new streaming dataloaders
    train_loader, valid_loader, pred_shape = get_streaming_dataloaders(
        [

                         # scroll 1
                        "20231022170901",
                        "20231210121321",
                        "20231221180251",
                        "20230530164535",
                        "20231007101615",
                        "20230530172803",
                        "20230522215721",
                        "20230620230617",
                        "20230902141231",
                        "20231016151000",
                        "20230520175435",
                        "20230531121653",
                        "20230826170124",
                        "20230901184804",
                        "20230813_real_1",
                        "20230531193658",
                        "20230904020426",
                        "20230601193301",
                        "20231012184420",
                        "20230903193206",
                        "20230929220924",
                        "20230820203112",
                        "20231005123333",
                        "20230620230619",
                        "20230530212931",
                        "20231012173610",
                        "20230702185753",
                        "20231106155350",
                        "20231031143850",
                        "20231012184421",
                        "20231012184423",
                        "20231106155351",
                        "20231031143852",
                        "20231005123336",
                        "20230929220926",
                        "verso",
                        "recto",
                        # added 01.04.25
                        '20230522181603',
                        '20230611014200',
                        '20230701020044',
                        '20230827161847',
                        '20230904135535',
                        '20230905134255',
                        '20230909121925',
                        '20231001164029',
                        '20231004222109',
                        '20231012085431',
                        '20231022170900_superseded',
                        # были в пятом свитке почему-то
                        "20231007101619",
                        "20231016151002",
                        "20240218140920",
                        "20240222111510",
                        "20240301161650",
                        "20240227085920",
                        "20240221073650",
                        # added 06.04
                        "20240223130140",
                        # fragments
                        ###
                        #"frag1_surface_volume",
                        #"frag2_surface_volume",
                        #"frag3_surface_volume",
                        #"frag4_surface_volume",
                        #"frag5_surface_volume",
                        #"frag6_surface_volume",
                        ###
                        # Rescaled
                        "Frag1",
                        "Frag2",
                        "Frag3",
                        "Frag4",
                        "Frag5",
                        # scroll5
                        "20241025145341",
                        "20241025145701",
                        "20241025150211",
                        "20241028121111",
                        "20241030152031",
                        "20241102125650",
                        "20241102160330",
                        "20241108111522",
                        "20241108115232",
                        "20241108120732",
                        "20241113070770",
                        "20241113080880",
                        "20241113090990",
        ], 
        CFG
    )
    
    # Memory cleanup
    gc.collect()
    monitor_memory_usage()

    # Continue with your training as before
    wandb_logger = WandbLogger(project="vesivus", name=run_slug+f'try-og')
    model = RegressionPLModel(pred_shape=pred_shape, size=CFG.size)
    wandb_logger.watch(model, log="all", log_freq=100)
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="gpu",
        devices=-1,  # [0],
        logger=wandb_logger,
        default_root_dir="./models",
        accumulate_grad_batches=1,
        precision='16-mixed',
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        strategy='ddp',
        callbacks=[
            ModelCheckpoint(
                filename=f'timesformer_wild16_{fid}_fr'+'{epoch}',
                dirpath=CFG.model_dir,
                monitor='train/total_loss',
                mode='min',
                save_top_k=CFG.epochs
            ),
        ],
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    wandb.finish()
