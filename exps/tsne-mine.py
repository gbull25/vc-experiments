import numpy as np
import os
import torch
import glob
from PIL import Image
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from sklearn.manifold import TSNE
import umap
import pandas as pd
from tqdm import tqdm
from torch.nn.functional import interpolate
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2**32)  # 4 гигапикселя
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 933120000
import cv2

CROP_SIZE = 600 # Size of the crop that we want to compute PCA on. The crop size will be rounded to the smallest multiple of the patch size
PATCH_SIZE = 14 # patch size that the DINO model uses, do not change!
CROP_SIZE = (CROP_SIZE // PATCH_SIZE) * PATCH_SIZE

model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
if torch.cuda.is_available():
    model = model.cuda()
model.eval()

def read_image_mask(fragment_id, scroll_path, start_idx=20, end_idx=36,):
    fragment_id_ = fragment_id.split("_")[0]
    images = []
    idxs = range(start_idx, end_idx)

    if fragment_id not in ['20240304161941-20240304144031-20240304141531-20231210132040', '20231030220150', '20231031231220',]:
        for i in idxs:
            if os.path.exists(f"/home/jovyan/Bulygin/villa/ink-detection/{scroll_path}/{fragment_id}/layers/{i:02}.tif"):
                image = cv2.imread(f"/home/jovyan/Bulygin/villa/ink-detection/{scroll_path}/{fragment_id}/layers/{i:02}.tif", 0)
            elif os.path.exists(f"/home/jovyan/Bulygin/villa/ink-detection/{scroll_path}/{fragment_id}/layers/{i:03}.tif"):
                image = cv2.imread(f"/home/jovyan/Bulygin/villa/ink-detection/{scroll_path}/{fragment_id}/layers/{i:03}.tif", 0)
            elif os.path.exists(f"/home/jovyan/Bulygin/villa/ink-detection/{scroll_path}/{fragment_id}/layers/{i:02}.jpg"):
                image = cv2.imread(f"/home/jovyan/Bulygin/villa/ink-detection/{scroll_path}/{fragment_id}/layers/{i:02}.jpg", 0)
            else:
                image = cv2.imread(f"/home/jovyan/Bulygin/villa/ink-detection/{scroll_path}/{fragment_id}/layers/{i:02}.png", 0)
            pad0 = (256 - image.shape[0] % 256)
            pad1 = (256 - image.shape[1] % 256)
            image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)        
            image=np.clip(image,0,200)
            images.append(image)
        images = np.stack(images, axis=2)
        # Get the list of files that match the pattern
        inklabel_files = glob.glob(f"/home/jovyan/Bulygin/villa/ink-detection/{scroll_path}/{fragment_id}/*inklabels.*")
        if len(inklabel_files) > 0:
            mask = cv2.imread( inklabel_files[0], 0)
            if mask is None or mask.shape[:2] != images.shape[:2]:
                mask = np.zeros(images.shape[:2], dtype=np.float32)
        else:
            print(f"Creating empty mask for {fragment_id}")
            mask = np.zeros(images[:2].shape)
        fragment_mask=cv2.imread(f"/home/jovyan/Bulygin/villa/ink-detection/{scroll_path}/{fragment_id}/{fragment_id_}_mask.png", 0)
        fragment_mask = np.pad(fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)
        mask = mask.astype('float32')
        mask/=255
        return images, mask, fragment_mask


def process_segment(scroll_path, segment_id):
    size = 224 #112
    in_chans = 16 #8
    tile_size = 512#256
    stride = tile_size // 8
    train_images = []
    train_masks = []
    valid_images = []
    valid_masks = []
    valid_xyxys = []

    fragment_id = segment_id
    valid_id = None
    #image, mask,fragment_mask = read_image_mask(fragment_id, scroll_path)
    try:
        image, mask,fragment_mask = read_image_mask(fragment_id, scroll_path)

        x1_list = list(range(0, image.shape[1]-tile_size+1, stride))
        y1_list = list(range(0, image.shape[0]-tile_size+1, stride))

        windows_dict={}
                
        # Use tile_size as stride to avoid overlapping tiles completely
        x1_list = list(range(0, image.shape[1]-tile_size+1, tile_size))
        y1_list = list(range(0, image.shape[0]-tile_size+1, tile_size))
        windows_dict = {}
        for a in y1_list:
                    for b in x1_list:
                        if not np.any(fragment_mask[a:a + tile_size, b:b + tile_size]==0):
                            #if (fragment_id==valid_id) or (not np.all(mask[a:a + tile_size, b:b + tile_size]<0.05)):
                                # Use 2*size as stride to space out crops within each tile
                                for yi in range(0, tile_size-size+1, 2*size):
                                    for xi in range(0, tile_size-size+1, 2*size):
                                        y1 = a+yi
                                        x1 = b+xi
                                        y2 = y1+size
                                        x2 = x1+size
                                        if fragment_id!=valid_id:
                                            train_images.append(image[y1:y2, x1:x2])
                                            train_masks.append(mask[y1:y2, x1:x2, None])
                                            assert image[y1:y2, x1:x2].shape==(size,size,in_chans)
                                        if fragment_id==valid_id:
                                            if (y1,y2,x1,x2) not in windows_dict:
                                                valid_images.append(image[y1:y2, x1:x2])
                                                valid_masks.append(mask[y1:y2, x1:x2, None])
                                                valid_xyxys.append([x1, y1, x2, y2])
                                                assert image[y1:y2, x1:x2].shape==(size,size,in_chans)
                                                windows_dict[(y1,y2,x1,x2)]='1'

        print("finished reading fragment", fragment_id)

        return train_images, train_masks, valid_images, valid_masks, valid_xyxys
    except:
          print('pff')

import torchvision.transforms as T

def preprocess_image(inp_image):
    """Загружает и предобрабатывает изображение для модели DINOv2."""
    try:
        tif_grayscale = []
        for i in range(inp_image.shape[0]):
            tif_grayscale.append(cv2.cvtColor(inp_image[i],cv2.COLOR_GRAY2RGB))
        tif_grayscale = torch.tensor(np.stack(tif_grayscale, axis=0) / 255, dtype=torch.float32)
        tif_grayscale = tif_grayscale.permute(0, 3, 1, 2)
        tif_grayscale = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(tif_grayscale)
        
        if torch.cuda.is_available():
            device = torch.device('cuda')
            img_tensor =  tif_grayscale.to(device)
            
        return img_tensor,  tif_grayscale.size
    except Exception as e:
        print(f"Ошибка при обработке {inp_image}: {e}")
        raise

# Функция для разделения большого изображения на патчи
def extract_patches(img_tensor, patch_size=PATCH_SIZE):
    """
    Разделяет большое изображение на неперекрывающиеся патчи.
    img_tensor: тензор изображения [1, C, H, W]
    patch_size: размер патча (предполагается квадратный)
    """
    _, c, h, w = img_tensor.shape
    patches = []
    for i in range(0, h - patch_size + 1, patch_size):
        for j in range(0, w - patch_size + 1, patch_size):
            patch = img_tensor[:, :, i:i+patch_size, j:j+patch_size]
            patches.append(patch)
            
    return patches


def extract_features_dinov2(inp_image, model, patch_size=PATCH_SIZE):
    img_tensor, original_size = preprocess_image(inp_image)
    
    # Получаем патчи из изображения
    patches = extract_patches(img_tensor, patch_size=patch_size)
    
    patch_features = []
    
    # Извлекаем признаки для каждого патча
    for patch in patches:
        with torch.no_grad():

            features = model.forward_features(patch)
            #print(features)
            
            # Определяем тип выходных данных и извлекаем нужные признаки
            if isinstance(features, dict):
                if 'x_norm_clstoken' in features:
                    # DINOv2 формат
                    cls_token = features['x_norm_clstoken']
                elif 'cls_token' in features:
                    cls_token = features['cls_token']
                else:
                    cls_token = features[list(features.keys())[0]]
            elif isinstance(features, torch.Tensor):
                if features.dim() == 3:  # [B, N, D]
                    cls_token = features[:, 0]  # Берем CLS-токен
                else:
                    cls_token = features
            else:
                raise TypeError(f"Неожиданный тип выходных данных: {type(features)}")
                
            patch_features.append(cls_token.cpu())
    
    # Усредняем признаки по всем патчам
    if patch_features:
        aggregated_features = torch.mean(torch.cat(patch_features, dim=0), dim=0)
    else:
        # Если изображение слишком маленькое или по другой причине патчи не получены
        with torch.no_grad():
            # Изменяем размер изображения до patch_size
            resized = interpolate(img_tensor, size=(patch_size, patch_size), 
                                mode='bilinear', align_corners=False)
            features = model.forward_features(resized)
            
            # Та же логика, что и выше
            if isinstance(features, dict):
                if 'x_norm_clstoken' in features:
                    aggregated_features = features['x_norm_clstoken'][0].cpu()
                elif 'cls_token' in features:
                    aggregated_features = features['cls_token'][0].cpu()
                else:
                    aggregated_features = features[list(features.keys())[0]][0].cpu()
            elif isinstance(features, torch.Tensor):
                if features.dim() == 3:
                    aggregated_features = features[0, 0].cpu()
                else:
                    aggregated_features = features[0].cpu()
            else:
                raise TypeError(f"Неожиданный тип выходных данных: {type(features)}")
    
    return aggregated_features.numpy(), len(patches), original_size



if __name__ == "__main__":
    scroll_result = {}
    for segment in os.listdir('/home/jovyan/Bulygin/villa/ink-detection/train_scrolls/scroll1'):
        #segment = '20230531121653'    
    #try:
        train_images, train_masks, valid_images, valid_masks, valid_xyxys = process_segment(segment_id=segment, scroll_path='train_scrolls/scroll1')
        segment_features = []
        for image in train_images:
            feature = extract_features_dinov2(image, model=model)
            segment_features.append(feature[0])
        scroll_result.update({str(segment):segment_features})
    #except:
    #        print('skip')
    print(scroll_result)