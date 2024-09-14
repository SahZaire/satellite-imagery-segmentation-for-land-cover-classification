import os
import cv2
import numpy as np
from tqdm import tqdm
import logging
from patchify import patchify
from PIL import Image
import shutil

class TiffCreator:
    def __init__(self, output_dir='tiff_files', patch_size=256):
        self.output_dir = output_dir
        self.patch_size = patch_size
        self.images_dir = os.path.join(output_dir, 'images256')
        self.masks_dir = os.path.join(output_dir, 'masks256')

    def create_tiff_files(self, df):
        if self.tiff_files_exist():
            logging.info("TIFF files already exist. Skipping creation.")
            return

        self._setup_directories()
        
        num_of_saved_files = 0
        for img_path, mask_path in tqdm(df[['sat_image_path', 'mask_path']].to_numpy()):
            try:
                image = self._read_image(img_path)
                mask = self._read_mask(mask_path)

                if image is None or mask is None:
                    continue

                if image.shape != mask.shape:
                    logging.warning(f"Image and mask shapes do not match for {img_path}")
                    continue

                patches_img, patches_mask = self._create_patches(image, mask)
                num_of_saved_files += self._save_patches(patches_img, patches_mask, num_of_saved_files)

            except Exception as e:
                logging.error(f"Error processing {img_path}: {str(e)}")

        logging.info(f"Total number of TIFF files created: {num_of_saved_files}")

    def tiff_files_exist(self):
        return os.path.exists(self.images_dir) and os.path.exists(self.masks_dir)

    def _setup_directories(self):
        for directory in [self.images_dir, self.masks_dir]:
            if os.path.exists(directory):
                shutil.rmtree(directory)
            os.makedirs(directory)

    def _read_image(self, path):
        image = cv2.imread(path)
        if image is None:
            logging.warning(f"Failed to read image: {path}")
        return image

    def _read_mask(self, path):
        if path is not None and os.path.exists(path):
            mask = cv2.imread(path)
            if mask is None:
                logging.warning(f"Failed to read mask: {path}")
        else:
            mask = None
        return mask

    def _create_patches(self, image, mask):
        SIZE_X = (image.shape[1] // self.patch_size) * self.patch_size
        SIZE_Y = (image.shape[0] // self.patch_size) * self.patch_size
        
        image = Image.fromarray(image)
        image = image.crop((0, 0, SIZE_X, SIZE_Y))
        image = np.array(image)

        patches_img = patchify(image, (self.patch_size, self.patch_size, 3), step=self.patch_size)
        
        if mask is not None:
            mask = Image.fromarray(mask)
            mask = mask.crop((0, 0, SIZE_X, SIZE_Y))
            mask = np.array(mask)
            patches_mask = patchify(mask, (self.patch_size, self.patch_size, 3), step=self.patch_size)
        else:
            patches_mask = np.zeros_like(patches_img)

        return patches_img, patches_mask

    def _save_patches(self, patches_img, patches_mask, start_index):
        saved_files = 0
        for i in range(patches_img.shape[0]):
            for j in range(patches_img.shape[1]):
                single_patch_mask = patches_mask[i, j, :, :]
                single_patch_mask = single_patch_mask[0]
                
                val, counts = np.unique(single_patch_mask, return_counts=True)
                max_counts = np.max(counts) / counts.sum()
                if max_counts < 0.95:
                    single_patch_img = patches_img[i, j, :, :]
                    single_patch_img = single_patch_img[0]
                    cv2.imwrite(f'{self.images_dir}/{start_index + saved_files}.tif', single_patch_img)
                    cv2.imwrite(f'{self.masks_dir}/{start_index + saved_files}.tif', single_patch_mask)
                    saved_files += 1
        return saved_files