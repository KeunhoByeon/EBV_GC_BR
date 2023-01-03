import os
import shutil

target_path = '/media/kwaklab_103/sda/data/patch_data/KBSMC/gastric/gastric_EBV_1024/Gastric_WSI/gastric_wsi_1024_08/S 2012017007/patch_1343_class_2.jpg'
save_dir = '../data'

if __name__ == '__main__':
    shutil.copy2(target_path, os.path.join(save_dir, 'target_0.png'))
