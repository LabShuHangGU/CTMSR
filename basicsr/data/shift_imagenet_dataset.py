import cv2
import numpy as np
import os.path as osp
from torch.utils import data as data
from torchvision.transforms.functional import normalize


from basicsr.data.data_util import paths_from_lmdb, scandir
from basicsr.data.transforms import augment, paired_random_crop, random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.matlab_functions import imresize, rgb2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class ShiftImagenetDataset(data.Dataset):
    
    def __init__(self, opt):
        super(ShiftImagenetDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.gt_folder = opt['dataroot_gt']

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            self.paths = paths_from_lmdb(self.gt_folder)
        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [osp.join(self.gt_folder, line.split(' ')[0]) for line in fin]
        else:
            self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        upscale = None
        downscale = None
        if 'upscale' in self.opt and 'downscale' in self.opt:
            upscale = self.opt['upscale']
            downscale = self.opt['downscale']
        scale = self.opt['scale']

        if upscale is not None and downscale is not None:
            scale = upscale / downscale

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        img_lq_upsample = None

        # # modcrop
        # size_h, size_w, _ = img_gt.shape
        # size_h = size_h - size_h % scale
        # size_w = size_w - size_w % scale
        # img_gt = img_gt[0:size_h, 0:size_w, :]

        # generate training pairs
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            h, w = img_gt.shape[:2]
            if h < gt_size or w < gt_size:
                pad_h = max(0, gt_size - h)
                pad_w = max(0, gt_size - w)
                img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            img_gt = random_crop(img_gt, gt_size)
            # img_lq = cv2.resize(img_gt, dsize=None, fx=1/scale, fy=1/scale, interpolation=cv2.INTER_CUBIC)
            img_lq = imresize(img_gt, 1/scale, antialiasing=True)
            if self.opt['upsample']:
                if self.opt['upsampler'] == 'bicubic':
                    # img_lq_upsample = cv2.resize(img_lq, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    img_lq_upsample = imresize(img_lq, scale, antialiasing=True)
                elif self.opt['upsampler'] == 'nearest':
                    img_lq_upsample = cv2.resize(img_lq, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
         

            # img_gt = augment([img_gt,], self.opt['use_hflip'], self.opt['use_rot'])

            
        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_lq = imresize(img_gt, 1/scale, antialiasing=True)
            # img_lq = cv2.resize(img_gt, dsize=None, fx=1/scale, fy=1/scale, interpolation=cv2.INTER_CUBIC)
            if self.opt['upsample']:
                if self.opt['upsampler'] == 'bicubic':
                    # img_lq_upsample = cv2.resize(img_lq, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    img_lq_upsample = imresize(img_lq, scale, antialiasing=True)
                elif self.opt['upsampler'] == 'nearest':
                    img_lq_upsample = cv2.resize(img_lq, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
         
            # img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True)
        img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)
        if self.opt['upsample']:
            img_lq_upsample = img2tensor(img_lq_upsample, bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None: 
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            if self.opt['upsample']:
                normalize(img_lq_upsample, self.mean, self.std, inplace=True)
        if img_lq_upsample is None:
            return {'gt': img_gt, 'gt_path': gt_path, 'lq': img_lq}
        # print(img_gt.shape)
        return {'lq': img_lq, 'gt': img_gt, 'gt_path': gt_path, 'lq_up': img_lq_upsample,}

    def __len__(self):
        return len(self.paths)