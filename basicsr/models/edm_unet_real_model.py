import math
import torch

from basicsr.utils.diffusion import karras_schedule
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_edm_unet_real_model import SREdmUNetRealModel
from basicsr.utils.util_image import ImageSpliterTh
import time
import torch.nn.functional as F



# from basicsr.utils import generate_lq


@MODEL_REGISTRY.register()
class EdmUNetRealModel(SREdmUNetRealModel):

    def test(self,):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        train_opt = self.opt['train']
        if train_opt.get('consistency_opt'):
            diffusion_opt = train_opt['consistency_opt']
        elif train_opt.get('diffloss_opt'):
            diffusion_opt = train_opt['diffloss_opt']
        sigma_max = diffusion_opt['sigma_max']
        use_enc = self.opt['network_g']['use_enc']
        sf = self.opt['network_g']['scale']
        chop_size = self.opt['val']['chop_size']
        chop_stride = self.opt['val']['chop_stride']
        sigma = sigma_max
        sigma = torch.as_tensor(sigma).to(self.device)
        desired_min_size = self.opt['val']['min_size']
        ori_h, ori_w = self.lq.shape[2:]
        if not (ori_h % desired_min_size == 0 and ori_w % desired_min_size == 0):
            flag_pad = True
            pad_h = (math.ceil(ori_h / desired_min_size)) * desired_min_size - ori_h
            pad_w = (math.ceil(ori_w / desired_min_size)) * desired_min_size - ori_w
            self.lq = F.pad(self.lq, pad=(0, pad_w, 0, pad_h), mode='reflect')
        else:
            flag_pad = False
            
        if self.lq.shape[2] > chop_size or self.lq.shape[3] > chop_size:
            
            im_spliter = ImageSpliterTh(
                        self.lq,
                        chop_size,
                        stride=chop_stride,
                        sf=sf,
                        extra_bs=1,
                        )
            start_event.record()
            for im_lq_pch, index_infos in im_spliter:
                im_lq_up_pch = F.interpolate(im_lq_pch, scale_factor=sf, mode='bicubic')
                
                latent = torch.randn_like(im_lq_up_pch, device=self.device)
                input = im_lq_up_pch + sigma * latent.to(torch.float32)
                
                with torch.no_grad():
                    im_sr_pch = self.sample_func(input, im_lq_pch, im_lq_up_pch, sigma, use_enc)     # 1 x c x h x w, [-1, 1]
                im_spliter.update(im_sr_pch, index_infos)
            im_sr_tensor = im_spliter.gather()
            end_event.record()
        else:
            
            lq_up = F.interpolate(self.lq, scale_factor=sf, mode='bicubic')
            latent = torch.randn_like(lq_up, device=self.device)
            input = lq_up + sigma * latent
            start_event.record()
            with torch.no_grad():
                im_sr_tensor = self.sample_func(input, self.lq, lq_up, sigma, use_enc)
            end_event.record()

        torch.cuda.synchronize()
        self.inference_time = start_event.elapsed_time(end_event)
        
        if flag_pad:
            self.output = im_sr_tensor[:, :, :ori_h * sf, :ori_w * sf]
        else:
            self.output = im_sr_tensor

    def sample_func(self, input, lq, lq_up, sigma, use_enc,):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        if use_enc:
            cond_lq = lq
        else:
            cond_lq = lq_up
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            
            with torch.no_grad():
                sr_tensor = self.net_g_ema(input, sigma, cond_lq)

        else:
            self.net_g.eval()
            with torch.no_grad():
                start_event.record()
                sr_tensor = self.net_g(input, sigma, cond_lq)
                end_event.record()
                torch.cuda.synchronize()
            self.net_g.train()

        return sr_tensor
    