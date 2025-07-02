<div align="center">
<h2> Consistency Trajectory Matching for One-Step Generative Super-Resolution (ICCV 2025) </h2>



[Weiyi You](https://scholar.google.com/citations?user=q4uALoAAAAAJ),  [Mingyang Zhang](),  [Leheng Zhang](https://scholar.google.com/citations?user=DH1CJqkAAAAJ&hl=zh-CN),  [Xingyu Zhou](https://scholar.google.com/citations?user=dgO3CyMAAAAJ),  [Kexuan Shi](https://scholar.google.com/citations?user=dX-aOIwAAAAJ&hl=zh-CN),  [Shuhang Gu](https://scholar.google.com/citations?user=-kSTt40AAAAJ)


[![arXiv](https://img.shields.io/badge/arXiv-2503.20349-b31b1b.svg)](https://arxiv.org/abs/2503.20349v3)
[![GitHub Stars](https://img.shields.io/github/stars/LabShuHangGU/CTMSR?style=social)](https://github.com/LabShuHangGU/CTMSR)




<img width="800" src="assets/method.png"> 
<img width="800" src="assets/visual_result.png"> 

</div>


## News

- 📄 **[2025.03.27]** Paper preprint released!
- 🏆 **[2025.06.26]** Our paper has been accepted to **ICCV 2025**!
- 💾 **[2025.06.30]** Codebase and model checkpoints are now available.


## Environment
- Python 3.9
- PyTorch 2.0.1

### Installation
```bash
git clone https://github.com/LabShuHangGU/CTMSR.git

conda create -n ctmsr python=3.9
conda activate ctmsr

pip install -r requirements.txt
python setup.py develop
```

## Training
### Data Preparation
- Download the training dataset [ImageNet](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php) and put them in the folder `./datasets`.

### Training Commands
- Refer to the training configuration files in `./options/train` folder for detailed settings.
```bash
# batch size = 4 (GPUs) × 8 (per GPU)

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --use-env --nproc_per_node=4 --master_port=1145  basicsr/train.py -opt options/train/ctmsr_train.yml --launcher pytorch
```

## Testing
### Data Preparation
- Download and generate the testing data ([ImageNet-Test](https://github.com/zsyOAOA/ResShift/tree/journal) + [RealSR](https://github.com/csjcai/RealSR) + [RealSet65](https://github.com/zsyOAOA/ResShift/tree/journal)) and put them in the folder `./datasets`.

### Pretrained Models
- Download the [pretrained models](https://huggingface.co/ywy123/CTMSR/blob/main/CTMSR.pth) and put them in the folder `./experiments/pretrained_models`.

### Testing Commands
- Refer to the testing configuration files in `./options/test` folder for detailed settings.
```bash
CUDA_VISIBLE_DEVICES=0 python basicsr/test.py -opt options/test/ctmsr_test.yml
```


## Citation

```
@article{you2025consistency,
  title={Consistency Trajectory Matching for One-Step Generative Super-Resolution},
  author={You, Weiyi and Zhang, Mingyang and Zhang, Leheng and Zhou, Xingyu and Shi, Kexuan and Gu, Shuhang},
  journal={arXiv preprint arXiv:2503.20349},
  year={2025}
}
```

## Acknowledgements
This code is built on [BasicSR](https://github.com/XPixelGroup/BasicSR) and [ResShift](https://github.com/zsyOAOA/ResShift/tree/journal).
