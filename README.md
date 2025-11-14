<div align="center">

<h1>An Advanced Vision Foundation Model and Comprehensive Benchmark Dataset for Agricultural Image Recognition</h1> 

<div>
    <a>Guorun Li</a>;
    <a>Yucong Wang</a>;
    <a>Liu Lei</a>;
    <a>Xiaoyu Li</a>;
    <a>Shuohua Zhang</a>;
    <a>Wang Liu</a>;
    <a>Yuefeng Du*</a>;
</div>

<h3><strong>submitted to Computers and Electronics in Agriculture in 2025</strong></h3>

</div>

## 🏠 TODOs
* [x] Agri430K dataset (val).    [Download Link](https://drive.google.com/drive/folders/1-EuB92O74W9MO_590YGsfAwipZbS6kQ7?usp=sharing).
* [x] AgriFM model.
* [ ] 10 Agriculture Downstream Task Datasets. 
* [ ] pretrained weights in Agri420K
* [ ] pretrained weights in ten downstream tasks

**Notes**:
- The details of Agri430K please see our paper and [excel](agri420k.xlsx).
- Please see excel(downstream tasks.xlsx) for the details of 10 downstream tasks.
- Due to the investment of funds for the work, we will make the training data publicly available after the article has been accepted.



## 🏠 Abstract
Developing a domain-specific vision foundation model (VFM) for the agricultural sector is crucial for improving the robustness and generalization of visual recognition systems in dynamic farmland environments. However, this goal requires both large-scale, high-quality domain data and a tailored model architecture. Therefore, to address the scarcity of large-scale annotated agricultural data, this work develops a benchmark dataset—Agri430K to train agricultural-specific VFMs. It contains 430,107 high-quality images covering 123 agricultural categories across 10 scenarios. Additionally, this work proposes an agricultural foundation model (AgriFM) for agricultural image recognition (AIR). It incorporates a Dynamic Attention (DA) mechanism that integrates multi-scale feature aggregation and cross-window global semantic interaction. Based on DA, this work further develops a Dynamic Transformer and a progressive hybrid architecture, enabling a smooth transition from semantic-level to element-level global representation modeling. Finally, to validate the performance and generalization of AgriFM and the importance of Agri430K, this work conducts over 200 comparative experiments. The results demonstrate that AgriFM achieves the state-of-the-art (SOTA) Top-1 accuracy of 97.4% on Agri430K and the SOTA Top-1 average accuracy of 97.6% across ten downstream AIR datasets, exhibiting strong domain generalization capability. Moreover, ten baseline models pre-trained on Agri430K outperform their ImageNet pre-trained counterparts across all downstream tasks, underscoring the value of domain-specific data for improving model performance in agricultural applications. Beyond filling the long-standing gap of VFM research in the AIR field, this work has the potential to shift the focus of AIR research from developing task-specific models toward building domain-specific foundation models. 

## 🏠 Overview
![1](figures/4.png)
![2](figures/123.png)
![3](figures/1.png)
![4](figures/3.png)


## 🎁 Train and Test
We have provided detailed instructions for model training and testing, and experimental details. 

### Install
- Clone this repo:

```bash
conda create -n agrifm python=3.10 -y
conda activate agrifm
git clone git@github.com:CAU-COE-VEICLab/AgriFM.git
cd AgriFM
```
- Install `CUDA>=10.2` with `cudnn>=7` following
  the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `PyTorch>=1.8.0` and `torchvision>=0.9.0` with `CUDA>=10.2`:

```bash
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install timm==0.4.12
pip install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8 pyyaml scipy
```


### Data preparation
We use standard ImageNet dataset, you can download it from http://image-net.org/. We provide the following two ways to
load data:

- For standard folder dataset, move validation images to labeled sub-folders. The file structure should look like:
  ```bash
  $ tree data
  ImageNet1K and Agri430K
  ├── train
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...
 
  ```

## 🏠 Supported models
* [x] swin transformer v1&v2
* [x] vision transformer
* [x] pvt 
* [x] convnext
* [x] resnet
* [x] vgg
* [x] mobilenet v2&v3
* [x] efficientnet
* [x] xception
* [x] densenet

### Evaluation

To evaluate a pre-trained `MobileNet` on Agri430K val, run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use>  main_agri430k.py --eval \
--cfg <config-file, e.g.,  configs/agri430k_benchmark/mobilenet/pretrain/mobilenetv2.yaml > --pretrained <checkpoint> --data-path <imagenet-path> 
```

## Training from scratch 

To train the `SwinTransformer-B` on Agri430K, run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use>  main.py \
--cfg <config-file, e.g.,  configs/agri430k_benchmark/swin/pretrain/swin_base_patch4_window7_224.yaml > --data-path <imagenet-path> [--batch-size <batch-size-per-gpu> --output <output-directory> --tag <job-tag>]
```

## Model Hub
TODOs
| Rank | Models           | Flops/G | Params/M | Top-1 ACC/% ImageNet1K | ImageNet1K Model | Top-1 ACC/% Agri430K | Agri430K Model |
|:----:|:----------------:|:-------:|:--------:|:----------------------:|:----------------:|:--------------------:|:--------------:|
|  1   | ResNet           |   7.9   |   44.5   |          81.7          |      [link]()    |          86.8        |     [link]()   |
|  2   | VGG              |  19.6   |  143.7   |          71.5          |      [link]()    |          81.5        |     [link]()   |
|  3   | MobileNet        |   0.3   |    3.5   |          72.0          |      [link]()    |          79.6        |     [link]()   |
|  4   | Xception         |   4.6   |   22.9   |          79.0          |      [link]()    |          91.0        |     [link]()   |
|  5   | EfficientNet     |  18.1   |  206.8   |          84.4          |      [link]()    |          90.4        |     [link]()   |
|  6   | DenseNet         |   4.4   |   20.0   |          77.5          |      [link]()    |          91.6        |     [link]()   |
|  7   | Vision Transformer|  12.7  |   88.2   |          77.9          |      [link]()    |          64.8        |     [link]()   |
|  8   | Swin Transformer |  15.2   |   87.7   |          83.5          |      [link]()    |          91.9        |     [link]()   |
|  9   | Swin Transformer V2| 15.0  |   67.0   |          83.9          |      [link]()    |          92.2        |     [link]()   |
| 10   | PVT              |  11.4   |   82.0   |          83.8          |      [link]()    |          92.2        |     [link]()   |



## Acknowledge

Our implementations are partially inspired by [Swin Transformer](https://github.com/microsoft/Swin-Transformer).

Thanks for their great works!


## Citation
