# Analyzing and Combating Attribute Bias for Face Restoration
> [[Paper]()] &emsp; [[Supplementary](https://drive.google.com/file/d/1FqAAC_c8mK6NO-FsDx0MkhvfdAirUsXs/view?usp=share_link)] &emsp; [[Colab](https://colab.research.google.com/drive/11V7E4-qAOtgiOK_4NaqJ6OPRSbHIdBph?usp=sharing)] &emsp;[Video] &emsp; [ [Poster] &emsp; [Slides]<br>
## :wrench: Dependencies and Installation

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.7](https://pytorch.org/)
- Option: NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Option: Linux

1. Clone repo

    ```bash
    git clone https://github.com/Seeyn/DebiasFR.git
    cd DebiasFR
    ```

1. Install dependent packages

    ```bash
    pip install -r requirements.txt
    pip install git+https://github.com/openai/CLIP.git
    ```
## 📖 Model Zoo

|  Model    | Link     |      
| ---- | ---- |
|  DebiasFR   |  [Google Drive](https://drive.google.com/file/d/10DmjHUC_3GCzi4G1WBEYxYLtgbuHIYdm/view?usp=share_link)   |   
|  StyleGAN (not officially trained)   |  [Github](https://github.com/Seeyn/DebiasFR/releases/download/v1/DebiasFR_stylegan.pth)     |   
|  Attribute-Predictor    |  [Github](https://github.com/Seeyn/DebiasFR/releases/download/v1/Attribute_predictor.pth)    |      
|  CLIP-Classifier    |  [Github](https://github.com/Seeyn/DebiasFR/releases/download/v1/Clip_classifier.pth)    |      

## 📖 Datasets

|  Name   | Link     |      
| ---- | ---- |
|  FFHQ   |  [Link]()   |   
|  CelebA-HQ   |  [Link]()     |   
|  IMDB  |  [Link]()    |      
|  COX   |  [Link]()    |   

Attribute labels (name2age/gender) for FFHQ:

- Age: [Google Drive](https://drive.google.com/file/d/1tf62TuCWwmgiswEvRoFx2Uz67ZNFG-AD/view?usp=drive_link)
- Gender: [Google Drive](https://drive.google.com/file/d/1zWqGzn9XzF85tCf4dTifVp6n0xEhKhB2/view?usp=drive_link)
## :computer: Training
```
python -m torch.distributed.launch --nproc_per_node=1 --master_port=22021 dbfr/train.py -opt options/degradation_tune_lr_wpseudo.yml --launcher pytorch
```
## :zap: Inferrence


```
python inference.py -i examples/ -o results
```

## ToDo

- Datasets

## Acknowledgments
We borrow some code from the open-source project: [GFPGAN](https://github.com/TencentARC/GFPGAN).
