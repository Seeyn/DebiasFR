# Analyzing and Combating Attribute Bias for Face Restoration
> [[Paper]()] &emsp; [[Supplementary](https://drive.google.com/file/d/1FqAAC_c8mK6NO-FsDx0MkhvfdAirUsXs/view?usp=share_link)] &emsp; [[Colab]()] &emsp;[Video] &emsp; [ [Poster] &emsp; [Slides]<br>
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
|  StyleGAN   |  ToDo     |   
|  Attribute-Predictor    |  ToDo    |      
|  CLIP-Classifier    |  ToDo    |      



## :computer: Training


## :zap: Inferrence


```
python inference.py -i examples/ -o 
```
```
python var_age_obs.py -i examples/69006.jpg
```
## ToDo
- Training code
- Datasets

## Acknowledgments
We borrow some code from the open-source project: [GFPGAN](https://github.com/TencentARC/GFPGAN).
