# SpeedPlus

Starter kit for Kelvins SPEED+ pose estimation competition. 


### Introduction
The purpose of this repository is to help competitiors of the 
[Kelvins Satellite Pose Estimation 2021 Competition](https://kelvins.esa.int/pose-estimation-2021/home/)
 to get started with working on the SPEED+ dataset, by providing utility scripts and examples:
  * `visualize_pose.ipynb`: a Jupyter Notebook for inspecting the dataset: it plots example images,
  pose label is visualized by projected axes.
  * `submission.py`: utility class for generating valid submissions.
  * `pytorch_example.py` and `keras_example.py`: training on SPEED+ with Keras and Pytorch deep learning
  frameworks.
  * `utils.py`: utility scripts for the above examples (projection to camera frame, Keras DataGenerator
  for SPEED+, PyTorch Dataset, etc.). 
  * `starter_kit_method.ipynb`: a Jupyter Notebook for training a deep learning method on SPEED+
  
### Setting up
Clone this repository:
```
git clone https://gitlab.com/EuropeanSpaceAgency/speedplus-utils.git
cd speedplus-utils
```
Install dependencies:  
```
pip install numpy pillow matplotlib
pip install torch torchvision  # optional for running pytorch example
pip install tensorflow-gpu  # optional for running keras example
pip install jupyter  # optional for running notebook
```

### Dataset

1. Create a folder called `datasets` in `speedplus-utils`. 
2. Extract the .zip file of the [SPEED+ dataset](https://kelvins.esa.int/pose-estimation-2021/data/) to the folder `datasets`

### Training examples

We provide example training scripts for two popular Deep Learning frameworks: for Keras and PyTorch.
These examples are intentionally kept simple, leaving lots of room for improvement (dataset augmentation,
more suitable loss functions, normalizing outputs, exploring different network architectures, and so on).

Starting PyTorch training:

```
python pytorch_example.py --dataset [path to speed+] --epochs [num epochs] --batch [batch size]  --run_gpu [True] --gpu_id [0]
```
 

Similarly, to start Keras training:

```
python keras_example.py --dataset [path to speed+] --epochs [num epochs] --batch [batch size] --run_gpu[True] --gpu_id [0]
```

As the training is finished, the model is evaluated on all images of the `lightbox` and `sunlamp`
sets, and a submission file is generated that can be directly submitted on the competition page.
