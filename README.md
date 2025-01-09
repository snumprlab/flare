# FLARE
[**Multi-Modal Grounded Planning and Efficient Replanning For Learning Embodied Agents with A Few Examples**](https://twoongg.github.io/projects/flare),            
[Taewoong Kim](https://twoongg.github.io), 
[Byeonghwi Kim](https://bhkim94.github.io/),
[Jonghyun Choi](https://ppolon.github.io) \
[AAAI 2025](https://aaai.org/conference/aaai/aaai-25/)

[![paper](https://img.shields.io/badge/Paper-Arxiv-green)](https://arxiv.org/abs/2412.17288)

Our codebase is largely built on [FILM](https://github.com/soyeonm/FILM) and [CAPEAM](https://github.com/snumprlab/capeam)
## Environment

### Install requirements
```
$ conda create -n flare python=3.6
$ conda activate flare
$ pip install -r requirements.txt
```


### Install PyTorch
You also need to install Pytorch depending on your system. e.g., PyTorch v1.8.1 + cuda 11.1 <br>
Refer [here](https://pytorch.kr/get-started/previous-versions/)
```
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

### (Optional) Install Detectron2
If you want to visualize the semantic segmentation output, install [detectron2](https://github.com/facebookresearch/detectron2/).
```
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.6/index.html #(Linux with GPU)
CC=clang CXX=clang++ ARCHFLAGS="-arch x86_64" python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' #(Mac OS)
```

## Download Pretrained Model
We use the same pretrained perception models' wight from [FILM](https://github.com/soyeonm/FILM) and [MOCA](https://github.com/gistvision/moca).

1. Download pretrained models from [FILM]("https://github.com/soyeonm/FILM?tab=readme-ov-file#download-trained-models")
```
mv Pretrained_Models_FILM/maskrcnn_alfworld models/segmentation/maskrcnn_alfworld
mv Pretrained_Models_FILM/depth_models models/depth/depth_models
mv Pretrained_Models_FILM/new_best_model.pt models/semantic_policy/best_model_multi.pt
```

2. Download pretrained mask rcnn weight from [MOCA](https://github.com/gistvision/moca/blob/main/download_model.sh).
```
wget https://alfred-colorswap.s3.us-east-2.amazonaws.com/weight_maskrcnn.pt
```

## Additional preliminaries to use ALFRED scenes

Please make sure that ai2thor's version is 2.1.0 (if version is newer, ALFRED will break).  

* First clone the ALFRED repo
```
$ git clone https://github.com/askforalfred/alfred.git
$ cd alfred
$ export ALFRED_ROOT=$(pwd)
```

* Then run
```
$ cd $ALFRED_ROOT
$ python models/train/train_seq2seq.py --data data/json_2.1.0/ --splits data/splits/oct21.json --preprocess
```

**Please note that we do not require entire train split for model training or inference.**

The will take 5~15 minutes. You will see this:

<img width="578" src="https://user-images.githubusercontent.com/77866067/116317384-437fc980-a781-11eb-8c01-f6cdee98f824.png">

Once the bars for preprocessing are all filled, the code will break with an error message. (You can ignore and proceed). 

* Now run,  
```
$ cd .. # this repo
$ ln -s $ALFRED_ROOT/data/json_2.1.0 alfred_data_all
$ ln -s $ALFRED_ROOT/data/splits alfred_data_all/splits
```

## (Optional) MMP subgoal generation
Although we provide pre-generated subgoals with MMP, you may refer to `planner/README.md` if you are interested.

## Model Inference

**On a headless machine with gpu:**
You need to first run a Xserver with 
```
tmux
python alfred_utils/scripts/startx.py 0
```
If you set a Xdisplay other than 0 (if you ran python alfred_utils/scripts/startx.py 1, for example), run
```
export DISPLAY=:1
```
(change 1 accordingly to the Xdisplay you set up.)

Now, get out of tmux and run the following (change `DISPLAY` with what you set up e.g. 1): 

```
bash leaderboard_all.sh
```
This will initiate 48 instances at once. If you do not have 8 GPUS, you can modify shell script accordingly.

## Hardware 
Tested on:
- **GPU** - RTX A6000
- **CPU** - Intel(R) Core(TM) i7-12700K CPU @ 3.60GHz
- **RAM** - 64GB
- **OS** - Ubuntu 20.04

## Citation
```
@inproceedings{kim2025multimodal,
  author    = {Kim, Taewoong and Kim, Byeonghwi and Choi, Jonghyun},
  title     = {Multi-Modal Grounded Planning and Efficient Replanning For Learning Embodied Agents with A Few Examples},
  booktitle = {AAAI},
  year      = {2025},
}
```