## Usage
The directory containing the nuScenes data should be like this:
```
|-- path to nuScenes root directory
|   |-- maps
|   |-- samples
|   |-- sweeps
|   |-- v1.0-trainval
```
## Step

```
1. export PYTHONPATH=/home/yimingli/when2com/PoseAwareCollaborativePerception/nuscenes-devkit/python-sdk:/home/yimingli/when2com/PoseAwareCollaborativePerception:$PYTHONPATH
2. cd data/mapping; set eigen3 path; delete cmake cache file; run commend: cmake ./ & make all
3. Install conda
4. create evironment: conda env create -f environment.yml
5. modify parameters in create_data_com.py
6. generate split.npy file to split data into train/val/test, the exist split.npy file is an example 
7. create data: cd /PATH_TO_PROJECT; conda activate object_det; python create_data_com.py -r DATASET -s SPLIT -p '/home/yml/PoseAwareCollaborativePerception/data/dataset/'
8. visualization: python/Dataset_com.py (modify parameters in Dataset_com.py if necessary)
9. sample command for training:
train_faf_com.py --data /PATH/TO/DATASET --mode train --log --lr 0.001 --batch 2 --nepoch 100 --binary 1
10. sample command for validation:
train_faf_com.py --data /PATH/TO/DATASET --resume /PATH/TO/NETWORK.pth --mode val
```
## Difference from Baseline Detector

1. In create_data_com.py, we create training data of different agents in multiple folders (/train/agent0,agent1,...,agent4), but do not calculate relative pose of them. Note that the agent number is fixed, we just pad 0 for those empty positions.
2. In train_faf_com.py, we load training data using five data loaders, and write them as batch form. The codewords will communicate with each other and then be decoded. 


CUDA_VISIBLE_DEVICES=3 python train_faf_com_kd.py --data /DATA_SSD/slren/dataset_warp_kd/train --logpath /GPFS/data/zxlei/CollaborativePerception/Forcast/LatencyVersion/pairwise_fast/pairwise_fusion_kd/log --mode train --lr 0.001 --batch 4 --nepoch 100 --binary 1 --resume_teacher /DATA_SSD/slren/teacher_aug_batch_4_epoch_100.pth --log


CUDA_VISIBLE_DEVICES=4 python train_faf_com_kd.py --data /GPFS/data/slren/dataset_warp_kd/train --logpath /GPFS/data/zxlei/CollaborativePerception/Forcast/LatencyVersion/pairwise_fast/pairwise_fusion_kd/log --mode train --lr 0.001 --batch 4 --nepoch 100 --binary 1 --log

--resume_teacher /DATA_SSD/slren/warp2com_weighted_fusion/epoch_100.pth


CUDA_VISIBLE_DEVICES=4 python train_faf_com_kd.py --mode val --data /GPFS/data/slren/dataset_warp_kd/val --nworker 0 --resume //GPFS/data/zxlei/CollaborativePerception/Forcast/LatencyVersion/pairwise_fast/pairwise_fusion_kd/log/train_single_seq/2021-08-05_19-47-55/epoch_90.pth --layer 3 --logname pairwise_layer3_wf_latency_100epoch --log --logpath /GPFS/data/zxlei/CollaborativePerception/Forcast/LatencyVersion/pairwise_fast/pairwise_fusion_kd/log

CUDA_VISIBLE_DEVICES=0,1 python train_faf_com_kd_distributed.py --mode train --lr 0.001 --data /GPFS/data/zxlei/dataset/test/train --nworker 0 --layer 3 --logname latency5forecast1 --batch 8 --log --latency_lambda 5 5 5 5 5 --forecast_num 3 --logpath /GPFS/data/zxlei/CollaborativePerception/Forcast/LatencyVersion/pair_fast_forecast_distributed/pairwise_fusion_kd/log/motionNet_l5_f3 --world_size 2 --nepoch 200 --log

CUDA_VISIBLE_DEVICES=2,3 python train_faf_com_kd_distributed.py --mode train --lr 0.001 --data /GPFS/data/zxlei/dataset/test/train --nworker 0 --layer 3 --logname latency5forecast1 --batch 8 --log --latency_lambda 10 10 10 10 10 --forecast_num 3 --logpath /GPFS/data/zxlei/CollaborativePerception/Forcast/LatencyVersion/pair_fast_forecast_distributed/pairwise_fusion_kd/log/motionNet_l10_f3 --world_size 2 --nepoch 200 --log
