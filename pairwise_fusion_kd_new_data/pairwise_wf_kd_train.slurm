#!/bin/bash

#SBATCH --job-name=dgx2_test
#SBATCH --partition=dgx2
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end
#SBATCH --mail-user=renshunli@sjtu.edu.cn
#SBATCH --output=%j.out
#SBATCH --error=%j.err

module load miniconda3
source activate torch1.6
python python train_faf_com_kd.py --data /lustre/home/acct-eezy/eezy-user1/slren/dataset_warp_kd/train --logpath /lustre/home/acct-eezy/eezy-user1/slren/log --mode train --lr 0.001 --batch 4 --nepoch 100 --binary 1 --resume_teacher /lustre/home/acct-eezy/eezy-user1/slren/teacher_aug_batch_4_epoch_100.pth