CUDA_VISIBLE_DEVICES=4,7 python train_faf_com_kd_distributed.py --mode train --lr 0.001 --data /GPFS/data/zxlei/dataset/test/train --nworker 0 --layer 3 --logname latency5forecast3 --resume /DB/rhome/slren/data_gpfs/log/train_single_seq/2021-09-05_22-06-09/epoch_100.pth --batch 8 --log --latency_lambda 10 10 10 10 10 --utp encoder decoder adafusion classification regression --forecast_num 3 --logpath /GPFS/data/zxlei/CollaborativePerception/Forcast/LatencyVersion/pair_fast_forecast_distributed/pairwise_fusion_kd/log/motionNet_l10_f3/11301719 --world_size 2 --nepoch 400 --port 10026 --forecast_loss True --encoder False --decode False --forecast_model MotionNet --log