CUDA_VISIBLE_DEVICES=2,3 python train_faf_com_kd_distributed.py --mode train --lr 0.001 --data /GPFS/data/zxlei/dataset/test/train --nworker 0 --layer 3 --logname latency5forecast1 --resume /DB/rhome/slren/data_gpfs/log/train_single_seq/2021-09-05_22-06-09/epoch_100.pth --batch 12 --log --latency_lambda 0 0 0 0 0 --forecast_num 1 --logpath /GPFS/data/zxlei/CollaborativePerception/Forcast/LatencyVersion/pair_fast_forecast_distributed/pairwise_fusion_kd/log/baseline/TrueTrue --world_size 2 --nepoch 200 --forecast_loss False --forecast_model Baseline --encoder True --decoder True --port 10002 --log