CUDA_VISIBLE_DEVICES=4 python train_faf_com_kd.py --mode val --data /GPFS/data/zxlei/dataset/test1/val --nworker 0 --layer 3 --logname latency5forecast3 --batch 1 --log --latency_lambda 5 5 5 5 5 --forecast_num 3 --logpath /GPFS/data/zxlei/CollaborativePerception/Forcast/LatencyVersion/pair_fast_forecast_distributed/pairwise_fusion_kd/log/l5f3motionnet --resume /GPFS/data/zxlei/CollaborativePerception/Forcast/LatencyVersion/pair_fast_forecast_distributed/pairwise_fusion_kd/log/motionNet_l5_f3/12261327/epoch_395.pth --forecast_model MotionNet --log