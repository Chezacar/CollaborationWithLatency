CUDA_VISIBLE_DEVICES=3 python train_faf_com_kd.py --mode val --data /GPFS/data/zxlei/dataset/test1/val --nworker 0 --layer 3 --logname latency5forecast3 --batch 1 --log --latency_lambda 10 10 10 10 10 --forecast_num 3 --logpath /GPFS/data/zxlei/CollaborativePerception/Forcast/LatencyVersion/pair_fast_forecast_distributed/pairwise_fusion_kd/log/l5baseline --resume /GPFS/data/zxlei/CollaborativePerception/Forcast/LatencyVersion/pair_fast_forecast_distributed/pairwise_fusion_kd/log/MotionLSTM/01171715/epoch_320.pth --forecast_model MotionLSTM --forecast_loss False --log
