CUDA_VISIBLE_DEVICES=0 python train_faf_com_kd.py --mode val --data /GPFS/data/zxlei/dataset/test1/val --nworker 0 --layer 3 --logname latency5forecast3 --batch 1 --log --latency_lambda 0 0 0 0 0 --forecast_num 3 --logpath /GPFS/data/zxlei/CollaborativePerception/Forcast/LatencyVersion/pair_fast_forecast_distributed/pairwise_fusion_kd/log/l5baseline --resume /GPFS/data/zxlei/CollaborativePerception/Forcast/LatencyVersion/pair_fast_forecast_distributed/pairwise_fusion_kd/log/MotionLSTM/12182148/epoch_130.pth --forecast_model MotionRNN --forecast_loss False --log