CUDA_VISIBLE_DEVICES=4 python train_faf_com_kd.py --mode val --data /GPFS/data/zxlei/dataset/test1/val --nworker 0 --layer 3 --logname latency5forecast3 --batch 1 --log --latency_lambda 1 1 1 1 1 --forecast_num 1 --logpath /GPFS/data/zxlei/CollaborativePerception/Forcast/LatencyVersion/pair_fast_forecast_distributed/pairwise_fusion_kd/log/l5baseline --resume /GPFS/data/zxlei/CollaborativePerception/Forcast/LatencyVersion/pair_fast_forecast_distributed/pairwise_fusion_kd/log/baseline/newraw/epoch_100.pth --forecast_model MotionLSTM --forecast_loss False --log