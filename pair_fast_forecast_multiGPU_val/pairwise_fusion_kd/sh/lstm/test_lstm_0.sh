CUDA_VISIBLE_DEVICES=7 python train_faf_com_kd.py --mode val --data /GPFS/data/zxlei/dataset/test/val --nworker 0 --layer 3 --logname latency5forecast3 --batch 1 --log --latency_lambda 0 0 0 0 0 --forecast_num 3 --logpath /GPFS/data/zxlei/CollaborativePerception/Forcast/LatencyVersion/pair_fast_forecast_distributed/pairwise_fusion_kd/log/lstm_l10f3 --resume /GPFS/data/zxlei/CollaborativePerception/Forcast/LatencyVersion/pair_fast_forecast_distributed/pairwise_fusion_kd/log/lstm_l10_f3/11182230/epoch_190.pth --forecast_mode LSTM --log