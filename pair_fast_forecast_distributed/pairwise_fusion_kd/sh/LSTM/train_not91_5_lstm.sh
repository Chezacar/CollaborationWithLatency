CUDA_VISIBLE_DEVICES=5,6 python train_faf_com_kd_distributed.py --mode train --lr 0.001 --data /GPFS/data/zxlei/dataset/test1/train --nworker 0 --layer 3 --logname latency5forecast3 --resume /GPFS/data/zxlei/CollaborativePerception/Forcast/LatencyVersion/pair_fast_forecast_distributed/pairwise_fusion_kd/log/baseline/newraw/epoch_100.pth --batch 16 --log --latency_lambda 5 5 5 5 5 --utp encoder decoder adafusion classification regression --forecast_num 3 --logpath /GPFS/data/zxlei/CollaborativePerception/Forcast/LatencyVersion/pair_fast_forecast_distributed/pairwise_fusion_kd/log/lstm_l5_f3/12151311 --world_size 2 --nepoch 400 --port 10036 --forecast_loss True --forecast_model LSTM --log