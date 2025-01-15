export CUDA_VISIBLE_DEVICES=0


model_name=DFGCN


for pred_len in 12 24 48
do
python -u run.py \
      --is_training 1 \
      --root_path ./dataset/PEMS/ \
      --data_path PEMS08.npz \
      --model_id PEMS08_96_$pred_len'_'$alpha \
      --model $model_name \
      --data PEMS \
      --features M \
      --seq_len 96 \
      --pred_len $pred_len \
      --e_layers 1 \
      --use_norm 1 \
      --enc_in 170 \
      --des 'Exp' \
      --d_model 512 \
      --learning_rate 0.001 \
      --batch_size 32 \
      --train_epochs 10 \
      --k 2 \
      --patch_len 16 \
      --activation relu \
      --dropout 0
done


for pred_len in 96
do
python -u run.py \
      --is_training 1 \
      --root_path ./dataset/PEMS/ \
      --data_path PEMS08.npz \
      --model_id PEMS08_96_$pred_len \
      --model $model_name \
      --data PEMS \
      --features M \
      --seq_len 96 \
      --pred_len $pred_len \
      --e_layers 1 \
      --enc_in 170 \
      --des 'Exp' \
      --d_model 512 \
      --learning_rate 0.002 \
      --batch_size 16 \
      --train_epochs 10 \
      --k 2 \
      --patch_len 8 \
      --use_norm 0 \
      --activation relu \
      --dropout 0.1
done


