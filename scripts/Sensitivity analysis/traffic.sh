export CUDA_VISIBLE_DEVICES=0

model_name=DFGCN

for k in 0 1 2 3 4 8 16 32 64 128 256 300 400 500 600 700 800
do
python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path traffic.csv \
      --model_id traffic_96_96_$k \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 96 \
      --pred_len 96 \
      --e_layers 1 \
      --factor 3 \
      --enc_in 862 \
      --des 'Exp' \
      --d_model 512 \
      --batch_size 4 \
      --dropout 0 \
      --learning_rate 0.001 \
      --train_epochs 10 \
      --n_heads 4 \
      --patch_len 24 \
      --activation relu \
      --k $k
done


