export CUDA_VISIBLE_DEVICES=0


model_name=DFGCN

for k in 0 1 2 3 4 8 16 32 40 50 60 70 80 90 100 110 120 130
do
python -u run.py \
      --is_training 1 \
      --root_path ./dataset/solar/ \
      --data_path solar_AL.csv \
      --model_id solar_96_96_$k \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 96 \
      --pred_len 96 \
      --enc_in 137 \
      --des 'Exp' \
      --e_layers 1 \
      --d_model 512 \
      --dropout 0.1 \
      --n_heads 4 \
      --batch_size 16 \
      --train_epochs 10 \
      --learning_rate 0.001 \
      --patch_len 16 \
      --k $k
done


