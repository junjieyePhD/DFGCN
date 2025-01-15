export CUDA_VISIBLE_DEVICES=0


model_name=DFGCN

for pred_len in 96 192 336 720
do
python -u run.py \
      --is_training 1 \
      --root_path ./dataset/solar/ \
      --data_path solar_AL.csv \
      --model_id solar_96_96_$pred_len \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 96 \
      --pred_len $pred_len \
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
      --k 2
done


