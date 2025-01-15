#export CUDA_VISIBLE_DEVICES=1


model_name=FourierGNN

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
      --d_model 128 \
      --batch_size 16 \
      --patience 3 \
      --learning_rate 0.0001 \
      --train_epochs 20
done
