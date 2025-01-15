#export CUDA_VISIBLE_DEVICES=1


model_name=FourierGNN

for pred_len in 96 192 336 720
do
python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh2.csv \
      --model_id ETTh2_96_96_$pred_len \
      --model $model_name \
      --data ETTh2 \
      --features M \
      --seq_len 96 \
      --pred_len $pred_len \
      --enc_in 7 \
      --des 'Exp' \
      --d_model 128 \
      --batch_size 32 \
      --patience 5 \
      --learning_rate 0.0001 \
      --train_epochs 100
done
