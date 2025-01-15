#export CUDA_VISIBLE_DEVICES=1

model_name=DFGCN

for pred_len in 96 192 336 720
do
python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path exchange_rate.csv \
      --model_id exchange_rate_96_96_$pred_len \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 96 \
      --pred_len $pred_len \
      --e_layers 1 \
      --enc_in 8 \
      --des 'Exp' \
      --d_model 128 \
      --batch_size 32 \
      --dropout 0.5 \
      --learning_rate 0.0001 \
      --train_epochs 10 \
      --patch_len 8 \
      --k 2 \
      --activation relu \
      --n_heads 1
done

