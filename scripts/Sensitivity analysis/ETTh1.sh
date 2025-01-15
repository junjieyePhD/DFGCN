#export CUDA_VISIBLE_DEVICES=1


model_name=DFGCN

for k in 0 1 2 3 4 5 6
do
python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh1.csv \
      --model_id ETTh1_96_96_$k \
      --model $model_name \
      --data ETTh1 \
      --features M \
      --seq_len 96 \
      --pred_len 192 \
      --e_layers 1 \
      --enc_in 7 \
      --des 'Exp' \
      --d_model 128 \
      --batch_size 32 \
      --dropout 0.1 \
      --learning_rate 0.001 \
      --train_epochs 10 \
      --k $k \
      --patch_len 16 \
      --n_heads 2
done