export CUDA_VISIBLE_DEVICES=0


model_name=DFGCN


for k in 1 2 4 8 16 32
do
python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path Flight.csv \
      --model_id Flight_96_96_$k \
      --model $model_name \
      --data custom \
      --features M \
      --seq_len 96 \
      --pred_len 96 \
      --e_layers 1 \
      --enc_in 7 \
      --des 'Exp' \
      --d_model 256 \
      --batch_size 32 \
      --dropout 0.1 \
      --learning_rate 0.001 \
      --train_epochs 10 \
      --k $k \
      --patch_len 16 \
      --activation relu \
      --n_heads 1
done

