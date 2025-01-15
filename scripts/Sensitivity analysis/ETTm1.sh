#export CUDA_VISIBLE_DEVICES=1


model_name=DFGCN

for k in 0 1 2 3 4 5 6
do
    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTm1.csv \
        --model_id ETTm1_96_96_$k \
        --model $model_name \
        --data ETTm1 \
        --features M \
        --seq_len 96 \
        --pred_len 96 \
        --e_layers 1 \
        --enc_in 7 \
        --des 'Exp' \
        --d_model 128 \
        --batch_size 32 \
        --dropout 0.1 \
        --learning_rate 0.001 \
        --train_epochs 10 \
        --n_heads 2 \
        --patience 3 \
        --patch_len 16 \
        --k $k
done




for k in 0 1 2 3 4 5 6
do
    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTm2.csv \
        --model_id ETTm1_96_96_$k \
        --model $model_name \
        --data ETTm2 \
        --features M \
        --seq_len 96 \
        --pred_len 96 \
        --e_layers 1 \
        --enc_in 7 \
        --des 'Exp' \
        --d_model 64 \
        --batch_size 32 \
        --dropout 0.3 \
        --learning_rate 0.001 \
        --train_epochs 10 \
        --n_heads 1 \
        --patience 3 \
        --patch_len 16 \
        --activation relu \
        --k $k
done


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
      --pred_len 96 \
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


for k in 0 1 2 3 4 5 6
do
python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh2.csv \
      --model_id ETTh2_96_96_$k \
      --model $model_name \
      --data ETTh2 \
      --features M \
      --seq_len 96 \
      --pred_len 96 \
      --e_layers 1 \
      --enc_in 7 \
      --des 'Exp' \
      --d_model 64 \
      --batch_size 64 \
      --dropout 0.3 \
      --learning_rate 0.001 \
      --train_epochs 10 \
      --k $k \
      --patch_len 16
done



for k in 0 1 2 3 4 5 6
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




for k in 0 1 2 3 4 5 6
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
      --e_layers 2 \
      --factor 3 \
      --enc_in 862 \
      --des 'Exp' \
      --d_model 512 \
      --batch_size 16 \
      --dropout 0 \
      --learning_rate 0.001 \
      --train_epochs 10 \
      --n_heads 4 \
      --patch_len 24 \
      --activation relu \
      --k $k
done



