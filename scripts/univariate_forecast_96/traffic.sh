export CUDA_VISIBLE_DEVICES=0

model_name=DFGCN

for pred_len in 96 192 336 720
do
python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path traffic.csv \
      --model_id traffic_96_96_$pred_len \
      --model $model_name \
      --data custom \
      --features MS \
      --seq_len 96 \
      --pred_len $pred_len \
      --e_layers 1 \
      --factor 3 \
      --enc_in 862 \
      --des 'Exp' \
      --d_model 512 \
      --batch_size 32 \
      --dropout 0 \
      --learning_rate 0.001 \
      --train_epochs 10 \
      --n_heads 4 \
      --patch_len 16 \
      --activation relu
done


#for pred_len in 96 192 336 720
#do
#python -u run.py \
#      --is_training 1 \
#      --root_path ./dataset/ \
#      --data_path traffic.csv \
#      --model_id traffic_96_96_$pred_len \
#      --model $model_name \
#      --data custom \
#      --features S \
#      --seq_len 96 \
#      --pred_len $pred_len \
#      --e_layers 1 \
#      --factor 3 \
#      --enc_in 862 \
#      --des 'Exp' \
#      --d_model 512 \
#      --batch_size 32 \
#      --dropout 0 \
#      --learning_rate 0.001 \
#      --train_epochs 10 \
#      --n_heads 4 \
#      --patch_len 16 \
#      --activation relu
#done
