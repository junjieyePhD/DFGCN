#export CUDA_VISIBLE_DEVICES=1


model_name=DFGCN

for pred_len in 96 192 336 720
do
python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh1.csv \
      --model_id ETTh1_96_96_$pred_len \
      --model $model_name \
      --data ETTh1 \
      --features S \
      --seq_len 96 \
      --pred_len $pred_len \
      --e_layers 2 \
      --enc_in 7 \
      --des 'Exp' \
      --d_model 64 \
      --batch_size 32 \
      --dropout 0.1 \
      --learning_rate 0.001 \
      --train_epochs 10 \
      --patch_len 8 \
      --n_heads 1
done



#for pred_len in 96 192 336
#do
#python -u run.py \
#      --is_training 1 \
#      --root_path ./dataset/ETT-small/ \
#      --data_path ETTh1.csv \
#      --model_id ETTh1_96_96_$pred_len \
#      --model $model_name \
#      --data ETTh1 \
#      --features S \
#      --seq_len 96 \
#      --pred_len $pred_len \
#      --e_layers 1 \
#      --enc_in 7 \
#      --des 'Exp' \
#      --d_model 64 \
#      --batch_size 32 \
#      --dropout 0.1 \
#      --learning_rate 0.001 \
#      --train_epochs 10 \
#      --patch_len 8 \
#      --n_heads 1
#done

#for pred_len in 720
#do
#python -u run.py \
#      --is_training 1 \
#      --root_path ./dataset/ETT-small/ \
#      --data_path ETTh1.csv \
#      --model_id ETTh1_96_96_$pred_len \
#      --model $model_name \
#      --data ETTh1 \
#      --features S \
#      --seq_len 96 \
#      --pred_len $pred_len \
#      --e_layers 2 \
#      --enc_in 7 \
#      --des 'Exp' \
#      --d_model 64 \
#      --batch_size 32 \
#      --dropout 0.1 \
#      --learning_rate 0.001 \
#      --train_epochs 10 \
#      --patch_len 8 \
#      --n_heads 1
#done