#export CUDA_VISIBLE_DEVICES=1


model_name=FourierGNN

for pred_len in 12 24 48 96
do
python -u run.py \
      --is_training 1 \
      --root_path ./dataset/PEMS/ \
      --data_path PEMS08.npz \
      --model_id PEMS08_96_$pred_len'_'$alpha \
      --model $model_name \
      --data PEMS \
      --features M \
      --seq_len 96 \
      --pred_len $pred_len \
      --enc_in 170 \
      --des 'Exp' \
      --d_model 128 \
      --batch_size 16 \
      --patience 3 \
      --learning_rate 0.0001 \
      --train_epochs 20
done

