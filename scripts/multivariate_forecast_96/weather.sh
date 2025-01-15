export CUDA_VISIBLE_DEVICES=0

model_name=DFGCN


for pred_len in 96 192 336 720
do
python -u run.py \
        --is_training 1 \
        --root_path ./dataset/weather/ \
        --data_path weather.csv \
        --model_id weather_96_96_$pred_len \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len 96 \
        --pred_len $pred_len \
        --e_layers 1 \
        --enc_in 21 \
        --des 'Exp' \
        --d_model 512 \
        --dropout 0.1 \
        --train_epochs 10 \
        --n_heads 8 \
        --batch_size 8 \
        --learning_rate 0.001 \
        --patch_len 16 \
        --k 1
done


