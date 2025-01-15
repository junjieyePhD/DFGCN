export CUDA_VISIBLE_DEVICES=0

model_name=DFGCN


for k in 1 2 3 4 5 6 7 9 10 11 12 13 14 15 17 18 19
do
python -u run.py \
        --is_training 1 \
        --root_path ./dataset/weather/ \
        --data_path weather.csv \
        --model_id weather_96_96_$k \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len 96 \
        --pred_len 96 \
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
        --k $k
done

