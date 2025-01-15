export CUDA_VISIBLE_DEVICES=0

model_name=DFGCN

for k in 1 2 4 8 16 32 64 128 150 170 190 210 230 250 270 290 320
do
python -u run.py \
        --is_training 1 \
        --root_path ./dataset/electricity/ \
        --data_path electricity.csv \
        --model_id ECL_96_96_$k \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len 96 \
        --pred_len 96 \
        --e_layers 2 \
        --enc_in 321 \
        --des 'Exp' \
        --d_model 512 \
        --batch_size 16 \
        --dropout 0.1 \
        --learning_rate 0.001 \
        --n_heads 4 \
        --train_epochs 10 \
        --patch_len 24 \
        --activation relu \
        --k $k
done

