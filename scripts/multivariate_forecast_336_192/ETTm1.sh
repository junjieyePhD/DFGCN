#export CUDA_VISIBLE_DEVICES=1


model_name=DFGCN
seq_len=336

for pred_len in 96 192 336 720
do
    python -u run.py \
        --is_training 1 \
        --root_path ./dataset/ETT-small/ \
        --data_path ETTm1.csv \
        --model_id ETTm1_96_$seq_len'_'$pred_len \
        --model "$model_name" \
        --data ETTm1 \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --e_layers 1 \
        --enc_in 7 \
        --des 'Exp' \
        --d_model 64 \
        --batch_size 128 \
        --dropout 0.3 \
        --learning_rate 0.001 \
        --train_epochs 10 \
        --n_heads 1 \
        --patience 3 \
        --patch_len 24 \
        --k 2
done


#for pred_len in 96 192 336 720
#do
#    python -u run.py \
#        --is_training 1 \
#        --root_path ./dataset/ETT-small/ \
#        --data_path ETTm1.csv \
#        --model_id ETTm1_96_$seq_len'_'$pred_len \
#        --model "$model_name" \
#        --data ETTm1 \
#        --features M \
#        --seq_len $seq_len \
#        --pred_len $pred_len \
#        --e_layers 1 \
#        --enc_in 7 \
#        --des 'Exp' \
#        --d_model 64 \
#        --batch_size 128 \
#        --dropout 0.3 \
#        --learning_rate 0.001 \
#        --train_epochs 10 \
#        --n_heads 1 \
#        --patience 3 \
#        --patch_len 24 \
#        --k 2
#done