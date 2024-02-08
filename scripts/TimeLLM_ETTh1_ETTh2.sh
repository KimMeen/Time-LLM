model_name=TimeLLM
learning_rate=0.01
llama_layers=32

master_port=00097
num_process=8
batch_size=24
d_model=32
d_ff=128

comment='TimeLLM-ETTh1_ETTh2'

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_pretrain.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path_pretrain ETTh1.csv \
  --data_path ETTh2.csv \
  --model_id ETTh1_ETTh2_512_96 \
  --model $model_name \
  --data_pretrain ETTh1 \
  --data ETTh2 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs 5 \
  --model_comment $comment

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_pretrain.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path_pretrain ETTh1.csv \
  --data_path ETTh2.csv \
  --model_id ETTh1_ETTh2_512_192 \
  --model $model_name \
  --data_pretrain ETTh1 \
  --data ETTh2 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 192 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model 32 \
  --d_ff 128 \
  --batch_size $batch_size \
  --learning_rate 0.02 \
  --llm_layers $llama_layers \
  --train_epochs 5 \
  --model_comment $comment

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_pretrain.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path_pretrain ETTh1.csv \
  --data_path ETTh2.csv \
  --model_id ETTh1_ETTh2_512_336 \
  --model $model_name \
  --data_pretrain ETTh1 \
  --data ETTh2 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 336 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --lradj 'COS'\
  --learning_rate 0.001 \
  --llm_layers $llama_layers \
  --train_epochs 5 \
  --model_comment $comment

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_pretrain.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path_pretrain ETTh1.csv \
  --data_path ETTh2.csv \
  --model_id ETTh1_ETTh2_512_720 \
  --model $model_name \
  --data_pretrain ETTh1 \
  --data ETTh2 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 720 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs 5 \
  --model_comment $comment