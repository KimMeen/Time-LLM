#!/bin/bash

# Set script variables
model_name="TimeLLM"
train_epochs=2
learning_rate=0.01
llama_layers=32

num_process=8
batch_size=10
d_model=16
d_ff=32

comment="TimeLLM-EAN"

# Launch the training script without GPU acceleration
accelerate launch --num_processes "$num_process" run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path data.csv \
  --model_id EAN_id \
  --model "$model_name" \
  --data ean \
  --target sold_units\
  --features S \
  --seq_len 13 \
  --label_len 1 \
  --pred_len 1 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment
