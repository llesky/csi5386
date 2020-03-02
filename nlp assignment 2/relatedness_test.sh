export BERT_BASE_DIR=./uncased_L-12_H-768_A-12
export SICK=./Data
export TRAINED_CLASSIFIER=./Result_2

python ./bert/run_relatedness.py \
  --task_name=SICK \
  --do_predict=true \
  --do_eval=true \
  --data_dir=./Data \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER \
  --max_seq_length=128 \
  --output_dir=./Result_2