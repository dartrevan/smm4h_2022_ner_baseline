#!/usr/bin/env bash
export DATADIR=./data
export MAX_LENGTH=128
export BERT_MODEL=cimm-kzn/rudr-bert
export OUTPUT_DIR=workdir
export BATCH_SIZE=32
export NUM_EPOCHS=20
export SAVE_STEPS=750
export SEED=1

python preprocess.py --input $DATADIR/train.csv --output $DATADIR/train.json --do_lower
python preprocess.py --input $DATADIR/valid.csv --output $DATADIR/valid.json --do_lower
python preprocess.py --input $DATADIR/test.csv --output $DATADIR/test.json --do_lower

CUDA_VISIBLE_DEVICES=0 python run_ner.py --train_file $DATADIR/train.json \
                                         --validation_file $DATADIR/valid.json \
                                         --test_file $DATADIR/test.json \
                                         --model_name_or_path $BERT_MODEL \
                                         --output_dir $OUTPUT_DIR \
                                         --max_seq_length  $MAX_LENGTH \
                                         --num_train_epochs $NUM_EPOCHS \
                                         --per_device_train_batch_size $BATCH_SIZE \
                                         --save_steps $SAVE_STEPS \
                                         --seed $SEED \
                                         --do_train \
                                         --do_eval \
                                         --do_predict

python postprocess.py --predicted_labels $OUTPUT_DIR/predictions.txt \
                      --tokens $DATADIR/test.json \
                      --documents $DATADIR/test.csv \
                      --save_to $DATADIR/pred.csv
