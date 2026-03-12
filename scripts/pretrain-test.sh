# this script pretrains a model on a test dataset

python \
    src/netFoundPretraining.py \
    --train_dir data/test/pretraining/final/combined/ \
    --output_dir models/test/pretraining/pretrained_model \
    --report_to tensorboard \
    --do_train \
    --num_train_epochs 3 \
    --overwrite_output_dir \
    --learning_rate 2e-5 \
    --size small \
    --do_eval \
    --validation_split_percentage 30
