# this script finetunes a model on a test dataset

python \
    src/netFoundFinetuning.py \
    --train_dir data/test/finetuning/final/combined \
    --model_name_or_path models/test/pretraining/pretrained_model \
    --output_dir models/test/finetuning/finetuned_model \
    --overwrite_output_dir \
    --validation_split_percentage 20 \
    --do_train \
    --do_eval \
    --eval_strategy epoch \
    --save_strategy epoch \
    --learning_rate 0.01 \
    --num_train_epochs 4 \
    --problem_type single_label_classification \
    --num_labels 2 \
    --freeze_base True \
    --size small \
    --load_best_model_at_end


