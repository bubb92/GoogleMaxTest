RUN_NAME=${1}

ckpt_dir=gs://maxtext-experiments-multipod/mattdavidow-save-b/checkpoints
bd=gs://maxtext-experiments-multipod
dataset=gs://max-datasets-rogue
python3 MaxText/train.py MaxText/configs/base.yml run_name=${RUN_NAME} steps=6 base_output_directory=$bd dataset_path=$dataset learning_rate=1e-2 global_parameter_scale=1 per_device_batch_size=1 enable_checkpointing=False save_period=4