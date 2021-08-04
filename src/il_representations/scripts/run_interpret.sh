model_dir=/scratch/cynthiachen/ilr-results/cluster-results/bc-repl-pretrain-procgen-dmc/models-dmc

for model_path in $model_dir/* ; do
    model_filename=$(echo "${model_path%.*}")
    model_filename=$(echo $model_filename | rev | cut -d'/' -f 1 | rev)
    agent=$(echo $model_filename | cut -d'-' -f 1)
    action=$(echo $model_filename | cut -d'-' -f 2)
    algo=$(echo $model_filename | cut -d'-' -f 3)
    task_name=$agent-$action
    echo $task_name, $algo
    CUDA_VISIBLE_DEVICES=$device python src/il_representations/scripts/interpret.py with \
        encoder_path=$model_path \
        save_video=True \
        length=1000 \
        save_image=True \
        filename=$task_name-$algo \
        env_cfg.benchmark_name=dm_control \
        env_cfg.task_name=$task_name
done
