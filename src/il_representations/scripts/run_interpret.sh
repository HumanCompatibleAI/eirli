model_dir=/scratch/cynthiachen/ilr-results/joint-train-procgen-dmc-2020-06-24/interpret

for model_path in $model_dir/* ; do
    model_filename=$(echo "${model_path%.*}")
    model_filename=$(echo $model_filename | rev | cut -d'/' -f 1 | rev)
    # Check number of '-'s (Helpful for recognizing DMC or procgen env)
    num_dashes=$(echo "${model_filename}" | awk -F"-" '{print NF-1}')
    # echo $num_dashes
    if [ $num_dashes -eq 1 ]
    then
	task_name=$(echo $model_filename | cut -d'-' -f 1)
        algo=$(echo $model_filename | cut -d'-' -f 2)
    else
	agent=$(echo $model_filename | cut -d'-' -f 1)
	action=$(echo $model_filename | cut -d'-' -f 2)
        task_name=$agent-$action
        algo=$(echo $model_filename | cut -d'-' -f 3)
    fi
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
