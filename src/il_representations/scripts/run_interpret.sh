# CUDA_VISIBLE_DEVICES=1 python ./src/il_representations/scripts/interpret.py with \
#     encoder_path=/scratch/cynthiachen/ilr-results/procgen-simclr-2021-06-24/8/il_train/1/snapshots/policy_00012500_batches.pt \
#     save_video=True \
#     chosen_algo=integrated_gradient \
#     length=1000

for f in /home/cynthiachen/il-representations/analysis/data/procgen/coinrun-aug*; do
    name_clean=$(echo $f| rev | cut -d'/' -f 1 | rev)
    filename=$(echo $name_clean| cut -d'.' -f 1)
    taskname=$(echo $filename| cut -d'-' -f 1)
    # for dmc
    subtask=$(echo $filename| cut -d'-' -f 2)
    echo $name_clean, $filename, $taskname, $subtask
    CUDA_VISIBLE_DEVICES=0 python ./src/il_representations/scripts/interpret.py with \
        encoder_path=${f} \
        save_video=True \
        save_image=True \
        save_original_image=True \
        filename=${filename} \
        chosen_algo=saliency \
        length=1000 \
        env_cfg.benchmark_name=procgen \
        env_cfg.task_name=${taskname}
done
