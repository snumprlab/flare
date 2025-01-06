export FILM=$(pwd)

SPLIT=$1
FROM_IDX=$2
TO_IDX=$3
DN=$4

python main.py \
        --max_episode_length 1000       \
        --num_local_step 25     \
        --num_processes 1       \
        --eval_split $SPLIT     \
        --from_idx $FROM_IDX    \
        --to_idx $TO_IDX        \
        --max_fails 10  \
        --debug_local   \
        --learned_depth \
        --set_dn $DN  \
        --appended      \
        --use_sem_seg   \
        --use_sem_policy \
        --which_gpu 0   \
        --sem_gpu_id 0  \
        --sem_seg_gpu 0 \
        --depth_gpu 0   \
        --x_display 1   \
        --seed 6        \
        

