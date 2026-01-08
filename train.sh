DATA_FOLDER=$HOME/chLi/Dataset/GS/haizei_1

python train.py \
  -s ${DATA_FOLDER}/gs/ \
  -m ${DATA_FOLDER}/cdgs/ \
  --depth \
  -r 1 \
  --eval \
  --predefined_depth_path ${DATA_FOLDER}/vggt_depth/ \
  --confidence_map_output \
  --debug_output \
  --debug_dir ${DATA_FOLDER}/cdgs/debug/ \
  --iterations 1000
