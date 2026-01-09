DATA_FOLDER=$HOME/chLi/Dataset/GS/haizei_1
ITERATIONS=1000

CUDA_VISIBLE_DEVICES=2 \
  python train.py \
  -s ${DATA_FOLDER}/mv_2d3d_match/colmap/ \
  -m ${DATA_FOLDER}/cdgs/ \
  --depth \
  -r 1 \
  --eval \
  --predefined_depth_path ${DATA_FOLDER}/mv_2d3d_match/colmap/depth_npy/ \
  --confidence_map_output \
  --debug_output \
  --debug_dir ${DATA_FOLDER}/cdgs/debug/ \
  --iterations ${ITERATIONS}
