# scripts/run_pipeline.sh
python data_pipeline/generate_graphs.py \
  --conditions Optimal Suboptimal \
  --num_graphs 1000 \
  --output_dir data/graphs

python data_pipeline/init_edge_features.py \
  --input_dir data/graphs \
  --output_dir data/modified_graphs