# scripts/run_pipeline.sh
python -m data_pipeline.graph_generator \
  --conditions Optimal Suboptimal \
  --num_graphs 100 \
  --output_dir data/graphs

python -m data_pipeline.init_edge_features \
  --input_dir data/graphs \
  --output_dir data/modified_graphs