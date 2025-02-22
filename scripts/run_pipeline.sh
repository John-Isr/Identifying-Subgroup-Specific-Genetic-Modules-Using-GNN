# scripts/run_pipeline.sh
python -m data_pipeline.graph_generator \
  --conditions all \
  --num_graphs 1000 \
  --output_dir data/graphs

python -m data_pipeline.init_edge_features \
  --input_dir data/graphs \
  --output_dir data/modified_graphs