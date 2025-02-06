# Identifying Subgroup-Specific Genetic Modules in Gene-Gene Correlation Networks Using Graph Neural Networks

## 📌 Project Overview
This project leverages Graph Neural Networks (GNNs) to **identify subgroup-specific clusters** within gene-gene correlation networks using a supervised learning approach. Each graph in the dataset contains exactly **one subgroup-specific cluster**, and our objective is to classify nodes as either **belonging (1)** or **not belonging (0)** to this cluster.

Key components include:

- **Graph Simulation & Preprocessing:** Generate raw graphs and initialize edge features.
- **GNN Architecture:** A custom classifier built with GNN layers (e.g., GATv2Conv) that incorporates edge attributes, attention mechanisms, and normalization (GraphNorm) to effectively learn node representations.
- **Training & Hyperparameter Tuning:** Training routines and hyperparameter optimization using Optuna with W&B logging.
- **Inference Pipeline:** A clean interface to convert NetworkX graphs to annotated graphs with node predictions.
- **Experiment Tracking:** Real-time logging and visualization with Weights & Biases (wandb).

---

## 📁 Repository Structure


```
Identifying-Subgroup-Specific-Genetic-Modules-Using-GNN
├── data/
│   ├── graphs/                    # Raw generated graphs
│   └── modified_graphs/           # Graphs with initialized edge features
├── data_pipeline/
│   ├── __init__.py                # Module initializer for the data pipeline
│   ├── graph_generator.py         # Graph simulation and generation
│   ├── init_edge_features.py      # Edge feature initialization
│   └── simulation.py              # Graph simulation framework
├── experiments/
│   ├── optuna_studies/
│   │   ├── example_study.db       # Example database for Optuna studies
│   │   └── study.db               # Optuna study results
│   ├── current_trial_best.yaml    # Best trial configuration from the currently running study
│   ├── default_config.yaml        # Default hyper parameter configuration
│   └── trial_best.yaml            # Best trial configuration after hyperparameter tuning
├── models/
│   ├── __init__.py                # Module initializer for models
│   ├── architecture.py            # GNN architecture and weight initialization
│   ├── hyperparameter_tuning.py   # Hyperparameter tuning routines
│   └── training.py                # Training routines for the model
├── notebooks/
│   └── Project.ipynb              # Jupyter Notebook for project exploration
├── scripts/
│   ├── __init__.py                # Module initializer for scripts
│   ├── inference.py               # Script for inference on graphs
│   ├── run_pipeline.sh            # Shell script to run the entire pipeline
│   ├── train_model.py             # Script to train the model using default or custom configuration
│   └── tune_hyperparams.py        # Script to tune model hyperparameters
├── tests/
│   ├── edge_init_test.py          # Test for edge initialization
│   ├── edge_init_test_load_parse.py  # Test for loading and parsing edge initialization
│   ├── graph_generator_test.py    # Tests for graph generator
│   ├── graph_generator_test2.py   # Additional tests for graph generator
│   └── test_graph_generator_cli.py # CLI tests for graph generator
├── utils/
│   ├── __init__.py                # Module initializer for utilities
│   ├── logging.py                 # Logging utilities for W&B integration
│   └── preprocessing.py           # Preprocessing utilities
├── .gitignore                     # Git ignore file
├── LICENSE                        # License information
├── README.md                      # Project documentation
└── requirements.txt               # Python dependencies

```

---

## 🔧 Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/John-Isr/Identifying-Subgroup-Specific-Genetic-Modules-Using-GNN.git
   cd Identifying-Subgroup-Specific-Genetic-Modules
   ```

2. **Create and Activate a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Weights & Biases (wandb):**
   ```bash
   wandb login
   ```
   *(Obtain your API key by creating a free account at [wandb.ai](https://wandb.ai/).)*

---
#TODO: running the project needs updating
## 🚀 Running the Project

### 1. Generate and Preprocess Data
Generate raw graphs and initialize their edge features:
```bash
python data_pipeline/generate_graphs.py
python data_pipeline/init_edge_features.py
```

### 2. Train the Model
Train your GNN model using the training script and configuration file:
```bash
python scripts/train_model.py --config experiments/default_config.yaml --use_wandb
```

### 3. Hyperparameter Tuning
Optimize the model’s hyperparameters with Optuna:
```bash
python scripts/tune_hyperparams.py --config experiments/default_config.yaml
```

### 4. Run Inference
Use the inference script to process a NetworkX graph and obtain node predictions:
```python
import networkx as nx
from scripts.inference import run_inference

# Create or load your NetworkX graph
nx_graph = nx.erdos_renyi_graph(100, 0.15)

# Ensure each node has the features expected by your model
for node in nx_graph.nodes():
    nx_graph.nodes[node]['feature'] = [0.5]  # Example feature

# Run inference to add the 'classification' attribute to nodes
result_graph = run_inference(
    nx_graph=nx_graph,
    model_path="experiments/best_model.pth",
    device='auto'
)

# Retrieve predictions
predictions = nx.get_node_attributes(result_graph, 'classification')
print(predictions)
```

---

## 📊 Results & Visualization
Real-time experiment tracking and visualization are handled via **Weights & Biases (wandb)**.

---

## 📄 References
- **Brody et al. (2021)**, *How Attentive are Graph Attention Networks?*
- **Shiran Gerassy-Vainberg & Shai S. Shen-Orr (2024)**, *A Personalized Network Framework Reveals Predictive Axis of Anti-TNF Response Across Diseases.*
- **Tsitsulin et al. (2023)**, *Graph Clustering with Graph Neural Networks.*

---

## 🤝 Contributors
- **Yaniv Slor Futterman** (yaniv.slor@campus.technion.ac.il)
- **Jonathan Israel** (jonathani@campus.technion.ac.il)

---

## 📜 License
This project is licensed under the Apache 2.0 License – see the [`LICENSE`](LICENSE) file for details.

