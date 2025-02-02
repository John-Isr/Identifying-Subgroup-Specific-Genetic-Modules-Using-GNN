# Identifying Subgroup-Specific Genetic Modules in Gene-Gene Correlation Networks Using Graph Neural Networks

## 📌 Project Overview
This project aims to **identify subgroup-specific clusters** in gene-gene correlation networks using **supervised learning**. Each graph in our dataset contains exactly **one subgroup-specific cluster**, and our objective is to classify nodes as **belonging to this cluster (1) or not (0)**.

To achieve this, we employ **Graph Neural Networks (GNNs)** enhanced with **attention mechanisms** and **positional encodings**.

We preprocess the graph by **reordering patients based on a custom similarity score**, ensuring **invariance to permutation** in edge features. Additionally, edge attributes are incorporated into the learning process by weighting node embeddings using correlation values between gene expressions.

The initial node embeddings are optimized using **Optuna**, selecting between:
- **Spectral Positional Encoding (SPE)**, which leverages graph eigenvectors.
- **Custom Initial Node Embeddings**, embedding patient similarity into the representation.

Our model architecture utilizes **GATv2Conv layers** for message passing, integrating edge attributes into the learning process, with **GraphNorm layers** for normalization between them. After message passing, we use an **MLP head** for node classification.

For optimization, we use **Optuna** with a **TPE Sampler** for efficient hyperparameter selection and a **Hyperband Pruner** based on the **F1 score** to halt underperforming trials. The model is trained using **BCEWithLogitsLoss**, ensuring a differentiable loss function for binary classification.

To streamline experimentation, we leverage **Weights & Biases (wandb)** for real-time tracking, visualization, and hyperparameter analysis.

## 📁 Repository Structure
```
📦 Identifying-Subgroup-Specific-Genetic-Modules
├── 📂 data/                 # Raw, processed, and simulated datasets
├── 📂 models/               # Model definitions and training scripts
├── 📂 experiments/          # Logs, checkpoints, and configurations
├── 📂 notebooks/            # Jupyter notebooks for exploration
├── 📂 utils/                # Helper functions (graph processing, metrics, visualization)
├── 📂 docs/                 # Project documentation and references
├── 📂 tests/                # Unit tests for model and data validation
├── 📜 requirements.txt       # Dependencies
├── 📜 README.md              # Project overview and instructions
├── 📜 .gitignore             # Files to ignore in version control
└── 📜 LICENSE                # Open-source license (if applicable)
```

## 🔧 Setup Instructions
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Identifying-Subgroup-Specific-Genetic-Modules.git
   cd Identifying-Subgroup-Specific-Genetic-Modules
   ```
2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Set up Weights & Biases (wandb) for experiment tracking**:
   ```bash
   wandb login
   ```
   You will need an API key, which you can obtain by creating a free account at [wandb.ai](https://wandb.ai/).

## 🚀 Running the Project
- **Preprocessing Data**:
  ```bash
  python data/data_preprocessing.py
  ```
- **Training the Model with Optuna and wandb logging**:
  ```bash
  python models/train.py --config experiments/config.yaml --use_wandb
  ```
- **Evaluating the Model**:
  ```bash
  python models/evaluate.py --checkpoint experiments/checkpoints/model.pth
  ```

## 📊 Results & Visualization
We use **Weights & Biases (wandb)** for logging experiments and visualizing results. Additionally, we provide **visualization tools** in `utils/visualization.py` to inspect gene clusters and correlations in the learned representations.

## 📄 References
- **Shiran Gerassy-Vainberg & Shai S. Shen-Orr (2024)**, *A Personalized Network Framework Reveals Predictive Axis of Anti-TNF Response Across Diseases.*
- **Tsitsulin et al. (2023)**, *Graph Clustering with Graph Neural Networks.*

## 🤝 Contributors
- **Yaniv Slot-Futterman** (yaniv.slor@campus.technion.ac.il)
- **Jonathan Israel** (jonathani@campus.technion.ac.il)

## 📜 License
This project is licensed under the MIT License - see the `LICENSE` file for details.

