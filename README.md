# GT-HAD: Group-Transformer-based Hyperspectral Anomaly Detection

This repository contains the official TensorFlow implementation of the GT-HAD model for hyperspectral anomaly detection. The model leverages a Group Transformer architecture to effectively identify anomalous regions in hyperspectral images.

## Features

- **Group Transformer (GT) Block**: A novel attention mechanism that operates on groups of patches (mini-patches) within a larger image block, enabling both local and global context modeling.
- **Cooperative Match-based Module (CMM)**: A search strategy that periodically identifies and leverages "easy" (low-reconstruction-error) blocks to improve the model's focus on true anomalies.
- **End-to-End Training**: A complete pipeline for training the GT-HAD model, including data loading, block extraction, training loop, and final residual map generation.

## Project Structure

```
.
├── data/                 # Hyperspectral data files (.mat)
├── model/                # Model implementation
│   ├── attention_tf.py   # AttentionBlock implementation
│   ├── block_search_tf.py# BlockSearchTF (CMM) implementation
│   ├── block_tf.py       # BlockEmbeddingTF and BlockFoldTF
│   └── gtblock_tf.py     # Main TransformerBlock and NetTF model
├── tests/                # Test scripts
│   └── block_tf_test.py  # Test for block extraction/folding
├── train.py              # Main training script
└── requirements.txt      # Python dependencies
```

## Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- Additional libraries as listed in `requirements.txt`

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/GT-HAD.git
    cd GT-HAD
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Data

Place your hyperspectral data files (in `.mat` format) in the `data/` directory. Each `.mat` file should contain:
- `data`: An `H x W x C` NumPy array representing the hyperspectral image.
- `map`: An `H x W` NumPy array representing the ground truth anomaly map.

### Training

To train the model on a specific dataset, modify the `FILES` list in `train.py` to include the name of your data file (without the `.mat` extension).

```python
# train.py
...
FILES = ["your_dataset_name"]
...
```

Then, run the training script:

```bash
python train.py
```

The script will:
1.  Load the specified dataset.
2.  Train the GT-HAD model.
3.  Generate a residual map indicating the likelihood of anomalies.
4.  Compute the AUC score for performance evaluation.
5.  Save the residual map and ROC curve data to the `results/` directory.

### Testing

To run the provided test for the block extraction and folding functionality:

```bash
python -m tests.block_tf_test
```

This test ensures that the `BlockEmbeddingTF` and `BlockFoldTF` modules are working correctly by performing a round-trip check (extracting blocks and then folding them back into an image).

## How It Works

The core of the GT-HAD model is the `TransformerBlock`, which uses a specialized `AttentionBlock`. This attention mechanism operates on a grid of mini-patches within a larger block.

The training process involves a **Cooperative Match-based Module (CMM)**, implemented in `BlockSearchTF`. Periodically during training, the model identifies blocks that are easy to reconstruct. The attention mechanism is then adjusted to perform a more comprehensive search (all-to-all attention) within these "easy" blocks, while focusing on local context for "hard" blocks. This strategy helps the model to better distinguish between background and anomalous regions.

## Citation

Research paper : 1. GT-HAD: Gated Transformer for Hyperspectral - J. Lian, L. Wang, H. Sun and H. Huang, "GT-HAD: Gated Transformer for Hyperspectral Anomaly Detection," in IEEE Transactions on Neural Networks and Learning Systems, vol. 36, no. 2, pp. 3631-3645, Feb. 2025, doi: 10.1109/TNNLS.2024.3355166. keywords: {Image reconstruction;Feature extraction;Transformers;Tensors;Hyperspectral imaging;Anomaly detection;Task analysis;Content similarity;gating unit;hyperspectral anomaly detection (HAD);transformer},


                 2. Hyperspectral Anomaly Detection Methods: A Survey and Comparative Study Anomaly Detection - 	arXiv:2507.05730


```
[Citation information will be added here]
```

