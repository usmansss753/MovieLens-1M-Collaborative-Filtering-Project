# MovieLens 1M Collaborative Filtering Project

Welcome to a hands-on project exploring collaborative filtering techniques using the MovieLens 1M dataset. This repository demonstrates two distinct approaches: a neural network-based recommender (NCF_MLP) and an autoencoder model, both implemented in PyTorch.

## Overview
This project aims to predict user ratings for movies by leveraging user-item interaction data. We compare the effectiveness of a neural collaborative filtering model and an autoencoder-based recommender.

## Directory Guide
```
CS412_A3/
├── data/
│   └── movielens_1m/
│       └── ratings.dat         # MovieLens 1M ratings file
├── results/
│   └── results.txt            # Output: model evaluation scores
├── src/
│   ├── data_loader.py         # Data loading and preprocessing
│   ├── evaluate.py            # Main experiment script
│   ├── train.py               # Model training routines
│   └── models/
│       ├── ncf_mlp.py         # Neural CF model
│       └── autoencoder.py     # Autoencoder model
├── requirements.txt           # Python dependencies
└── run_experiment.bat         # Windows batch file for running experiments
```

## Getting Started

### 1. Install Python Packages
Make sure you have Python 3.8 or newer. Install the required libraries:
```bash
pip install -r requirements.txt
```

### 2. Download the Dataset
- Get the [MovieLens 1M dataset](https://grouplens.org/datasets/movielens/1m/).
- Place the `ratings.dat` file in `data/movielens_1m/`.

### 3. Run the Experiment
You can launch the experiment in two ways:
- **Command Line:**
  ```bash
  python src/evaluate.py
  ```
- **Windows Batch File:**
  Double-click or run:
  ```bash
  run_experiment.bat
  ```

## What Happens When You Run It?
- The script loads and processes the MovieLens ratings.
- Both the NCF_MLP and autoencoder models are trained on the data.
- Each model is evaluated using Mean Absolute Error (MAE) on a test split.
- Results are saved to `results/results.txt`.

## File Descriptions
- **src/data_loader.py**: Reads the ratings file, maps user/movie IDs to indices, and prepares PyTorch DataLoaders.
- **src/models/ncf_mlp.py**: Implements a neural network for collaborative filtering.
- **src/models/autoencoder.py**: Contains the autoencoder recommender architecture.
- **src/train.py**: Training logic for both models.
- **src/evaluate.py**: Runs the full pipeline: data loading, training, evaluation, and saving results.

## Example Output
After running, you'll find something like this in `results/results.txt`:
```
Model      MAE
NCF_MLP    0.XXXX
Autoencoder 0.XXXX
```

## Troubleshooting & Tips
- **Index errors:** Make sure the dataset is in the right place and the code is mapping IDs to indices (handled in this repo).
- **Missing files:** Double-check the path to `ratings.dat`.
- **No GPU?** The code will use your CPU if CUDA is not available.

## Further Exploration
- Try adjusting model hyperparameters in the code for different results.
- Experiment with other collaborative filtering techniques or datasets.

## Credits
- MovieLens data courtesy of [GroupLens](https://grouplens.org/datasets/movielens/1m/)
- Built with PyTorch, pandas, and scikit-learn


