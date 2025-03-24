# CoATF: Convolution and Attention based Tensor Factorization Model for Context-aware Recommendation

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Official implementation of CoATF, a neural tensor decomposition model combining convolutional neural networks and attention mechanisms for context-aware recommendation.

## Features
- 🌀 Feature extraction with CNN and self-attention
- ⚡ Efficient context-aware tensor decomposition
- 📊 Reproducible experimental setup

## Requirements
```bash
Python 3.8
torch >= 1.10
scipy >= 1.7
numpy >= 1.21
scikit-learn >= 0.24
matplotlib >= 3.5
```

## Installation
```bash
git clone https://github.com/livelihao/CoATF.git
cd CoATF
```

## Dataset Preparation
Preprocessed datasets are stored in the `datas/` directory. 

```python
dataset = "Ciao"  # for Ciao
# dataset = 'mov100k' # for MovieLens
```

## Usage
### Training Configuration
```python
--kfold        # Cross-validation folds (5 or 10)
--seed 2025    # Random seed
--rank 30      # Tensor factorization rank
--epochs 200    # Maximum training epochs
--batch_size 64 # Training batch size
```

### Basic Training
```bash
# 10-fold cross-validation
python main.py --kfold 10

# 5-fold cross-validation 
python main.py --kfold 5
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
``` 

Key elements implemented:
1. Clear experiment reproduction instructions
2. Version-specific Python requirement
3. Standardized result reporting format
4. Flexible configuration parameters
5. Complete training pipeline from preprocessing to visualization
6. Academic paper citation template
7. License information for open-source compliance

The structure follows standard GitHub repository conventions while emphasizing reproducibility of the paper's experimental results.