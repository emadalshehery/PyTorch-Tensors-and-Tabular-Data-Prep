# PyTorch Tensors and Tabular Data Prep

**Notebook:** `day1\_lab1.ipynb`

An introduction to PyTorch tensors, followed by a full data-preparation pipeline for the Kaggle **House Prices** dataset.

## Part 1: Tensors

|Topic|Key calls|
|-|-|
|Creating tensors|`torch.tensor`, `torch.randn`, `torch.zeros`|
|CPU vs GPU|`torch.cuda.is\_available()`, `.to(device)` (all tensors must share a device)|
|Reshaping|`view`, `reshape`, `arange`|
|Cleaning dimensions|`squeeze`, `flatten`|

## Part 2: House Price Data Prep

Dataset: `yasserh/housing-prices-dataset` (545 houses, downloaded with `kagglehub`).

1. **EDA and NaNs**: check missing values, inject NaNs into `area`, fix by filling with the mean.
2. **Encoding**: `LabelEncoder` on all text columns (`mainroad`, `furnishingstatus`, ...).
3. **Scaling**: `StandardScaler` on all input features (not on `price`).
4. **Outliers**: remove houses with `price > 10,000,000` (545 → 537 rows).
5. **Split**: 80% train / 20% test (`random\_state=42`).
6. **Tensors and DataLoaders**: float32 tensors, target reshaped to `\[N, 1]`, batch size 32 (14 training batches).

## Requirements

```bash
pip install torch pandas numpy scikit-learn kagglehub
```

Designed for Google Colab (GPU optional).

