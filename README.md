# YouTube Virality Predictor

A ML model for predicting YouTube video virality scores using multimodal features from video metadata, titles, and thumbnails.

## Overview

This project tackles the **Viral Vision: YouTube Virality Predictor Challenge** on Kaggle. The goal is to predict a video's virality score based on:
- Video metadata (subscriber count, duration, publish time)
- Title text (multilingual, with emojis and hashtags)
- Thumbnail images

## Approach

### Feature Engineering

#### Tabular Features
- **Log-scaled numerical features**: Subscriber count, duration, and days since publish are highly skewed → log transformation improves linearity with target
- **Temporal features**: Hour, weekday, and month encoded using cyclical sin/cos to preserve periodicity
- **Target encoding**: Quantile-binned versions of subscriber count, duration, and days since publish with fold-wise statistics (mean, std, count, min, max)

#### Title Features
- **Lexical signals**: Number of emojis, hashtags, exclamation marks, question marks, and ALL-CAPS words
- **Sentiment**: VADER compound sentiment score
- **Text embeddings**:
  - **E5** (`intfloat/multilingual-e5-large`): Multilingual embeddings robust to language variation and informal tokens
  - **Qwen** (`Qwen/Qwen3-Embedding-0.6B`): Instruction-tuned embeddings with task-specific prompting for virality prediction

#### Thumbnail Features
- **Deep embeddings**:
  - **CLIP** (`laion/CLIP-ViT-g-14-laion2B-s12B-b42K`): Strong general-purpose visual representations
  - **SigLIP** (`google/siglip-so400m-patch14-384`): Complementary visual features
- **Low-level features**: Brightness, saturation, and edge density from HSV/Canny analysis
- **Missing thumbnail handling**: Zero-image imputation with binary indicator feature

#### Dimensionality Reduction
All embeddings reduced to **50 dimensions via PCA** — empirically found sweet-spot between information retention and noise reduction.

## Model Architecture

### Base Models (5-Fold CV)
| Model | Description |
|-------|-------------|
| LightGBM | Gradient boosting with early stopping |
| LightGBM (DART) | Dropout-regularized gradient boosting |
| XGBoost | Histogram-based gradient boosting |
| CatBoost | Categorical-aware gradient boosting |
| Random Forest | Bagged decision trees |
| RealMLP | Tabular deep learning |
| LightAutoML (LAMA) | AutoML with linear + GBM ensemble |

### Ensemble Strategy
- Out-of-fold predictions from all 7 models
- **Ridge Regression** meta-learner for final blending

## Key Findings

### Title Analysis
- Higher emoji/hashtag counts → lower variance in virality, tighter upward-shifted distribution
- Weekend uploads show slightly higher median virality with more stable performance
- N-gram analysis reveals viral patterns: "try not to laugh", "viral shorts", "DIY craft"

## Project Structure

```
├── main.ipynb          # Full training pipeline
├── train.csv           # Training data
├── test.csv            # Test data
├── embeddings/         # Pre-computed embeddings
│   ├── clip_train.npy
│   ├── clip_test.npy
│   ├── siglip_train.npy
│   ├── siglip_test.npy
│   ├── qwen_train.npy
│   ├── qwen_test.npy
│   ├── train_emb.npy   # E5 embeddings
│   └── test_emb.npy
└── thumbnail/          # Downloaded thumbnails
    ├── train/
    └── test/
```

## Requirements

```
torch
transformers
sentence-transformers
lightgbm
xgboost
catboost
scikit-learn
pytabkit
lightautoml
vaderSentiment
opencv-python
pandas
numpy
emoji
```

## Usage

1. Place `train.csv` and `test.csv` in the project directory
2. Run the notebook cells sequentially:
   - Install dependencies
   - Generate text/image embeddings (or use pre-computed)
   - Train all base models with 5-fold CV
   - Stack with Ridge regression
   - Generate submission


