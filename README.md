# Stock Price Prediction using RNN, CNN & Hybrid CNN-RNN

Deep learning models for forecasting stock prices using historical time-series data (Apple & IBM stocks).

## Overview
This project compares three architectures for time-series forecasting:
- Simple RNN
- 1D CNN
- Hybrid CNN-RNN

## Results

**AAPL Stock:**
| Model | RMSE |
|---|---|
| RNN | 2.45 |
| 1D CNN | 30.05 |
| Hybrid CNN-RNN | **2.29** (best) |

**IBM Stock:**
| Model | RMSE |
|---|---|
| Hybrid CNN-RNN | 3.04 |

The Hybrid CNN-RNN model achieved the lowest RMSE on both AAPL and IBM datasets. The standalone 1D CNN underperformed significantly (RMSE 30.05), converging to a near-constant prediction — likely due to the limited receptive field of a single-layer CNN on this lookback window. This is included transparently as part of the model comparison rather than omitted.

![AAPL Hybrid Prediction](images/aapl_hybrid_prediction.png)
![IBM Prediction](images/ibm_prediction.png)

## Tech Stack
Python · TensorFlow/Keras · NumPy · Pandas · Matplotlib · Scikit-learn

## Key Steps
1. Time-series preprocessing & normalization
2. Sequence generation for supervised learning
3. Model training & comparison (RNN vs CNN vs Hybrid)
4. Evaluation using RMSE

## Author
Uswa Aslam
