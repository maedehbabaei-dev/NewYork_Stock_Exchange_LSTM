# NewYork_Stock_Exchange_LSTM
A Deep Learning project using TensorFlow/Keras LSTM to predict AAPL (Apple) stock closing prices from historical OHLCV data.

# 📈 NYSE Stock Price Prediction using LSTM

A Deep Learning project for predicting the next day's Apple (AAPL) closing stock price using a Long Short-Term Memory (LSTM) neural network built with TensorFlow/Keras.

The model uses the previous 60 trading days of Open, High, Low, Close, and Volume data to learn temporal patterns and predict the next day's closing price.

---

## 📌 Project Overview

Stock market prices form a time series where historical observations can contain useful temporal patterns.

In this project, an LSTM neural network is used to model these sequential dependencies and predict the next day's Close price of Apple Inc. (AAPL).

The project focuses on:

- Time-series data preprocessing
- Chronological train/validation/test splitting
- Data normalization using Min-Max Scaling
- Sliding-window sequence generation
- LSTM-based Deep Learning
- Early stopping and learning-rate reduction
- Regression evaluation using multiple metrics
- Visualization of actual vs. predicted stock prices

---

## 🎯 Objective

The main objective is to build an LSTM-based regression model that can learn patterns from historical AAPL stock data and predict the next trading day's closing price.

### Input

The model uses five features:

- Open
- High
- Low
- Close
- Volume

For each prediction, the model receives the previous 60 trading days.

### Output

The model predicts:

> The next day's closing price of AAPL

---

## 📊 Dataset

The project uses the NYSE Stock Prices Dataset available on Kaggle.

Dataset: NYSE Stock Prices  
Source: Kaggle  
Dataset Link:  
https://www.kaggle.com/datasets/dgawlik/nyse

The dataset contains historical stock market information for multiple companies.

For this project, only AAPL (Apple Inc.) data is selected.

The following features are used:

| Feature | Description |
|---|---|
| open | Opening stock price |
| high | Highest price during the trading day |
| low | Lowest price during the trading day |
| close | Closing stock price |
| volume | Number of shares traded |

---

## 🔄 Data Preprocessing

The data preprocessing pipeline consists of several steps.

### 1. Select AAPL

Only records corresponding to the AAPL stock symbol are selected.

### 2. Convert Dates

The date column is converted to a datetime format and the data is sorted chronologically.

### 3. Handle Missing Values

Rows containing missing values are removed.

### 4. Select Features

The following five numerical features are used:

`text
Open
High
Low
Close
Volume

5. Chronological Data Split
Because this is a time-series forecasting problem, the data is not randomly shuffled.
The dataset is divided chronologically:
Range: 0 to 1

Importantly, the scaler is fitted only on the training data and then applied to the validation and test sets.
This helps prevent data leakage.
🪟 Sequence Generation
LSTM models require sequential input.
A sliding window of 60 trading days is used.
For example:
Days 1–60   → Predict Day 61
Days 2–61   → Predict Day 62
Days 3–62   → Predict Day 63
...
Therefore, each input sample has the shape: 
(60, 5)

where:
60 = previous trading days
5 = Open, High, Low, Close, Volume
🧠 Model Architecture
The project uses a TensorFlow/Keras LSTM neural network.

Architecture
Input
  │
  ▼
LSTM (128 units)
  │
  ▼
Dropout (0.2)
  │
  ▼
Dense (1)
  │
  ▼
Predicted Close Price

Model Configuration
**Component**    **Configuration**
Input sequence      60 trading days
Input features      5
LSTM units          128
Dropout             0.2
Output layer        Dense(1)
Optimizer           Adam
Learning rate       0.001
Loss function       Mean Squared Error (MSE)
Metric              Mean Absolute Error (MAE)
Batch size          32
Maximum epochs      150


⚙️ Training Strategy
Two callbacks are used during training.
Early Stopping
Training monitors the validation loss.
If the validation loss does not improve for 15 consecutive epochs, training stops and the best model weights are restored.

Patience = 15
Restore best weights = True

Reduce Learning Rate
The learning rate is reduced when validation loss stops improving.
Configuration:
Factor = 0.5
Patience = 5
Minimum learning rate = 1e-6
This allows the model to make smaller optimization steps when learning becomes slower.

📈 Training Performance
The training and validation loss curves show that the model learns rapidly during the early epochs and the loss decreases substantially over training.
�
The training and validation losses remain relatively close, indicating that there is no obvious severe overfitting in the training process.

📊 Prediction Results
The following graph compares the actual AAPL closing prices with the prices predicted by the LSTM model on the test set.
�
The predicted values follow the overall movement of the actual stock price reasonably well.
The model captures major upward and downward trends, although some short-term fluctuations and sharp movements are not predicted perfectly.

🧪 Evaluation
The trained model is evaluated on the unseen test set using four regression metrics.
Results   
**Metric**     **Result**
MSE              5.8299
MAE              1.8507
RMSE             2.4145
R²               0.8962
Metric Interpretation
Mean Squared Error (MSE)
MSE = 5.8299

MSE measures the average squared difference between the actual and predicted closing prices.
Lower values indicate better performance.
Mean Absolute Error (MAE)

MAE = 1.8507

On average, the model's prediction differs from the actual closing price by approximately $1.85 on the test set.
Root Mean Squared Error (RMSE)

RMSE = 2.4145

RMSE represents the square root of the mean squared prediction error and is expressed in the same unit as the stock price.
R² Score

R² = 0.8962

The model achieved an R² score of 0.8962 on the unseen test set, indicating that the model explains a substantial portion of the variation in the test-set closing prices.
Note: R² should not be interpreted as stock-price prediction accuracy or as a guarantee of future market performance.

🏗️ Project Workflow
 NYSE Dataset
     │
     ▼
Select AAPL
     │
     ▼
Sort by Date
     │
     ▼
Handle Missing Values
     │
     ▼
Select OHLCV Features
     │
     ▼
Chronological Split
     │
     ├── 70% Training
     ├── 15% Validation
     └── 15% Testing
     │
     ▼
Min-Max Scaling
     │
     ▼
Create 60-Day Sequences
     │
     ▼
LSTM Neural Network
     │
     ▼
Model Training
     │
     ├── Early Stopping
     └── Learning Rate Reduction
     │
     ▼
Test Set Prediction
     │
     ▼
Inverse Scaling
     │
     ▼
Regression Evaluation
     │
     ▼
Visualization


💻 Technologies Used
•Python
•NumPy
•Pandas
•Matplotlib
•Scikit-learn
•TensorFlow
•Keras
•Google Colab
•Jupyter Notebook
•Main Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

📁 Project Structure
NYSE-Stock-Price-Prediction-LSTM/
│
├── NYSE_Stock_Exchange_LSTM.ipynb
├── README.md
│
├── loss_curve.png
└── actual_vs_predicted.png

The image filenames can be changed to match the names used in the repository.

🚀 How to Run
1. Clone the Repository
git clone https://github.com/your-username/NYSE-Stock-Price-Prediction-LSTM.git

2. Open the Notebook
Open:
NYSE_Stock_Exchange_LSTM.ipynb

using:
•Google Colab
•Jupyter Notebook
•JupyterLab

3. Download the Dataset
Download the NYSE dataset from Kaggle:
https://www.kaggle.com/datasets/dgawlik/nyse

Place the dataset in the appropriate directory.

The notebook expects:
prices-split-adjusted.csv

4. Update the Dataset Path
Modify the path in the notebook if necessary:
df = pd.read_csv('/content/nyse_data/prices-split-adjusted.csv') 

5. Run the Notebook
Run the notebook from top to bottom.
The notebook will:
1_Load the dataset
2_Select AAPL
3_Preprocess the data
4_Scale the features
5_Create time-series sequences
6_Build the LSTM model
7_Train the model
8_Evaluate the model
9_Generate visualizations

🔍 Important Implementation Details
No Random Train/Test Split
A random split is avoided because it could allow future stock prices to influence the training process.
Instead, the project uses:
Past → Training
Later → Validation
Most Recent → Testing
This better reflects the real-world structure of time-series forecasting.
Leakage-Free Scaling
The MinMaxScaler is fitted only using the training data:
scaler.fit_transform(train_data)
The validation and test data are transformed using the already-fitted scaler:
scaler.transform(val_data)
scaler.transform(test_data)
This prevents information from the validation or test periods from influencing the scaling process.

📌 Limitations
Although the model achieves a strong test-set R² score, stock-price forecasting is inherently difficult and highly uncertain.
This project has several limitations:
•It focuses on only one stock: AAPL.
•It uses historical OHLCV data only.
•It does not include news or market sentiment.
•It does not include macroeconomic indicators.
•It does not implement a trading strategy.
•It does not account for transaction costs or slippage.
•The model is evaluated on historical data and does not guarantee future performance.
•A good regression score does not necessarily translate into profitable trading.
Therefore, this project should be considered a Deep Learning time-series forecasting experiment, not a financial trading system.

🔮 Future Improvements
Several improvements could be explored in future versions.
Data Improvements
  •Add technical indicators such as:
     °Moving Averages
     °RSI
    ° MACD
    °Bollinger Bands
•Include market indices such as S&P 500 and NASDAQ.
•Include economic indicators.
•Incorporate financial news and sentiment analysis.

Model Improvements
•Compare LSTM with GRU.
•Use stacked LSTM layers.
•Experiment with Bidirectional LSTM.
•Compare LSTM with Transformer-based models.
•Perform systematic hyperparameter tuning.

Evaluation Improvements
•Use walk-forward validation.
•Evaluate directional accuracy.
•Compare predicted returns instead of only prices.
•Test the model across multiple stocks.
•Compare the model with simple baselines such as:
   °Previous day's closing price
   °Moving average
   °Linear regression

Trading-Oriented Extensions

A future version could build a separate trading strategy based on model predictions and evaluate:

•Cumulative return
•Sharpe ratio
•Maximum drawdown
•Win rate
•Transaction costs

📚 Key Concepts Demonstrated
This project demonstrates practical understanding of:
•Time-Series Forecasting
•Deep Learning
•Recurrent Neural Networks (RNNs)
•Long Short-Term Memory (LSTM)
•Sequence Modeling
•Regression
•Feature Scaling
•Sliding Windows
•Chronological Data Splitting
•Data Leakage Prevention
•Early Stopping
•Learning Rate Scheduling
•Model Evaluation
•Data Visualization

⭐ Conclusion
This project demonstrates how an LSTM neural network can be applied to historical stock-market time-series data to predict the next day's closing price.
Using the previous 60 trading days and five OHLCV features, the model achieved:
MAE  = 1.8507
RMSE = 2.4145
R²   = 0.8962
The results show that the LSTM successfully learned meaningful temporal patterns from the historical AAPL data and was able to follow the overall movement of the stock price on the unseen test set.
However, financial markets are highly dynamic, and historical predictive performance should not be interpreted as a guarantee of future results.

👩‍💻 Author
Developed by Maedeh Babaei
GitHub:
https://github.com/maedehbabaei-dev
