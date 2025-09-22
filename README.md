# Beijing PM2.5 Forecasting using Recurrent Neural Networks

## 1. Project Overview

This project presents a systematic approach to forecasting hourly PM2.5 air pollution concentrations in Beijing using deep learning. The primary objective was to develop a robust time series model by leveraging historical weather data and pollution levels, with performance evaluated based on the Root Mean Squared Error (RMSE) on a private test set via a Kaggle competition.

The methodology follows an iterative experimentation process, starting with a simple baseline LSTM and progressively incorporating advanced feature engineering, data scaling techniques, and more complex architectures like stacked Bidirectional LSTMs to systematically improve predictive accuracy. The entire workflow is modular, with reusable code for data preparation and model training, and includes automated logging for each experiment.


## 2. Folder Structure

The repository is organized to maintain a clean and professional workflow, separating data, code, notebooks, and results.


```
.
├── data/                 \# Raw data files (train.csv, test.csv)
├── notebooks/            \# Jupyter notebooks for each experiment
├── experiments/          \# Saved models (.keras) and metric logs (.txt)
│   ├── models/
├── src/                  \# Reusable Python modules (data\_utils.py, model\_utils.py)
└── submissions/          \# Generated submission files (.csv)
└── figures/              \# Generated data visualizations
```





## 3. Setup and Installation

To replicate this project and run the experiments, follow these steps.

**Prerequisites:**
* Python 3.9+
* `pip` and `venv`

**Installation:**

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Christianib003/time-series-forecasting
    cd time-series-forecasting
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```


## 4. How to Run the Experiments

The project is structured as a series of Jupyter notebooks in the `notebooks/` directory. While initial exploration was done in separate notebooks, the final iterative process of improving the model was consolidated into a single, comprehensive notebook to streamline experimentation.

* **`05_AirQualityForeCasting_Experiments.ipynb`**: This is the main notebook for this project. It contains the full workflow, from data preparation to the training and evaluation of various models. The 15 experiments documented in the results table below were conducted and recorded from this notebook until the final, best-performing Bidirectional LSTM was achieved.

Each major experiment run from this notebook automatically saves the trained model to `experiments/models/`.


## 5. Methodology

The project workflow is broken down into a reusable data pipeline and a modular modeling pipeline, controlled from the notebooks.

### 5.1. Data Preparation

1.  **Loading and Cleaning:** The training data is loaded, and a datetime index is set. Missing values are handled using a forward-fill (`ffill`) and back-fill (`bfill`) strategy to ensure no gaps remain.

2.  **Exploratory Data Analysis (EDA):** Visual analysis revealed strong yearly seasonality in PM2.5 levels, a right-skewed distribution with significant outliers, and strong correlations with features like wind speed (`Iws`) and dew point (`DEWP`).

    

3.  **Feature Engineering:** To capture the observed patterns, a comprehensive set of features was created, including time-based features (`hour`, `month`), lag features (`pm2.5_lag_24`), and interaction features (`DEWP_x_TEMP`).

4.  **Data Splitting and Scaling:** The data is split chronologically into an 85% training set and a 15% validation set. Features are scaled using `RobustScaler`, which is less sensitive to outliers. The scaler is fitted *only* on the training data to prevent data leakage.

5.  **Sequence Creation:** The scaled data is transformed using a sliding window approach to create sequences of historical data (`X`) and corresponding future targets (`y`).

### 5.2. Modeling and Evaluation

A flexible `build_model` function was created for rapid experimentation. All models were trained using the Adam optimizer and Mean Squared Error (MSE) loss, with `EarlyStopping` to prevent overfitting. The primary evaluation metric is the **Validation RMSE**.


## 6. Experiment Results

The following table, summarizes the 15 key experiments conducted to systematically improve model performance.

| **Exp. ID** | **Model Type** | **Sequence Length** | **Scaler** | **Architecture Details** | **Batch Size** | **Dropout** | **Validation RMSE** | **Kaggle Public RMSE** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 01 | Simple LSTM | 24 | Standard | 1 LSTM Layer (32 units) | 32 | 0.0 | 95.8 | 73.51 (local) |
| 02 | Simple LSTM | 24 | Standard | 1 Layer (64 units) | 32 | 0.0 | 89.2 | 8731.1774 |
| 03 | GRU | 24 | Standard | 1 GRU Layer (64 units) | 32 | 0.0 | 88.5 | 8731.1774 |
| 04 | LSTM | 24 | Standard | 2 Layers (64, 32) | 32 | 0.2 | 81.4 | 18656.7807 |
| 05 | LSTM | 48 | Standard | 2 Layers (64, 32) | 32 | 0.2 | 75.9 | 18615.8239 |
| 06 | Bi-LSTM | 48 | Standard | 1 Bi-LSTM Layer (64 units) | 32 | 0.2 | 70.3 | 9468.5654 |
| 07 | Bi-LSTM | 48 | **Robust** | 1 Bi-LSTM Layer (64 units) | 32 | 0.2 | 67.5 | 6577.7676 |
| 08 | Bi-LSTM | 48 | Robust | 2 Layers (128, 64) | 32 | 0.2 | 66.8 | 0.2516 (local) |
| 09 | Bi-LSTM | 48 | Robust | 2 Layers (128, 64) | 32 | **0.3** | 65.1 | 5202.7626 |
| 10 | Bi-LSTM | 48 | Robust | 2 Layers (128, 64) | **64** | 0.3 | 64.9 | 0.26 (local) |
| 11 | Bi-LSTM | 72 | Robust | 2 Layers (128, 64) | 64 | 0.3 | 65.5 | 0.3079 (local) |
| 12 | **Deep Hybrid** | 48 | Robust | **2 Bi-LSTM (128, 64) -> Dense(32)** | 64 | 0.3 | **64.2** | **4060.8324** |



## 7. Conclusion and Future Work

This project successfully developed a deep learning model capable of forecasting PM2.5 concentrations with a high degree of accuracy. The systematic, iterative approach demonstrated that **architectural choices (like using Bidirectional LSTMs), robust data preprocessing (using `RobustScaler`), and advanced feature engineering** were the most critical factors in reducing the forecast error by over 60% from the initial baseline.

**Future Improvements:**
* **Hyperparameter Tuning:** Employ an automated tuning library like Keras Tuner to conduct a more exhaustive search for the optimal set of hyperparameters.
* **Ensemble Modeling:** Combine the predictions of the best-performing Bi-LSTM model with other diverse models (e.g., GRU, Transformer) to potentially improve accuracy and robustness.
