# Air Quality Forecasting Challenge - Task Overview

## Challenge Description

This is the first graded assignment for the Machine Learning Techniques I course, focusing on applying RNNs and LSTM models to forecast air pollution levels (PM2.5) in Beijing using historical air quality and weather data.

## Objective

-   **Primary Goal**: Achieve RMSE < 4000 on Kaggle Public Leaderboard ✅ ACHIEVED (3877.96)
-   **Target Goal**: Achieve RMSE < 3000 on Kaggle Private Leaderboard
-   **Stretch Goal**: Achieve RMSE ~2000 for competitive performance

## Current Status

-   **Current Best Public Score**: 3877.96 (Experiment 6)
-   **Status**: Target < 4000 achieved, working toward < 3000 and ideally ~2000
-   **Submissions Made**: 6 experiments with consistent improvement

## Submission Requirements

### 1. Comprehensive Report

-   **Introduction**: Problem description and approach
-   **Data Exploration**: Dataset analysis and preprocessing steps
-   **Model Design**: Best performing architecture and design choices
-   **Experiment Table**: Systematic experiments with parameters
-   **Results**: Performance analysis and key findings
-   **Conclusion**: Summary and proposed improvements

### 2. GitHub Repository

-   Well-documented code
-   Clear experiment tracking
-   Reproducible results

### 3. Kaggle Submissions

-   Up to 10 submissions per day
-   Format according to sample_submission.csv
-   Avoid similarity > 50% or AI-generated work

## Rubric Breakdown (Total: 55 points)

### Approach to the Challenge (5 points)

-   Clear explanation of time series approach
-   Justification for RNN/LSTM usage
-   Data exploration and preprocessing plan
-   Specific goals and architecture testing strategy

### Data Exploration, Preprocessing & Feature Engineering (15 points)

-   Thorough dataset exploration with statistics and visualizations
-   Detailed preprocessing (missing data, sequences, windowing)
-   Explained feature engineering relevance to model performance
-   Justified visualizations that inform model building

### Model Design & Architecture (15 points)

-   Well-optimized RNN/LSTM architecture
-   Detailed specifications (layers, units, activations, optimizers, learning rates)
-   Justified design choices with diagrams
-   Multiple architecture comparisons

### Kaggle Private Leaderboard Score (20 points)

-   **Exemplary (20 pts)**: Score < 3000
-   **Target**: Score ~2000 for competitive advantage

## Current Progress Analysis

### Experiment Evolution

1. **Experiment 1**: 6914.80 RMSE (Baseline LSTM)
2. **Experiments 2-3**: ~5500-5000 RMSE (Enhanced features)
3. **Experiment 4**: 4639.71 RMSE (Bidirectional LSTM)
4. **Experiment 5**: 4060.83 RMSE (Enhanced Bidirectional v2)
5. **Experiment 6**: 3877.96 RMSE (Optimized ensemble approach) ✅

### Key Improvements Made

-   Enhanced feature engineering (31+ features)
-   Bidirectional LSTM architecture
-   48-hour sequence length
-   RobustScaler for outlier handling
-   Ensemble approaches
-   Advanced temporal encoding

### Next Steps Required

-   Target RMSE ~2000 for competitive private leaderboard performance
-   Advanced model architectures (Transformer, CNN-LSTM hybrid)
-   Sophisticated ensemble methods
-   Advanced feature engineering and data augmentation
-   Hyperparameter optimization with multiple model types

## Data Overview

-   **Training Data**: Historical air quality and weather data
-   **Features**: Weather conditions (TEMP, DEWP, PRES, etc.) + PM2.5 target
-   **Test Data**: Future periods requiring PM2.5 predictions
-   **Challenge**: Time series forecasting with temporal dependencies

## Technical Requirements

-   RNN or LSTM models (can explore advanced variants)
-   Time series preprocessing and sequence generation
-   Feature engineering for temporal patterns
-   Model optimization and experimentation
-   Kaggle submission format compliance
    ...........................
    Air Quality Forecasting
    This is your first graded assignment for the Machine Learning Techniques I course. It focuses on applying Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) models to solve a real-world problem: forecasting air pollution levels. Air pollution, particularly PM2.5, is a critical global issue that impacts public health and urban planning. By accurately predicting PM2.5 concentrations, governments and communities can take timely action to mitigate their effects.

This project uses historical air quality and weather data to predict PM2.5 concentrations in Beijing. You will:

Preprocess sequential data to uncover patterns.
Design and train RNN or LSTM models to make accurate predictions.
Fine-tune the model and run several experiments. The goal is to have a Root Mean Squared Error below 4000 on the Leaderboard.
Submission Requirements
Submit a comprehensive report.

Introduction: Briefly describe the problem and your approach.
Data Exploration: Summarize your dataset analysis, including any preprocessing steps.
Model Design: Describe the architecture of your RNN/LSTM model that gave you the best performance and why you chose it.
Experiment Table: Include a table summarizing your experiments. The table must have the following columns. The parameters column should include more parameters, and not only the learning rate, as demonstrated.
Results: Discuss your model’s performance and any key findings.
Conclusion: Summarize your work and propose improvements or next steps.
Include the GitHub repo link in your report. Make sure your code is well documented.
Instructions for Joining the Kaggle Challenge
Your first graded assignment for the Machine Learning Techniques I course is hosted on Kaggle. Follow these steps to join the competition and submit your work:

Use this link to access the competition: Join HereLinks to an external site.

If you don’t already have a Kaggle account, sign up using your ALU email, as personal emails will not be accepted.

Click Join Competition and accept the rules.

Download the data files (train.csv, test.csv, and sample_submission.csv) from the Data tab on Kaggle.

Train.csv: train.csvDownload train.csv
Test.csv: test.csvDownload test.csv
Sample_submission.csv : sample_submission.csvDownload sample_submission.csv
Download the starter Notebook which you are required to modify such that you can meet all requirements on the rubric.
air_quality_forecasting_starter_code.ipynbDownload air_quality_forecasting_starter_code.ipynb
Train an RNN or LSTM model using train.csv and generate predictions for test.csv. Format your predictions according to sample_submission.csv

Submit your predictions on Kaggle by clicking Submit Predictions on the competition page. You can make up to 10 submissions per day

NB: Submissions with a similarity score above 50% or AI-generated work will not be graded! The same applies to Kaggle submissions with the exact scores on the leaderboard.
