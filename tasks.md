### **Phase 1: Project Foundation & Data Understanding**

#### **Task 1: Project Setup & Initial Data Inspection**
* **Description:** Initialize the project repository and perform a high-level inspection of the data to understand its basic properties.
* **Acceptance Criteria (✅ Done When):**
    * [ ] The professional folder structure has been created.
    * [ ] A Git repository is initialized with a `.gitignore` file.
    * [ ] The `notebooks/01_EDA_and_Preprocessing.ipynb` notebook is created.
    * [ ] The `train.csv` and `test.csv` files are loaded into the notebook.
    * [ ] The `.info()`, `.describe()`, and `.head()` methods are run, and their outputs are documented.
    * [ ] The `datetime` column is converted to the correct data type and set as the index.
* **Expected Outcome:** A fully initialized project folder and a notebook with the data loaded and initially inspected.

#### **Task 2: Exploratory Data Visualization & Analysis**
* **Description:** Conduct a deep visual analysis of the dataset to identify patterns, seasonality, correlations, and anomalies that will inform feature engineering and modeling.
* **Acceptance Criteria (✅ Done When):**
    * [ ] The `pm2.5` target variable is plotted over time to visualize trends.
    * [ ] The distribution of each numerical feature is visualized using histograms and box plots.
    * [ ] A correlation heatmap is generated and analyzed.
    * [ ] Missing data patterns are visualized.
    * [ ] Key findings from the visualizations are documented in markdown cells within the notebook.
* **Expected Outcome:** A notebook containing a comprehensive visual analysis of the dataset.

#### **Task 3: Create Core Preprocessing Functions**
* **Description:** Develop the basic, reusable functions for cleaning and preparing the data. These will be stored in a separate utility script.
* **Acceptance Criteria (✅ Done When):**
    * [ ] The `src/data_preprocessing.py` file is created.
    * [ ] A `load_data()` function is implemented.
    * [ ] A `handle_missing_values()` function is created based on the findings from the EDA.
    * [ ] A `create_time_features()` function is implemented to generate features from the datetime index.
    * [ ] A `scale_features()` function that uses `StandardScaler` is created.
* **Expected Outcome:** A Python script `src/data_preprocessing.py` with core data cleaning and feature engineering functions.

#### **Task 4: Implement Sequence Creation Function**
* **Description:** Develop the crucial "sliding window" function that transforms the time series data into a supervised learning format for the RNNs.
* **Acceptance Criteria (✅ Done When):**
    * [ ] The `create_sequences(data, n_past_steps, ...)` function is added to `src/data_preprocessing.py`.
    * [ ] The function correctly generates `X` (sequences of past data) and `y` (target values).
    * [ ] The function is thoroughly tested in the notebook to ensure its output shapes are correct.
* **Expected Outcome:** An updated `src/data_preprocessing.py` script containing the vital sequence creation logic.

---

### **Phase 2: Modeling & Experimentation**

#### **Task 5: Baseline LSTM Model (Experiment 1)**
* **Description:** Build, train, and document the baseline LSTM model. This serves as the starting point for all other experiments.
* **Acceptance Criteria (✅ Done When):**
    * [ ] The `notebooks/02_Baseline_LSTM.ipynb` is created.
    * [ ] The preprocessing functions are used to prepare the data with a 24-hour sequence length.
    * [ ] A time-based validation split is created.
    * [ ] The simple LSTM model is built, compiled, and trained with `EarlyStopping`.
    * [ ] The model's validation RMSE is calculated and the trained model is saved.
    * [ ] A submission file is generated.
    * [ ] **The `README.md` Experiment Table is updated with all details for Experiment 1.**
* **Expected Outcome:** A completed baseline notebook and the first entry in your experiment tracking table.

#### **Task 6: GRU & SARIMA Baselines (Experiments 2-3)**
* **Description:** Establish performance baselines for two alternative models: a GRU (another RNN) and a classical SARIMA model.
* **Acceptance Criteria (✅ Done When):**
    * [ ] A GRU model is built, trained, and evaluated in a new notebook, keeping other parameters identical to the LSTM baseline.
    * [ ] **The Experiment Table is updated with the results for Experiment 2 (GRU).**
    * [ ] A SARIMA model is built, fitted, and evaluated in a separate notebook.
    * [ ] **The Experiment Table is updated with the results for Experiment 3 (SARIMA).**
* **Expected Outcome:** Two new notebooks and two new, complete entries in the experiment table, providing strong comparison points.

#### **Task 7: Architecture Tuning Experiments (Experiments 4-9)**
* **Description:** Systematically experiment with different model architectures to find the most effective structure.
* **Acceptance Criteria (✅ Done When):**
    * [ ] Run an experiment by changing the sequence length (e.g., 48 hours). **Update table for Exp. 4.**
    * [ ] Run an experiment by stacking a second LSTM layer. **Update table for Exp. 5.**
    * [ ] Run an experiment by increasing the number of units (e.g., to 64). **Update table for Exp. 6.**
    * [ ] Run an experiment by adding `Dropout` layers for regularization. **Update table for Exp. 7.**
    * [ ] Run an experiment with a hybrid CNN-LSTM architecture. **Update table for Exp. 8.**
    * [ ] Run an experiment with a hybrid CNN-GRU architecture. **Update table for Exp. 9.**
* **Expected Outcome:** A notebook (`03_Advanced_Models.ipynb`) containing these experiments, with six new, fully documented rows in the experiment table.

#### **Task 8: Hyperparameter Tuning Experiments (Experiments 10-15+)**
* **Description:** Fine-tune the training process of the best-performing architecture found in the previous task.
* **Acceptance Criteria (✅ Done When):**
    * [ ] Run three experiments varying the learning rate (e.g., 0.005, 0.001, 0.0005). **Update table for Exps. 10-12.**
    * [ ] Run two experiments varying the batch size (e.g., 16, 64). **Update table for Exps. 13-14.**
    * [ ] Run an experiment with a different optimizer (e.g., `RMSprop`). **Update table for Exp. 15.**
    * [ ] (Optional but recommended) Run an automated search with Keras Tuner. **Update table for any additional experiments found.**
* **Expected Outcome:** A notebook detailing hyperparameter tuning and at least six new, complete entries in the experiment table, fulfilling the "15+ experiments" requirement.

---

### **Phase 3: Finalization**

#### **Task 9: Final Model Selection & Analysis**
* **Description:** Analyze the complete experiment table to select the single best model and summarize the key findings from the experimentation process.
* **Acceptance Criteria (✅ Done When):**
    * [ ] The experiment table is reviewed, and a champion model is selected based on validation RMSE and Kaggle score.
    * [ ] A brief written analysis in the `README.md` or a final notebook summarizes which changes had the most impact on performance.
    * [ ] A final, clean submission file is generated using the champion model.
* **Expected Outcome:** A final submission file and a clear, data-driven conclusion on the best model.

#### **Task 10: Final Project Documentation**
* **Description:** Complete the `README.md` file to create a professional, self-contained project summary.
* **Acceptance Criteria (✅ Done When):**
    * [ ] The `README.md` includes a final, polished version of the Experiment Table.
    * [ ] Sections for "Project Overview," "Setup Instructions," and "Summary of Results" are complete and well-written.
    * [ ] The code in all notebooks is clean, well-commented, and runnable.
* **Expected Outcome:** A polished, professional, and fully documented project repository.