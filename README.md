# Euro NCAP Safety Rating Prediction Project

This repository contains the end-to-end workflow used to collect Euro NCAP assessment data, convert it into a machine-learning-ready dataset, and train models to predict vehicle safety-star categories.

## Project Goal

The goal of this project is to build a predictive model that can classify vehicles into broad safety categories using Euro NCAP assessment metadata and safety equipment features.

The workflow in this project follows these major stages:

1. Data extraction from the Euro NCAP API
2. Raw JSON download and storage
3. Parsing and cleaning of assessment records
4. Feature engineering and target creation
5. Feature selection for model training
6. Model experimentation and evaluation
7. Saving the final trained model

---

## 1. Data Extraction

The first step starts with the API endpoints used to fetch the list of Euro NCAP assessments and the raw vehicle assessment JSON files.

### Step 1A — Extract assessment links

The notebook `ncap.ipynb` uses the Euro NCAP API to:

- call the assessment listing endpoint
- paginate through the results
- collect assessment IDs and metadata
- save them into `vehicle_links.json`

This produces a structured list of vehicle assessment links such as:

- assessment ID
- make
- model
- year
- assessment URL

### Step 1B — Download raw assessment JSON files

Using the saved vehicle links, the notebook downloads each assessment JSON file into the `test_data/` folder.

This creates a local dataset of raw Euro NCAP assessment responses for downstream processing.

### Step 1C — Optional direct download script

The file `euroncap.py` provides a smaller script that downloads a specific assessment JSON response from the API and saves it locally.

This script is useful when you want to fetch a specific assessment file quickly.

---

## 2. Data Parsing and Cleaning

After the raw files are collected, the notebook `ncap.ipynb` parses all JSON files and converts them into a tabular dataset.

### What happens in this stage

For every JSON assessment file, the project:

- opens the file
- safely reads nested values using helper functions
- maps injury colors to numeric values
- converts equipment presence into binary features
- extracts important metadata such as make, model, year, and star rating

### Important cleaning logic used

- Only valid assessment records are included.
- Missing or malformed values are handled safely.
- Injury color labels such as GREEN, YELLOW, ORANGE, BROWN, and RED are converted to numeric severity scores.
- Safety equipment values are converted to 1/0 indicators, where 1 means fitted as standard equipment.

This produces the raw structured dataset that is later saved to:

- `ncap_ml_ready.db`
- `euroncap_cardata_masked.csv`

---

## 3. Feature Engineering

The next stage creates the machine-learning features that will be used to train the model.

### Target variable

The original `stars` value from Euro NCAP is converted into a simpler categorical target:

- 1, 2, 3 stars → low
- 4 stars → medium
- 5 stars → high

This new target column is stored as `stars_category`.

### Protocol version feature

A helper function is used to map the protocol year into a simpler version bucket:

- 2019 and earlier → 1
- 2020 to 2021 → 2
- 2022 to 2025 → 3

This generated feature is stored as `protocol_version` and added to the model input features.

### Why this stage matters

The model is trained on a combination of:

- vehicle weight
- crash protection equipment
- child safety features
- active safety systems
- protocol version
- injury-related signals

These features are chosen because they represent both structural and safety-assist information that correlate with Euro NCAP outcomes.

---

## 4. Feature Selection

Feature selection in this project was done manually rather than using an automated selector.

### Initial feature list

The notebook defines a curated list of useful input columns such as:

- `kerb_weight`
- `seat1_knee_airbag`
- `row2_side_chest_airbag`
- `seat1_side_pelvis_airbag`
- `seat3_side_pelvis_airbag`
- `row2_side_pelvis_airbag`
- `seat1_centre_lateral_airbag`
- `seat3_centre_lateral_airbag`
- `seat3_isofix`
- `seat1_isize`
- `seat3_isize`
- `row2_isize`
- `row3_isize`
- `seat3_childdetection`
- `row2_childdetection`
- `has_active_bonnet`
- `has_distraction_detection`
- `has_fatigue_detection`
- `has_lane_assist`
- `has_speed_assist`
- `has_aeb_vru`
- `has_cyclistdoorprevention`
- `has_aeb_m2w`
- `has_aeb_cartocar`
- `protocol_version`

### Columns removed during selection

The following columns were dropped because they were not useful or were identifiers:

- `id`
- `make`
- `model`
- `year`
- `stars`
- `protocol_year`
- `seat_arrangement`
- `rating_year`
- many detailed airbag and seat-specific fields that were either redundant or not needed for the simplified model

This manual feature selection keeps the model focused on the most meaningful predictors for safety-star category prediction.

---

## 5. Train/Test Split

The dataset is split into training and testing sets using `train_test_split` from scikit-learn.

### Split configuration

- test size: 20%
- random state: 42
- stratify: `y`

This ensures that the class distribution of the target variable is preserved in both training and test sets.

The model is trained on `X_train` and evaluated on `X_test`.

---

## 6. Model Training Experiments

Several experiments were performed in the notebook `model.ipynb` to assess model behavior.

### 6A. Baseline Random Forest

The first experiment trains a basic Random Forest classifier.

It uses:

- `n_estimators=100`
- `random_state=42`

This baseline provides an initial measure of performance and helps identify whether the selected features are meaningful.

### 6B. Balanced Random Forest

A second version uses `class_weight='balanced'` to handle class imbalance more effectively.

This is important because safety-star categories are not perfectly evenly distributed in the dataset.

### 6C. Tuned Random Forest

Additional experiments were run with:

- `max_depth=10`
- `min_samples_split=2` or `5`
- `n_estimators=300`
- `class_weight='balanced'`

These variations help identify stronger settings for the classification task.

### 6D. Hyperparameter tuning with GridSearchCV

A grid search is also performed using `GridSearchCV` over several Random Forest parameters:

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `class_weight`

The scoring metric used is `f1_macro`, which is appropriate for multiclass performance evaluation.

### 6E. SMOTE-based training

To improve imbalance handling, the project also tries SMOTE (Synthetic Minority Over-sampling Technique):

- resample the training set
- train a Random Forest on the balanced training data
- evaluate on the untouched test set

This helps the model learn minority categories more effectively.

### 6F. XGBoost experiments

The project also tests XGBoost classifiers:

- simple XGBoost classifier
- XGBoost with class-balanced sample weights

These experiments compare the performance of boosting models against Random Forests.

---

## 7. Evaluation and Validation

Model performance is evaluated using:

- accuracy
- classification report
- feature importance ranking
- F1 macro score

The notebook also includes cross-validation using `StratifiedKFold` and `cross_val_score` to estimate model stability.

This gives a stronger indication of whether the model is generalizing well beyond one split.

---

## 8. Model Saving

The best-performing training pipeline is saved using `joblib` into:

- `euro_ncap_model.joblib`

This allows the trained model to be reused for predictions without retraining from scratch.

---

## 9. Repository Files

Key files in this project are:

- `euroncap.py` — simple API download helper
- `ncap.ipynb` — full data extraction and transformation workflow
- `model.ipynb` — model training, evaluation, and experiments
- `vehicle_links.json` — extracted assessment links
- `test_data/` — raw downloaded Euro NCAP assessment JSON files
- `euroncap_cardata_masked.csv` — cleaned ML-ready table
- `euro_ncap_model.joblib` — trained model artifact

---

## 10. Summary of the Full Workflow

The complete process used in this repository is:

1. Fetch assessment IDs and vehicle links from the Euro NCAP API.
2. Download raw assessment JSON files.
3. Parse the JSON files and extract safety-related features.
4. Convert star ratings into broad safety categories.
5. Engineer protocol and safety features for model input.
6. Manually select the most relevant features.
7. Split the dataset into train and test sets.
8. Train several Random Forest and XGBoost models.
9. Evaluate performance using classification metrics and cross-validation.
10. Save the final model for reuse.

This project combines web data collection, feature engineering, and machine learning into one structured workflow for Euro NCAP safety prediction.
