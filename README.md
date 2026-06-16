# Euro NCAP Safety Rating Prediction

Euro NCAP (European New Car Assessment Programme) is Europe's independent
vehicle safety testing body. It subjects new cars to standardised crash tests
across four domains — adult occupant protection, child occupant protection,
vulnerable road user (VRU) protection, and safety assist systems — and awards
a star rating from 1 to 5.

This project builds an end-to-end machine learning pipeline on Euro NCAP
assessment data to predict a vehicle's broad safety category from its crash
protection equipment and test outcomes. It also identifies which safety
features most strongly drive the difference between 4-star and 5-star vehicles.

---

## Key Results

| Model | F1 Macro | Accuracy |
|---|---|---|
| Baseline Random Forest | — | — |
| Balanced Random Forest (class_weight) | — | — |
| Tuned RF via GridSearchCV | — | — |
| SMOTE + Random Forest | — | — |
| XGBoost | — | — |
| XGBoost (balanced weights) | — | — |

> **Replace the dashes with your actual numbers before sharing this repo.**
> This table is the first thing a technical reviewer looks for.

**Top predictive features** *(from feature importance — fill in after training):*
- e.g. AEB fitment (`has_aeb_cartocar`)
- e.g. Side chest airbag presence (`row2_side_chest_airbag`)
- e.g. Protocol version bucket (`protocol_version`)

---

## Repository Structure

```
euroncap/
├── data/
│   └── combined_data.json          # Full parsed dataset, all assessments
├── models/
│   └── euro_ncap_model.joblib      # Saved best model pipeline
├── sample_data/                    # 3–5 raw API response examples (JSON)
├── 01_data_collection.ipynb        # API extraction and raw JSON download
├── 02_parsing_and_features.ipynb   # JSON → tabular dataset + feature engineering
├── 03_modelling.ipynb              # Model experiments, evaluation, feature importance
├── euroncap.py                     # Utility: fetch a single assessment by ID
└── README.md
```

> **Note on current state:** The repo is being reorganised. If you see
> `combined_data.json` or `euro_ncap_model.joblib` in the root, they belong
> in `data/` and `models/` respectively. The notebooks `ncap.ipynb` and
> `model.ipynb` correspond to `01_data_collection + 02_parsing` and
> `03_modelling` above.

---

## Data Source

Data is fetched from the Euro NCAP public API. Each vehicle assessment record
is a nested JSON object containing:

- **Per-seat injury colour ratings** — GREEN / YELLOW / ORANGE / RED across
  body regions (head, neck, chest, pelvis, femur, tibia) for each crash test
  type (frontal offset, full-width, side MDB, side pole, rear impact)
- **Safety equipment inventory** — airbag types and fitment positions,
  ISOFIX, i-Size, AEB systems, lane assist, speed assist, active bonnet
- **Normalised domain scores** — adult occupant, child occupant, VRU,
  safety assist (each expressed as a percentage of maximum)
- **Protocol year** — 2017, 2022, or 2026; each protocol uses different
  scoring thresholds and introduces new test scenarios

Raw sample files are in `sample_data/`. The full parsed and cleaned dataset
is at `data/combined_data.json`.

---

## Methodology

### Target Variable

Euro NCAP star ratings (1–5) are collapsed into three categories to reduce
class sparsity:

| Stars | Category |
|---|---|
| 1, 2, 3 | `low` |
| 4 | `medium` |
| 5 | `high` |

This is stored as the `stars_category` column.

### Key Design Decisions

**Protocol-aware colour encoding** — Injury colour ratings (GREEN → RED) are
kept as raw strings during parsing. Ordinal encoding is applied separately
per protocol year, because the same colour can represent a different severity
threshold across protocol generations. This prevents silent cross-protocol
contamination in model features.

**Protocol version bucketing** — Protocol year is mapped to three version
buckets rather than used as a raw integer:

| Protocol Year | Version Bucket |
|---|---|
| 2019 and earlier | 1 |
| 2020 – 2021 | 2 |
| 2022 and later | 3 |

**Leakage prevention** — Normalised domain scores (e.g. `adult_norm`) are
excluded from model features because they are direct components of the star
rating. Only equipment presence flags and protocol version are used as inputs.

**Dual DataFrame design** — Parsing produces two tables:
- `df_seats` — one row per (assessment × test type × seat position),
  preserving granular injury outcome data
- `df_scores` — one row per assessment, with aggregated scores used for
  modelling

### Features Used

| Feature | Type | Description |
|---|---|---|
| `kerb_weight` | Numeric | Vehicle kerb weight (kg) |
| `protocol_version` | Ordinal | Protocol generation bucket (1/2/3) |
| `has_aeb_cartocar` | Binary | AEB car-to-car fitted as standard |
| `has_aeb_vru` | Binary | AEB pedestrian/cyclist detection |
| `has_lane_assist` | Binary | Lane keep assist fitted |
| `has_speed_assist` | Binary | Speed assist system fitted |
| `has_active_bonnet` | Binary | Active bonnet for pedestrian protection |
| `has_fatigue_detection` | Binary | Driver fatigue monitoring |
| `has_distraction_detection` | Binary | Driver distraction monitoring |
| `seat1_side_chest_airbag` | Binary | Driver side chest airbag |
| `row2_side_chest_airbag` | Binary | Rear side chest airbag |
| `seat3_isofix` | Binary | Front passenger ISOFIX |
| `seat1_isize` | Binary | Driver seat i-Size |
| `row2_isize` | Binary | Rear seat i-Size |
| `seat3_childdetection` | Binary | Child presence detection |
| ... | | *(see notebook for full list)* |

### Models Trained

All models are evaluated on a stratified 20% holdout set (random state 42)
and validated with StratifiedKFold cross-validation (5 folds) to assess
generalisation stability.

1. **Baseline Random Forest** — 100 estimators, default settings
2. **Balanced Random Forest** — `class_weight='balanced'` to handle uneven
   class distribution
3. **Tuned Random Forest** — GridSearchCV over `n_estimators`, `max_depth`,
   `min_samples_split`, `class_weight`
4. **SMOTE + Random Forest** — synthetic oversampling of minority classes
   applied to training set only; test set untouched
5. **XGBoost** — baseline gradient boosting classifier
6. **XGBoost with balanced sample weights** — computed per class to handle
   imbalance

Evaluation metrics: accuracy, F1 macro, classification report per class,
feature importance ranking.

---

## Reproduction

### Requirements

```bash
pip install pandas scikit-learn xgboost imbalanced-learn joblib \
            beautifulsoup4 notebook
```

### Run Order

```bash
# Step 1 — (Optional) Re-fetch raw assessment data from Euro NCAP API
jupyter notebook 01_data_collection.ipynb

# Step 2 — Parse raw JSONs into tabular dataset and engineer features
jupyter notebook 02_parsing_and_features.ipynb

# Step 3 — Train models and evaluate
jupyter notebook 03_modelling.ipynb
```

The notebooks are designed to run in sequence. If you want to go straight
to the modelling, skip to step 3 — `data/combined_data.json` is already
included in the repository.

### Load the Saved Model

```python
import joblib

model = joblib.load("models/euro_ncap_model.joblib")
prediction = model.predict(X_new)
```

---

## Limitations and Known Issues

- **Sample size** — approximately 300–500 vehicles have full `assessmentData`
  fields in the API. This is workable but limits model complexity; avoid
  deep architectures or large feature spaces.
- **Protocol comparability** — pre-2020 and post-2022 records use different
  test scenarios and scoring thresholds. Protocol version bucketing partially
  addresses this but cross-protocol comparison remains an approximation.
- **Class collapse** — collapsing 5 stars into 3 categories loses
  granularity. A regression on raw normalised scores is a natural extension
  of this work.
- **Equipment-only features** — the current feature set uses equipment
  presence flags rather than granular injury outcome scores, to avoid
  leakage. Adding carefully encoded seat-level injury features is a planned
  improvement.

---

## Background and Motivation

This project is part of a broader effort to build open-source Python tooling
for major road safety datasets — including Euro NCAP, FARS (US), GIDAS
(Germany), and RASSI (India). The author works in automotive crash safety
analytics at Maruti Suzuki India and brings domain context to feature
engineering decisions, particularly around injury biomechanics, protocol
versioning, and what the colour-coded ATD (dummy) ratings represent in terms
of real-world injury risk.

The domain framing matters here: this is not a generic classification
exercise. The feature selection, target collapse, and protocol handling
decisions are all grounded in how Euro NCAP scoring actually works, not just
what maximises CV accuracy.

---

## Author

**Utkarsh Tyagi**  
Data Analyst — Road Safety & Crash Analytics  
Maruti Suzuki India Limited  
[GitHub](https://github.com/tyagiut1232002-spec)
