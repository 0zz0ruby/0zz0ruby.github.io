# Credit Card Customer Segmentation

Reproducible Python workflow for segmenting 8,950 credit card customers using
PCA, K-means clustering, statistical validation, and Monte Carlo simulation.

For the project narrative, business interpretation, and selected figures, see
the [portfolio project page](../../projects/credit-card-customer-segmentation.md).

## Repository contents

```text
credit-card-customer-segmentation/
|-- data/
|   `-- README.md
|-- outputs/
|   `-- tables/
|-- src/
|   |-- 01_data_preprocessing.py
|   |-- 02_pca_factor_clustering.py
|   |-- 03_validation_visualization.py
|   |-- 04_simulation_study.py
|   `-- run_all.py
|-- README.md
`-- requirements.txt
```

## Setup

1. Download `CC GENERAL.csv` from the
   [Kaggle dataset page](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata).
2. Place the file at `data/CC GENERAL.csv`.
3. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

4. Run the workflow from this directory:

```bash
python src/run_all.py
```

The scripts generate cleaned intermediate data, tables, summaries, and figures.

## Workflow

- `01_data_preprocessing.py`: data audit, median imputation, descriptive
  statistics, Z-score standardization, and exploratory figures.
- `02_pca_factor_clustering.py`: PCA, exploratory factor analysis, K-means
  model comparison, customer profiling, and clustering figures.
- `03_validation_visualization.py`: internal clustering metrics,
  Kruskal-Wallis tests, LDA separability analysis, and final visual checks.
- `04_simulation_study.py`: 240-run simulation study across sample sizes and
  noise levels using ARI, NMI, silhouette, CH, and DB metrics.

## Key results

- Five principal components retained 70.13% of total variance.
- Four customer segments were selected by balancing internal metrics and
  business interpretability.
- Ten key behavioral variables differed significantly across segments.
- LDA achieved 93.62% mean five-fold cross-validation accuracy when predicting
  the derived cluster labels.
- In the simulation study, mean ARI declined from approximately 0.998 at the
  lowest noise level to approximately 0.680 at the highest noise level.

## Data and attribution

The original dataset is not redistributed in this repository. It is available
from Kaggle under the title *Credit Card Dataset for Clustering* and is marked
CC0 on the dataset page.

The LDA result measures separability of model-derived cluster labels; it is not
accuracy against externally observed customer segments or default outcomes.
