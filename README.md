# Predictive Modeling of Water Quality for Potability

A machine-learning project that predicts whether a water sample is potable using physicochemical measurements such as pH, hardness, solids, chloramines, sulfate, conductivity, organic carbon, trihalomethanes, and turbidity.

This repository packages the original notebook, dataset, project report, and a reproducible training script so the project can be reviewed outside Google Drive.

## Highlights

- Cleaned missing values with mean imputation
- Performed exploratory data analysis using correlations, boxplots, and class-balance checks
- Trained a Random Forest classifier for binary potability prediction
- Evaluated with accuracy, ROC AUC, confusion matrix, classification report, and 5-fold stratified cross-validation
- Includes the original report PDF for project validation

## Dataset

The dataset contains 3,276 water-quality records and 10 columns:

- `ph`
- `Hardness`
- `Solids`
- `Chloramines`
- `Sulfate`
- `Conductivity`
- `Organic_carbon`
- `Trihalomethanes`
- `Turbidity`
- `Potability`

`Potability` is the target label:

- `0` = not potable
- `1` = potable

## Model

The reproducible script uses this pipeline:

1. Mean imputation for missing numerical values
2. Standard scaling
3. Random Forest classifier with 200 trees
4. Stratified train/test split and 5-fold cross-validation

Latest verified run:

```text
Dataset shape: (3276, 10)
Accuracy: 0.6524
ROC AUC: 0.6519
Mean 5-fold CV accuracy: 0.6728
```

## Repository Structure

```text
data/water_potability.csv                  Dataset
notebooks/water_potability_modeling.ipynb  Original notebook workflow
reports/water_potability_report.pdf        Project report PDF
src/train_model.py                         Reproducible training/evaluation script
requirements.txt                           Python dependencies
```

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train_model.py
```

To open the notebook:

```bash
jupyter notebook notebooks/water_potability_modeling.ipynb
```

## Project Context

Built as a machine-learning course project for water-potability prediction. The goal was not only to train a classifier, but also to show the end-to-end workflow: data inspection, preprocessing, EDA, model training, evaluation, and reporting.

## Original Drive Evidence

The original project files were maintained in Google Drive before being packaged here for public review. Drive folder:

https://drive.google.com/drive/folders/1Rj5vq-3NuAkW53cxUwKwPBFoZkCUIBtY
