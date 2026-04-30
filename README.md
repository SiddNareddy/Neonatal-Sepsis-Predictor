# Neonatal Sepsis Prediction

This project builds and evaluates models for predicting neonatal sepsis from MIMIC-III physiological time-series data. The workflow creates a cleaned admission-level tensor, evaluates classical baseline models across different observation windows, trains a CNN-BiLSTM model, and combines results into a unified report.

## Key Files

- `code/Admission Sepsis Prediction.ipynb` creates the neonatal sepsis/control cohort, extracts and cleans vital-sign chart events, builds the 24-hour admission-level tensor, and saves the processed train/validation/test dataset. It also performs basic cohort and data-density checks.

- `code/run_baselines.py` trains classical baseline models such as logistic regression, random forests, extra trees, gradient boosting, and MLPs across multiple time horizons and channel-set ablations. It saves metrics, predictions, tables, and plots for comparing baseline performance.

- `code/train_cnn_bilstm.py` trains the CNN-Attention-BiLSTM model on the processed tensor for the same time horizons and channel sets. It uses weighted loss, early stopping, validation threshold selection, checkpoints, predictions, and training/metric visualizations.

- `code/unified_results_report.py` combines the baseline and CNN-BiLSTM result files into one unified summary. It generates clean comparison tables, combined plots, and a one-page Markdown/HTML report under `results/unified_summary/`.
