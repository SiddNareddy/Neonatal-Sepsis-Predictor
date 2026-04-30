# Unified Sepsis Modeling Summary

This report combines baseline-model and CNN-Attention-BiLSTM runs.

## Top 15 Overall (ranked by test AUPRC)

| family     |   horizon | channel_set   | model_display         |   test_auprc |   test_auroc |   test_balanced_accuracy |   test_recall_sensitivity |   test_specificity |   test_f1 |   test_brier |   val_threshold |
|:-----------|----------:|:--------------|:----------------------|-------------:|-------------:|-------------------------:|--------------------------:|-------------------:|----------:|-------------:|----------------:|
| cnn_bdlstm |        24 | full          | CNN-Attn-BiLSTM       |       0.8671 |       0.7838 |                   0.615  |                    0.9471 |             0.283  |    0.7912 |       0.1857 |          0.2237 |
| baseline   |        24 | values_masks  | ExtraTrees_balanced   |       0.8597 |       0.7625 |                   0.5979 |                    0.9882 |             0.2075 |    0.7962 |       0.1934 |          0.325  |
| baseline   |        24 | full          | RandomForest_balanced |       0.8577 |       0.7609 |                   0.5873 |                    0.9294 |             0.2453 |    0.7745 |       0.1924 |          0.36   |
| baseline   |        12 | full          | HistGradBoost         |       0.8537 |       0.7671 |                   0.5832 |                    0.9588 |             0.2075 |    0.7818 |       0.195  |          0.2093 |
| cnn_bdlstm |        18 | full          | CNN-Attn-BiLSTM       |       0.8536 |       0.769  |                   0.6174 |                    0.9706 |             0.2642 |    0.799  |       0.1987 |          0.1579 |
| baseline   |        24 | full          | HistGradBoost         |       0.852  |       0.7661 |                   0.6529 |                    0.8529 |             0.4528 |    0.7775 |       0.1918 |          0.3949 |
| baseline   |        24 | full          | ExtraTrees_balanced   |       0.8518 |       0.7633 |                   0.6062 |                    0.9765 |             0.2358 |    0.7962 |       0.1921 |          0.2925 |
| baseline   |        18 | full          | RandomForest_balanced |       0.8483 |       0.7576 |                   0.5873 |                    0.9294 |             0.2453 |    0.7745 |       0.194  |          0.3675 |
| baseline   |        24 | full          | MLP_tabular           |       0.8475 |       0.7542 |                   0.5767 |                    0.9647 |             0.1887 |    0.781  |       0.1988 |          0.2159 |
| baseline   |        12 | full          | MLP_tabular           |       0.8467 |       0.769  |                   0.641  |                    0.9235 |             0.3585 |    0.7949 |       0.1895 |          0.3489 |
| baseline   |        18 | full          | HistGradBoost         |       0.8464 |       0.7447 |                   0.6388 |                    0.7588 |             0.5189 |    0.7371 |       0.2008 |          0.4552 |
| baseline   |        24 | values        | ExtraTrees_balanced   |       0.8453 |       0.7371 |                   0.5703 |                    0.9235 |             0.217  |    0.7659 |       0.2016 |          0.38   |
| cnn_bdlstm |         6 | full          | CNN-Attn-BiLSTM       |       0.8407 |       0.7593 |                   0.5703 |                    0.9235 |             0.217  |    0.7659 |       0.2005 |          0.1895 |
| baseline   |        18 | full          | ExtraTrees_balanced   |       0.8405 |       0.7536 |                   0.6251 |                    0.9294 |             0.3208 |    0.79   |       0.1937 |          0.3925 |
| cnn_bdlstm |        24 | values_masks  | CNN-Attn-BiLSTM       |       0.838  |       0.7356 |                   0.6323 |                    0.8118 |             0.4528 |    0.7541 |       0.2238 |          0.2913 |

## Best Model Per Horizon (overall)

| family     |   horizon | channel_set   | model_display   |   test_auprc |   test_auroc |   test_balanced_accuracy |   test_recall_sensitivity |   test_specificity |   test_f1 |   test_brier |   val_threshold |
|:-----------|----------:|:--------------|:----------------|-------------:|-------------:|-------------------------:|--------------------------:|-------------------:|----------:|-------------:|----------------:|
| cnn_bdlstm |         6 | full          | CNN-Attn-BiLSTM |       0.8407 |       0.7593 |                   0.5703 |                    0.9235 |             0.217  |    0.7659 |       0.2005 |          0.1895 |
| baseline   |        12 | full          | HistGradBoost   |       0.8537 |       0.7671 |                   0.5832 |                    0.9588 |             0.2075 |    0.7818 |       0.195  |          0.2093 |
| cnn_bdlstm |        18 | full          | CNN-Attn-BiLSTM |       0.8536 |       0.769  |                   0.6174 |                    0.9706 |             0.2642 |    0.799  |       0.1987 |          0.1579 |
| cnn_bdlstm |        24 | full          | CNN-Attn-BiLSTM |       0.8671 |       0.7838 |                   0.615  |                    0.9471 |             0.283  |    0.7912 |       0.1857 |          0.2237 |

## Best Per Family and Horizon

| family     |   horizon | channel_set   | model_display         |   test_auprc |   test_auroc |   test_balanced_accuracy |   test_recall_sensitivity |   test_specificity |   test_f1 |   test_brier |   val_threshold |
|:-----------|----------:|:--------------|:----------------------|-------------:|-------------:|-------------------------:|--------------------------:|-------------------:|----------:|-------------:|----------------:|
| baseline   |         6 | full          | MLP_tabular           |       0.8244 |       0.7224 |                   0.598  |                    0.8941 |             0.3019 |    0.7677 |       0.2078 |          0.3936 |
| cnn_bdlstm |         6 | full          | CNN-Attn-BiLSTM       |       0.8407 |       0.7593 |                   0.5703 |                    0.9235 |             0.217  |    0.7659 |       0.2005 |          0.1895 |
| baseline   |        12 | full          | HistGradBoost         |       0.8537 |       0.7671 |                   0.5832 |                    0.9588 |             0.2075 |    0.7818 |       0.195  |          0.2093 |
| cnn_bdlstm |        12 | full          | CNN-Attn-BiLSTM       |       0.832  |       0.7413 |                   0.6028 |                    0.8471 |             0.3585 |    0.7539 |       0.1984 |          0.3935 |
| baseline   |        18 | full          | RandomForest_balanced |       0.8483 |       0.7576 |                   0.5873 |                    0.9294 |             0.2453 |    0.7745 |       0.194  |          0.3675 |
| cnn_bdlstm |        18 | full          | CNN-Attn-BiLSTM       |       0.8536 |       0.769  |                   0.6174 |                    0.9706 |             0.2642 |    0.799  |       0.1987 |          0.1579 |
| baseline   |        24 | values_masks  | ExtraTrees_balanced   |       0.8597 |       0.7625 |                   0.5979 |                    0.9882 |             0.2075 |    0.7962 |       0.1934 |          0.325  |
| cnn_bdlstm |        24 | full          | CNN-Attn-BiLSTM       |       0.8671 |       0.7838 |                   0.615  |                    0.9471 |             0.283  |    0.7912 |       0.1857 |          0.2237 |

## Best Baseline Per Horizon

| family   |   horizon | channel_set   | model_display         |   test_auprc |   test_auroc |   test_balanced_accuracy |   test_recall_sensitivity |   test_specificity |   test_f1 |   test_brier |   val_threshold |
|:---------|----------:|:--------------|:----------------------|-------------:|-------------:|-------------------------:|--------------------------:|-------------------:|----------:|-------------:|----------------:|
| baseline |         6 | full          | MLP_tabular           |       0.8244 |       0.7224 |                   0.598  |                    0.8941 |             0.3019 |    0.7677 |       0.2078 |          0.3936 |
| baseline |        12 | full          | HistGradBoost         |       0.8537 |       0.7671 |                   0.5832 |                    0.9588 |             0.2075 |    0.7818 |       0.195  |          0.2093 |
| baseline |        18 | full          | RandomForest_balanced |       0.8483 |       0.7576 |                   0.5873 |                    0.9294 |             0.2453 |    0.7745 |       0.194  |          0.3675 |
| baseline |        24 | values_masks  | ExtraTrees_balanced   |       0.8597 |       0.7625 |                   0.5979 |                    0.9882 |             0.2075 |    0.7962 |       0.1934 |          0.325  |

## Best CNN Per Horizon

| family     |   horizon | channel_set   | model_display   |   test_auprc |   test_auroc |   test_balanced_accuracy |   test_recall_sensitivity |   test_specificity |   test_f1 |   test_brier |   val_threshold |
|:-----------|----------:|:--------------|:----------------|-------------:|-------------:|-------------------------:|--------------------------:|-------------------:|----------:|-------------:|----------------:|
| cnn_bdlstm |         6 | full          | CNN-Attn-BiLSTM |       0.8407 |       0.7593 |                   0.5703 |                    0.9235 |             0.217  |    0.7659 |       0.2005 |          0.1895 |
| cnn_bdlstm |        12 | full          | CNN-Attn-BiLSTM |       0.832  |       0.7413 |                   0.6028 |                    0.8471 |             0.3585 |    0.7539 |       0.1984 |          0.3935 |
| cnn_bdlstm |        18 | full          | CNN-Attn-BiLSTM |       0.8536 |       0.769  |                   0.6174 |                    0.9706 |             0.2642 |    0.799  |       0.1987 |          0.1579 |
| cnn_bdlstm |        24 | full          | CNN-Attn-BiLSTM |       0.8671 |       0.7838 |                   0.615  |                    0.9471 |             0.283  |    0.7912 |       0.1857 |          0.2237 |

## Plot Assets

- Combined plots: `/Users/sidd/Desktop/biol1595/final-project/code/results/unified_summary/plots`

Important generated figures:
- `family_best_test_auprc_by_horizon.png`
- `family_best_test_auroc_by_horizon.png`
- `family_best_test_balanced_accuracy_by_horizon.png`
- `family_best_sens_spec.png`
- `<family>_test_auprc_heatmap.png`
- `<family>_test_auroc_heatmap.png`
- `<family>_test_balanced_accuracy_heatmap.png`
