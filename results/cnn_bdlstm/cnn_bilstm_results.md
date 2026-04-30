# CNN-Attention-BiLSTM Results

## Top configurations by test AUPRC

|   horizon | channel_set   | model           |   test_auroc |   test_auprc |   test_balanced_accuracy |   test_recall_sensitivity |   test_specificity |   test_f1 |   test_brier |   val_threshold |   epochs_run |   n_params |   train_seconds |
|----------:|:--------------|:----------------|-------------:|-------------:|-------------------------:|--------------------------:|-------------------:|----------:|-------------:|----------------:|-------------:|-----------:|----------------:|
|        24 | full          | CNN-Attn-BiLSTM |       0.7838 |       0.8671 |                   0.615  |                    0.9471 |             0.283  |    0.7912 |       0.1857 |          0.2237 |           25 |     517570 |           5.985 |
|        18 | full          | CNN-Attn-BiLSTM |       0.769  |       0.8536 |                   0.6174 |                    0.9706 |             0.2642 |    0.799  |       0.1987 |          0.1579 |           28 |     517570 |           6.695 |
|         6 | full          | CNN-Attn-BiLSTM |       0.7593 |       0.8407 |                   0.5703 |                    0.9235 |             0.217  |    0.7659 |       0.2005 |          0.1895 |           19 |     517570 |           4.412 |
|        24 | values_masks  | CNN-Attn-BiLSTM |       0.7356 |       0.838  |                   0.6323 |                    0.8118 |             0.4528 |    0.7541 |       0.2238 |          0.2913 |           30 |     515522 |           7.321 |
|        24 | values        | CNN-Attn-BiLSTM |       0.7393 |       0.8329 |                   0.6387 |                    0.8529 |             0.4245 |    0.7713 |       0.2046 |          0.3628 |           26 |     514242 |           6.204 |
|        12 | full          | CNN-Attn-BiLSTM |       0.7413 |       0.832  |                   0.6028 |                    0.8471 |             0.3585 |    0.7539 |       0.1984 |          0.3935 |           21 |     517570 |           4.924 |
|        18 | values        | CNN-Attn-BiLSTM |       0.7093 |       0.8129 |                   0.5703 |                    0.9235 |             0.217  |    0.7659 |       0.2192 |          0.2631 |           29 |     514242 |           6.9   |
|        18 | values_masks  | CNN-Attn-BiLSTM |       0.6954 |       0.8085 |                   0.5892 |                    0.8765 |             0.3019 |    0.7583 |       0.2193 |          0.3215 |           21 |     515522 |           4.979 |
|         6 | values_masks  | CNN-Attn-BiLSTM |       0.6952 |       0.8007 |                   0.5602 |                    0.9412 |             0.1792 |    0.7674 |       0.215  |          0.3469 |           20 |     515522 |           4.658 |
|        12 | values_masks  | CNN-Attn-BiLSTM |       0.7086 |       0.798  |                   0.6063 |                    0.8824 |             0.3302 |    0.7673 |       0.2162 |          0.3365 |           21 |     515522 |           4.93  |

## Horizon 6h, channel set: `full`

| model           |   test_auroc |   test_auprc |   test_balanced_accuracy |   test_recall_sensitivity |   test_specificity |   test_f1 |   test_brier |   val_threshold |   epochs_run |   n_params |   train_seconds |
|:----------------|-------------:|-------------:|-------------------------:|--------------------------:|-------------------:|----------:|-------------:|----------------:|-------------:|-----------:|----------------:|
| CNN-Attn-BiLSTM |       0.7593 |       0.8407 |                   0.5703 |                    0.9235 |              0.217 |    0.7659 |       0.2005 |          0.1895 |           19 |     517570 |           4.412 |

## Horizon 6h, channel set: `values`

| model           |   test_auroc |   test_auprc |   test_balanced_accuracy |   test_recall_sensitivity |   test_specificity |   test_f1 |   test_brier |   val_threshold |   epochs_run |   n_params |   train_seconds |
|:----------------|-------------:|-------------:|-------------------------:|--------------------------:|-------------------:|----------:|-------------:|----------------:|-------------:|-----------:|----------------:|
| CNN-Attn-BiLSTM |       0.6519 |       0.7643 |                      0.5 |                         1 |                  0 |    0.7623 |       0.2551 |          0.0104 |           33 |     514242 |           9.625 |

## Horizon 6h, channel set: `values_masks`

| model           |   test_auroc |   test_auprc |   test_balanced_accuracy |   test_recall_sensitivity |   test_specificity |   test_f1 |   test_brier |   val_threshold |   epochs_run |   n_params |   train_seconds |
|:----------------|-------------:|-------------:|-------------------------:|--------------------------:|-------------------:|----------:|-------------:|----------------:|-------------:|-----------:|----------------:|
| CNN-Attn-BiLSTM |       0.6952 |       0.8007 |                   0.5602 |                    0.9412 |             0.1792 |    0.7674 |        0.215 |          0.3469 |           20 |     515522 |           4.658 |

## Horizon 12h, channel set: `full`

| model           |   test_auroc |   test_auprc |   test_balanced_accuracy |   test_recall_sensitivity |   test_specificity |   test_f1 |   test_brier |   val_threshold |   epochs_run |   n_params |   train_seconds |
|:----------------|-------------:|-------------:|-------------------------:|--------------------------:|-------------------:|----------:|-------------:|----------------:|-------------:|-----------:|----------------:|
| CNN-Attn-BiLSTM |       0.7413 |        0.832 |                   0.6028 |                    0.8471 |             0.3585 |    0.7539 |       0.1984 |          0.3935 |           21 |     517570 |           4.924 |

## Horizon 12h, channel set: `values`

| model           |   test_auroc |   test_auprc |   test_balanced_accuracy |   test_recall_sensitivity |   test_specificity |   test_f1 |   test_brier |   val_threshold |   epochs_run |   n_params |   train_seconds |
|:----------------|-------------:|-------------:|-------------------------:|--------------------------:|-------------------:|----------:|-------------:|----------------:|-------------:|-----------:|----------------:|
| CNN-Attn-BiLSTM |       0.6968 |       0.7897 |                   0.5661 |                    0.9059 |             0.2264 |    0.7586 |       0.2209 |          0.2957 |           24 |     514242 |            5.65 |

## Horizon 12h, channel set: `values_masks`

| model           |   test_auroc |   test_auprc |   test_balanced_accuracy |   test_recall_sensitivity |   test_specificity |   test_f1 |   test_brier |   val_threshold |   epochs_run |   n_params |   train_seconds |
|:----------------|-------------:|-------------:|-------------------------:|--------------------------:|-------------------:|----------:|-------------:|----------------:|-------------:|-----------:|----------------:|
| CNN-Attn-BiLSTM |       0.7086 |        0.798 |                   0.6063 |                    0.8824 |             0.3302 |    0.7673 |       0.2162 |          0.3365 |           21 |     515522 |            4.93 |

## Horizon 18h, channel set: `full`

| model           |   test_auroc |   test_auprc |   test_balanced_accuracy |   test_recall_sensitivity |   test_specificity |   test_f1 |   test_brier |   val_threshold |   epochs_run |   n_params |   train_seconds |
|:----------------|-------------:|-------------:|-------------------------:|--------------------------:|-------------------:|----------:|-------------:|----------------:|-------------:|-----------:|----------------:|
| CNN-Attn-BiLSTM |        0.769 |       0.8536 |                   0.6174 |                    0.9706 |             0.2642 |     0.799 |       0.1987 |          0.1579 |           28 |     517570 |           6.695 |

## Horizon 18h, channel set: `values`

| model           |   test_auroc |   test_auprc |   test_balanced_accuracy |   test_recall_sensitivity |   test_specificity |   test_f1 |   test_brier |   val_threshold |   epochs_run |   n_params |   train_seconds |
|:----------------|-------------:|-------------:|-------------------------:|--------------------------:|-------------------:|----------:|-------------:|----------------:|-------------:|-----------:|----------------:|
| CNN-Attn-BiLSTM |       0.7093 |       0.8129 |                   0.5703 |                    0.9235 |              0.217 |    0.7659 |       0.2192 |          0.2631 |           29 |     514242 |             6.9 |

## Horizon 18h, channel set: `values_masks`

| model           |   test_auroc |   test_auprc |   test_balanced_accuracy |   test_recall_sensitivity |   test_specificity |   test_f1 |   test_brier |   val_threshold |   epochs_run |   n_params |   train_seconds |
|:----------------|-------------:|-------------:|-------------------------:|--------------------------:|-------------------:|----------:|-------------:|----------------:|-------------:|-----------:|----------------:|
| CNN-Attn-BiLSTM |       0.6954 |       0.8085 |                   0.5892 |                    0.8765 |             0.3019 |    0.7583 |       0.2193 |          0.3215 |           21 |     515522 |           4.979 |

## Horizon 24h, channel set: `full`

| model           |   test_auroc |   test_auprc |   test_balanced_accuracy |   test_recall_sensitivity |   test_specificity |   test_f1 |   test_brier |   val_threshold |   epochs_run |   n_params |   train_seconds |
|:----------------|-------------:|-------------:|-------------------------:|--------------------------:|-------------------:|----------:|-------------:|----------------:|-------------:|-----------:|----------------:|
| CNN-Attn-BiLSTM |       0.7838 |       0.8671 |                    0.615 |                    0.9471 |              0.283 |    0.7912 |       0.1857 |          0.2237 |           25 |     517570 |           5.985 |

## Horizon 24h, channel set: `values`

| model           |   test_auroc |   test_auprc |   test_balanced_accuracy |   test_recall_sensitivity |   test_specificity |   test_f1 |   test_brier |   val_threshold |   epochs_run |   n_params |   train_seconds |
|:----------------|-------------:|-------------:|-------------------------:|--------------------------:|-------------------:|----------:|-------------:|----------------:|-------------:|-----------:|----------------:|
| CNN-Attn-BiLSTM |       0.7393 |       0.8329 |                   0.6387 |                    0.8529 |             0.4245 |    0.7713 |       0.2046 |          0.3628 |           26 |     514242 |           6.204 |

## Horizon 24h, channel set: `values_masks`

| model           |   test_auroc |   test_auprc |   test_balanced_accuracy |   test_recall_sensitivity |   test_specificity |   test_f1 |   test_brier |   val_threshold |   epochs_run |   n_params |   train_seconds |
|:----------------|-------------:|-------------:|-------------------------:|--------------------------:|-------------------:|----------:|-------------:|----------------:|-------------:|-----------:|----------------:|
| CNN-Attn-BiLSTM |       0.7356 |        0.838 |                   0.6323 |                    0.8118 |             0.4528 |    0.7541 |       0.2238 |          0.2913 |           30 |     515522 |           7.321 |
