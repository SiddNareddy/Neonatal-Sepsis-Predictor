# Baseline Results

## Top 10 overall by test AUPRC


| horizon | channel_set  | model                 | test_auroc | test_auprc | test_balanced_accuracy | test_recall_sensitivity | test_specificity | test_f1 | test_brier | val_threshold | train_seconds |
| ------- | ------------ | --------------------- | ---------- | ---------- | ---------------------- | ----------------------- | ---------------- | ------- | ---------- | ------------- | ------------- |
| 24      | values_masks | ExtraTrees_balanced   | 0.7625     | 0.8597     | 0.5979                 | 0.9882                  | 0.2075           | 0.7962  | 0.1934     | 0.325         | 0.811         |
| 24      | full         | RandomForest_balanced | 0.7609     | 0.8577     | 0.5873                 | 0.9294                  | 0.2453           | 0.7745  | 0.1924     | 0.36          | 1.352         |
| 12      | full         | HistGradBoost         | 0.7671     | 0.8537     | 0.5832                 | 0.9588                  | 0.2075           | 0.7818  | 0.195      | 0.2093        | 0.547         |
| 24      | full         | HistGradBoost         | 0.7661     | 0.852      | 0.6529                 | 0.8529                  | 0.4528           | 0.7775  | 0.1918     | 0.3949        | 0.843         |
| 24      | full         | ExtraTrees_balanced   | 0.7633     | 0.8518     | 0.6062                 | 0.9765                  | 0.2358           | 0.7962  | 0.1921     | 0.2925        | 0.968         |
| 18      | full         | RandomForest_balanced | 0.7576     | 0.8483     | 0.5873                 | 0.9294                  | 0.2453           | 0.7745  | 0.194      | 0.3675        | 1.212         |
| 24      | full         | MLP_tabular           | 0.7542     | 0.8475     | 0.5767                 | 0.9647                  | 0.1887           | 0.781   | 0.1988     | 0.2159        | 0.579         |
| 12      | full         | MLP_tabular           | 0.769      | 0.8467     | 0.641                  | 0.9235                  | 0.3585           | 0.7949  | 0.1895     | 0.3489        | 0.483         |
| 18      | full         | HistGradBoost         | 0.7447     | 0.8464     | 0.6388                 | 0.7588                  | 0.5189           | 0.7371  | 0.2008     | 0.4552        | 0.563         |
| 24      | values       | ExtraTrees_balanced   | 0.7371     | 0.8453     | 0.5703                 | 0.9235                  | 0.217            | 0.7659  | 0.2016     | 0.38          | 0.756         |


## Horizon 6h, channel set: `full`


| model                 | test_auroc | test_auprc | test_balanced_accuracy | test_recall_sensitivity | test_specificity | test_f1 | test_brier | val_threshold | train_seconds |
| --------------------- | ---------- | ---------- | ---------------------- | ----------------------- | ---------------- | ------- | ---------- | ------------- | ------------- |
| MLP_tabular           | 0.7224     | 0.8244     | 0.598                  | 0.8941                  | 0.3019           | 0.7677  | 0.2078     | 0.3936        | 0.362         |
| RandomForest_balanced | 0.7416     | 0.8244     | 0.6128                 | 0.8765                  | 0.3491           | 0.768   | 0.1985     | 0.3975        | 1.115         |
| HistGradBoost         | 0.7308     | 0.8208     | 0.6063                 | 0.8824                  | 0.3302           | 0.7673  | 0.2093     | 0.2966        | 0.281         |
| LogReg_L2_balanced    | 0.7378     | 0.816      | 0.6156                 | 0.9294                  | 0.3019           | 0.7861  | 0.2113     | 0.2421        | 0.066         |
| ExtraTrees_balanced   | 0.7216     | 0.8136     | 0.5519                 | 0.9529                  | 0.1509           | 0.7678  | 0.2057     | 0.3           | 0.799         |
| Dummy_StratifiedFloor | 0.4781     | 0.6059     | 0.5                    | 1                       | 0                | 0.7623  | 0.4964     | 0             | 0             |


## Horizon 6h, channel set: `values`


| model                 | test_auroc | test_auprc | test_balanced_accuracy | test_recall_sensitivity | test_specificity | test_f1 | test_brier | val_threshold | train_seconds |
| --------------------- | ---------- | ---------- | ---------------------- | ----------------------- | ---------------- | ------- | ---------- | ------------- | ------------- |
| RandomForest_balanced | 0.6968     | 0.8029     | 0.5184                 | 0.9235                  | 0.1132           | 0.7458  | 0.2113     | 0.375         | 1.359         |
| ExtraTrees_balanced   | 0.7161     | 0.7997     | 0.4977                 | 0.9765                  | 0.0189           | 0.7545  | 0.207      | 0.2975        | 0.698         |
| MLP_tabular           | 0.6931     | 0.7927     | 0.5124                 | 0.9588                  | 0.066            | 0.7546  | 0.2116     | 0.3757        | 0.276         |
| HistGradBoost         | 0.7019     | 0.7901     | 0.5484                 | 0.9176                  | 0.1792           | 0.7554  | 0.2163     | 0.3102        | 0.182         |
| LogReg_L2_balanced    | 0.6671     | 0.7773     | 0.5331                 | 0.9059                  | 0.1604           | 0.7458  | 0.2291     | 0.2974        | 0.059         |
| Dummy_StratifiedFloor | 0.4781     | 0.6059     | 0.5                    | 1                       | 0                | 0.7623  | 0.4964     | 0             | 0.006         |


## Horizon 6h, channel set: `values_masks`


| model                 | test_auroc | test_auprc | test_balanced_accuracy | test_recall_sensitivity | test_specificity | test_f1 | test_brier | val_threshold | train_seconds |
| --------------------- | ---------- | ---------- | ---------------------- | ----------------------- | ---------------- | ------- | ---------- | ------------- | ------------- |
| ExtraTrees_balanced   | 0.7254     | 0.8139     | 0.5655                 | 0.9706                  | 0.1604           | 0.7783  | 0.2043     | 0.3375        | 0.692         |
| RandomForest_balanced | 0.7127     | 0.8097     | 0.5113                 | 0.9471                  | 0.0755           | 0.7506  | 0.208      | 0.3275        | 1.182         |
| HistGradBoost         | 0.7166     | 0.8017     | 0.5891                 | 0.9235                  | 0.2547           | 0.7734  | 0.2131     | 0.2998        | 0.248         |
| MLP_tabular           | 0.7156     | 0.7981     | 0.5342                 | 0.9647                  | 0.1038           | 0.7646  | 0.2084     | 0.2803        | 0.434         |
| LogReg_L2_balanced    | 0.679      | 0.7752     | 0.5254                 | 0.9941                  | 0.0566           | 0.7699  | 0.2269     | 0.161         | 0.05          |
| Dummy_StratifiedFloor | 0.4781     | 0.6059     | 0.5                    | 1                       | 0                | 0.7623  | 0.4964     | 0             | 0             |


## Horizon 12h, channel set: `full`


| model                 | test_auroc | test_auprc | test_balanced_accuracy | test_recall_sensitivity | test_specificity | test_f1 | test_brier | val_threshold | train_seconds |
| --------------------- | ---------- | ---------- | ---------------------- | ----------------------- | ---------------- | ------- | ---------- | ------------- | ------------- |
| HistGradBoost         | 0.7671     | 0.8537     | 0.5832                 | 0.9588                  | 0.2075           | 0.7818  | 0.195      | 0.2093        | 0.547         |
| MLP_tabular           | 0.769      | 0.8467     | 0.641                  | 0.9235                  | 0.3585           | 0.7949  | 0.1895     | 0.3489        | 0.483         |
| RandomForest_balanced | 0.7575     | 0.8355     | 0.5744                 | 0.9412                  | 0.2075           | 0.7729  | 0.1929     | 0.3325        | 1.243         |
| ExtraTrees_balanced   | 0.7517     | 0.8284     | 0.6145                 | 0.8706                  | 0.3585           | 0.7668  | 0.1922     | 0.415         | 0.807         |
| LogReg_L2_balanced    | 0.7324     | 0.8109     | 0.6086                 | 0.9059                  | 0.3113           | 0.7758  | 0.2143     | 0.2083        | 0.091         |
| Dummy_StratifiedFloor | 0.4781     | 0.6059     | 0.5                    | 1                       | 0                | 0.7623  | 0.4964     | 0             | 0             |


## Horizon 12h, channel set: `values`


| model                 | test_auroc | test_auprc | test_balanced_accuracy | test_recall_sensitivity | test_specificity | test_f1 | test_brier | val_threshold | train_seconds |
| --------------------- | ---------- | ---------- | ---------------------- | ----------------------- | ---------------- | ------- | ---------- | ------------- | ------------- |
| ExtraTrees_balanced   | 0.7256     | 0.8179     | 0.5637                 | 0.9765                  | 0.1509           | 0.7793  | 0.2023     | 0.345         | 0.696         |
| RandomForest_balanced | 0.7208     | 0.8143     | 0.5454                 | 0.9588                  | 0.1321           | 0.7671  | 0.2044     | 0.3275        | 1.412         |
| HistGradBoost         | 0.7082     | 0.8123     | 0.5851                 | 0.8588                  | 0.3113           | 0.7506  | 0.2145     | 0.3119        | 0.207         |
| MLP_tabular           | 0.7124     | 0.7965     | 0.6458                 | 0.8765                  | 0.4151           | 0.7822  | 0.2227     | 0.3001        | 0.542         |
| LogReg_L2_balanced    | 0.6733     | 0.7794     | 0.5336                 | 0.9824                  | 0.0849           | 0.7696  | 0.2275     | 0.2031        | 0.031         |
| Dummy_StratifiedFloor | 0.4781     | 0.6059     | 0.5                    | 1                       | 0                | 0.7623  | 0.4964     | 0             | 0             |


## Horizon 12h, channel set: `values_masks`


| model                 | test_auroc | test_auprc | test_balanced_accuracy | test_recall_sensitivity | test_specificity | test_f1 | test_brier | val_threshold | train_seconds |
| --------------------- | ---------- | ---------- | ---------------------- | ----------------------- | ---------------- | ------- | ---------- | ------------- | ------------- |
| ExtraTrees_balanced   | 0.7478     | 0.8351     | 0.6128                 | 0.8765                  | 0.3491           | 0.768   | 0.197      | 0.4825        | 0.745         |
| RandomForest_balanced | 0.7279     | 0.8249     | 0.5626                 | 0.9176                  | 0.2075           | 0.761   | 0.202      | 0.39          | 1.256         |
| HistGradBoost         | 0.7223     | 0.8228     | 0.5525                 | 0.9824                  | 0.1226           | 0.7767  | 0.2093     | 0.2138        | 0.312         |
| MLP_tabular           | 0.7196     | 0.8141     | 0.5354                 | 0.9765                  | 0.0943           | 0.7685  | 0.2119     | 0.1247        | 0.444         |
| LogReg_L2_balanced    | 0.6978     | 0.7943     | 0.5437                 | 0.9647                  | 0.1226           | 0.7681  | 0.2231     | 0.1461        | 0.067         |
| Dummy_StratifiedFloor | 0.4781     | 0.6059     | 0.5                    | 1                       | 0                | 0.7623  | 0.4964     | 0             | 0             |


## Horizon 18h, channel set: `full`


| model                 | test_auroc | test_auprc | test_balanced_accuracy | test_recall_sensitivity | test_specificity | test_f1 | test_brier | val_threshold | train_seconds |
| --------------------- | ---------- | ---------- | ---------------------- | ----------------------- | ---------------- | ------- | ---------- | ------------- | ------------- |
| RandomForest_balanced | 0.7576     | 0.8483     | 0.5873                 | 0.9294                  | 0.2453           | 0.7745  | 0.194      | 0.3675        | 1.212         |
| HistGradBoost         | 0.7447     | 0.8464     | 0.6388                 | 0.7588                  | 0.5189           | 0.7371  | 0.2008     | 0.4552        | 0.563         |
| ExtraTrees_balanced   | 0.7536     | 0.8405     | 0.6251                 | 0.9294                  | 0.3208           | 0.79    | 0.1937     | 0.3925        | 0.874         |
| MLP_tabular           | 0.7291     | 0.8227     | 0.595                  | 0.9353                  | 0.2547           | 0.7794  | 0.2075     | 0.2647        | 0.489         |
| LogReg_L2_balanced    | 0.7181     | 0.7873     | 0.5856                 | 0.9353                  | 0.2358           | 0.7756  | 0.2195     | 0.1766        | 0.092         |
| Dummy_StratifiedFloor | 0.4781     | 0.6059     | 0.5                    | 1                       | 0                | 0.7623  | 0.4964     | 0             | 0             |


## Horizon 18h, channel set: `values`


| model                 | test_auroc | test_auprc | test_balanced_accuracy | test_recall_sensitivity | test_specificity | test_f1 | test_brier | val_threshold | train_seconds |
| --------------------- | ---------- | ---------- | ---------------------- | ----------------------- | ---------------- | ------- | ---------- | ------------- | ------------- |
| HistGradBoost         | 0.7143     | 0.8249     | 0.5684                 | 0.9765                  | 0.1604           | 0.7812  | 0.2119     | 0.213         | 0.245         |
| RandomForest_balanced | 0.7197     | 0.824      | 0.5703                 | 0.8765                  | 0.2642           | 0.7506  | 0.2051     | 0.43          | 1.472         |
| ExtraTrees_balanced   | 0.7135     | 0.8155     | 0.5974                 | 0.9118                  | 0.283            | 0.7731  | 0.2061     | 0.4325        | 0.725         |
| MLP_tabular           | 0.6993     | 0.8144     | 0.5549                 | 0.9118                  | 0.1981           | 0.7561  | 0.2125     | 0.3276        | 0.3           |
| LogReg_L2_balanced    | 0.6971     | 0.7908     | 0.5419                 | 0.9706                  | 0.1132           | 0.7692  | 0.221      | 0.2109        | 0.037         |
| Dummy_StratifiedFloor | 0.4781     | 0.6059     | 0.5                    | 1                       | 0                | 0.7623  | 0.4964     | 0             | 0             |


## Horizon 18h, channel set: `values_masks`


| model                 | test_auroc | test_auprc | test_balanced_accuracy | test_recall_sensitivity | test_specificity | test_f1 | test_brier | val_threshold | train_seconds |
| --------------------- | ---------- | ---------- | ---------------------- | ----------------------- | ---------------- | ------- | ---------- | ------------- | ------------- |
| ExtraTrees_balanced   | 0.7383     | 0.8354     | 0.6009                 | 0.9471                  | 0.2547           | 0.7854  | 0.2004     | 0.3575        | 0.775         |
| RandomForest_balanced | 0.7242     | 0.8254     | 0.5927                 | 0.9118                  | 0.2736           | 0.7711  | 0.2028     | 0.425         | 1.32          |
| HistGradBoost         | 0.7137     | 0.8108     | 0.5902                 | 0.9824                  | 0.1981           | 0.7915  | 0.2108     | 0.2187        | 0.346         |
| LogReg_L2_balanced    | 0.7046     | 0.781      | 0.5496                 | 0.9294                  | 0.1698           | 0.7596  | 0.2226     | 0.2072        | 0.059         |
| MLP_tabular           | 0.6769     | 0.7776     | 0.568                  | 0.8059                  | 0.3302           | 0.7249  | 0.2252     | 0.4748        | 0.368         |
| Dummy_StratifiedFloor | 0.4781     | 0.6059     | 0.5                    | 1                       | 0                | 0.7623  | 0.4964     | 0             | 0             |


## Horizon 24h, channel set: `full`


| model                 | test_auroc | test_auprc | test_balanced_accuracy | test_recall_sensitivity | test_specificity | test_f1 | test_brier | val_threshold | train_seconds |
| --------------------- | ---------- | ---------- | ---------------------- | ----------------------- | ---------------- | ------- | ---------- | ------------- | ------------- |
| RandomForest_balanced | 0.7609     | 0.8577     | 0.5873                 | 0.9294                  | 0.2453           | 0.7745  | 0.1924     | 0.36          | 1.352         |
| HistGradBoost         | 0.7661     | 0.852      | 0.6529                 | 0.8529                  | 0.4528           | 0.7775  | 0.1918     | 0.3949        | 0.843         |
| ExtraTrees_balanced   | 0.7633     | 0.8518     | 0.6062                 | 0.9765                  | 0.2358           | 0.7962  | 0.1921     | 0.2925        | 0.968         |
| MLP_tabular           | 0.7542     | 0.8475     | 0.5767                 | 0.9647                  | 0.1887           | 0.781   | 0.1988     | 0.2159        | 0.579         |
| LogReg_L2_balanced    | 0.7042     | 0.76       | 0.542                  | 0.9235                  | 0.1604           | 0.7548  | 0.2283     | 0.084         | 0.114         |
| Dummy_StratifiedFloor | 0.4781     | 0.6059     | 0.5                    | 1                       | 0                | 0.7623  | 0.4964     | 0             | 0             |


## Horizon 24h, channel set: `values`


| model                 | test_auroc | test_auprc | test_balanced_accuracy | test_recall_sensitivity | test_specificity | test_f1 | test_brier | val_threshold | train_seconds |
| --------------------- | ---------- | ---------- | ---------------------- | ----------------------- | ---------------- | ------- | ---------- | ------------- | ------------- |
| ExtraTrees_balanced   | 0.7371     | 0.8453     | 0.5703                 | 0.9235                  | 0.217            | 0.7659  | 0.2016     | 0.38          | 0.756         |
| RandomForest_balanced | 0.719      | 0.8247     | 0.5714                 | 0.9353                  | 0.2075           | 0.77    | 0.2044     | 0.36          | 1.505         |
| HistGradBoost         | 0.7171     | 0.8183     | 0.6063                 | 0.8353                  | 0.3774           | 0.7513  | 0.2091     | 0.3908        | 0.253         |
| MLP_tabular           | 0.7047     | 0.8092     | 0.556                  | 0.9706                  | 0.1415           | 0.7746  | 0.2105     | 0.2837        | 0.316         |
| LogReg_L2_balanced    | 0.6987     | 0.7934     | 0.5289                 | 0.9824                  | 0.0755           | 0.7678  | 0.2205     | 0.1519        | 0.035         |
| Dummy_StratifiedFloor | 0.4781     | 0.6059     | 0.5                    | 1                       | 0                | 0.7623  | 0.4964     | 0             | 0             |


## Horizon 24h, channel set: `values_masks`


| model                 | test_auroc | test_auprc | test_balanced_accuracy | test_recall_sensitivity | test_specificity | test_f1 | test_brier | val_threshold | train_seconds |
| --------------------- | ---------- | ---------- | ---------------------- | ----------------------- | ---------------- | ------- | ---------- | ------------- | ------------- |
| ExtraTrees_balanced   | 0.7625     | 0.8597     | 0.5979                 | 0.9882                  | 0.2075           | 0.7962  | 0.1934     | 0.325         | 0.811         |
| RandomForest_balanced | 0.7262     | 0.8284     | 0.5962                 | 0.9                     | 0.2925           | 0.7688  | 0.2026     | 0.43          | 1.392         |
| HistGradBoost         | 0.7295     | 0.8279     | 0.6016                 | 0.8824                  | 0.3208           | 0.7653  | 0.2042     | 0.3432        | 0.44          |
| MLP_tabular           | 0.7173     | 0.8055     | 0.5861                 | 0.9647                  | 0.2075           | 0.7847  | 0.2332     | 0.1041        | 0.667         |
| LogReg_L2_balanced    | 0.6814     | 0.7668     | 0.5248                 | 0.9647                  | 0.0849           | 0.761   | 0.2322     | 0.0613        | 0.059         |
| Dummy_StratifiedFloor | 0.4781     | 0.6059     | 0.5                    | 1                       | 0                | 0.7623  | 0.4964     | 0             | 0             |


