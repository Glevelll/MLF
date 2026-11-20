# Отчёт run_eval (2026-11-20 16:49)

Наборов: 20; режимы: baseline, +rag, +agents; доли данных: [0.1, 0.2, 0.5, 1.0]; holdout: [0.2]

Успешных запусков — 240. Запусков с ошибками — 0.

## classification

| mode     |   holdout |   fraction |   m_roc_auc |   m_f1 |   m_accuracy |   m_recall |   m_brier |   time_s |
|:---------|----------:|-----------:|------------:|-------:|-------------:|-----------:|----------:|---------:|
| +agents  |       0.2 |        0.1 |      0.8611 | 0.7097 |       0.8031 |     0.7317 |    0.1752 |  10.2221 |
| +agents  |       0.2 |        0.2 |      0.8829 | 0.7386 |       0.7999 |     0.7865 |    0.1519 |  10.908  |
| +agents  |       0.2 |        0.5 |      0.8845 | 0.7429 |       0.8034 |     0.7914 |    0.1392 |   4.4513 |
| +agents  |       0.2 |        1   |      0.8898 | 0.7513 |       0.8079 |     0.8092 |    0.1341 |   4.4156 |
| +rag     |       0.2 |        0.1 |      0.8688 | 0.7276 |       0.7906 |     0.765  |    0.1749 |   0.0728 |
| +rag     |       0.2 |        0.2 |      0.8854 | 0.7381 |       0.8018 |     0.786  |    0.1488 |   0.0882 |
| +rag     |       0.2 |        0.5 |      0.891  | 0.7551 |       0.8115 |     0.8089 |    0.1331 |   0.1043 |
| +rag     |       0.2 |        1   |      0.897  | 0.7612 |       0.8133 |     0.8251 |    0.1281 |   0.1519 |
| baseline |       0.2 |        0.1 |      0.8164 | 0.6732 |       0.7658 |     0.6644 |    0.1705 |   0.061  |
| baseline |       0.2 |        0.2 |      0.8598 | 0.7291 |       0.7884 |     0.7729 |    0.1544 |   0.0681 |
| baseline |       0.2 |        0.5 |      0.8736 | 0.7369 |       0.8002 |     0.7689 |    0.1441 |   0.0709 |
| baseline |       0.2 |        1   |      0.8874 | 0.7432 |       0.8024 |     0.7914 |    0.1342 |   0.0782 |

## regression

| mode     |   holdout |   fraction |   m_r2 |   m_mae |   m_rmse |   time_s |
|:---------|----------:|-----------:|-------:|--------:|---------:|---------:|
| +agents  |       0.2 |        0.1 | 0.574  |  6.1586 |   7.5734 |   9.0415 |
| +agents  |       0.2 |        0.2 | 0.6234 |  5.993  |   7.2698 |   9.7978 |
| +agents  |       0.2 |        0.5 | 0.6876 |  5.3764 |   6.6664 |   9.9928 |
| +agents  |       0.2 |        1   | 0.7111 |  5.1781 |   6.4115 |  10.015  |
| +rag     |       0.2 |        0.1 | 0.562  |  6.1797 |   7.5972 |   0.0525 |
| +rag     |       0.2 |        0.2 | 0.6085 |  6.0031 |   7.3022 |   0.0548 |
| +rag     |       0.2 |        0.5 | 0.6686 |  5.3971 |   6.7087 |   0.0693 |
| +rag     |       0.2 |        1   | 0.6841 |  5.1921 |   6.472  |   0.0942 |
| baseline |       0.2 |        0.1 | 0.2751 |  7.055  |   8.6968 |   0.0307 |
| baseline |       0.2 |        0.2 | 0.2657 |  7.6296 |   9.4427 |   0.0358 |
| baseline |       0.2 |        0.5 | 0.6102 |  5.9138 |   7.5006 |   0.0427 |
| baseline |       0.2 |        1   | 0.6583 |  5.3698 |   6.7397 |   0.0418 |

## Статистические тесты (парные по наборам)

| задача         |   holdout |   доля | сравнение           | метрика   |   Δ среднее | 95% CI            |   Wilcoxon p |   t-test p |   win rate |
|:---------------|----------:|-------:|:--------------------|:----------|------------:|:------------------|-------------:|-----------:|-----------:|
| classification |       0.2 |    0.1 | +rag vs baseline    | roc_auc   |      0.0524 | [0.0223; 0.0878]  |       0.0031 |     0.0095 |     0.7857 |
| classification |       0.2 |    0.1 | +agents vs baseline | roc_auc   |      0.0447 | [0.0027; 0.0847]  |       0.0245 |     0.0636 |     0.7857 |
| classification |       0.2 |    0.1 | +agents vs +rag     | roc_auc   |     -0.0077 | [-0.0302; 0.0085] |       0.4319 |     0.4884 |     0.0714 |
| classification |       0.2 |    0.2 | +rag vs baseline    | roc_auc   |      0.0256 | [0.0111; 0.0402]  |       0.0067 |     0.0066 |     0.8571 |
| classification |       0.2 |    0.2 | +agents vs baseline | roc_auc   |      0.0231 | [0.0021; 0.0421]  |       0.0419 |     0.0524 |     0.8571 |
| classification |       0.2 |    0.2 | +agents vs +rag     | roc_auc   |     -0.0024 | [-0.0157; 0.0107] |       0.4713 |     0.7309 |     0.0714 |
| classification |       0.2 |    0.5 | +rag vs baseline    | roc_auc   |      0.0174 | [0.0095; 0.0252]  |       0.0006 |     0.0012 |     0.8571 |
| classification |       0.2 |    0.5 | +agents vs baseline | roc_auc   |      0.0109 | [-0.0008; 0.0221] |       0.0785 |     0.0829 |     0.7143 |
| classification |       0.2 |    0.5 | +agents vs +rag     | roc_auc   |     -0.0065 | [-0.0155; 0.0000] |       0.1948 |     0.1422 |     0      |
| classification |       0.2 |    1   | +rag vs baseline    | roc_auc   |      0.0096 | [0.0019; 0.0177]  |       0.0676 |     0.0384 |     0.6429 |
| classification |       0.2 |    1   | +agents vs baseline | roc_auc   |      0.0024 | [-0.0100; 0.0138] |       0.7609 |     0.7017 |     0.5    |
| classification |       0.2 |    1   | +agents vs +rag     | roc_auc   |     -0.0072 | [-0.0162; 0.0000] |       0.1948 |     0.1155 |     0      |
| regression     |       0.2 |    0.1 | +rag vs baseline    | r2        |      0.287  | [0.0378; 0.7395]  |       0.0312 |     0.256  |     1      |
| regression     |       0.2 |    0.1 | +agents vs baseline | r2        |      0.299  | [0.0378; 0.7755]  |       0.0312 |     0.2606 |     1      |
| regression     |       0.2 |    0.1 | +agents vs +rag     | r2        |      0.012  | [0.0000; 0.0360]  |       1      |     0.3632 |     0.1667 |
| regression     |       0.2 |    0.2 | +rag vs baseline    | r2        |      0.3427 | [0.1150; 0.6285]  |       0.0312 |     0.068  |     1      |
| regression     |       0.2 |    0.2 | +agents vs baseline | r2        |      0.3577 | [0.1150; 0.6734]  |       0.0312 |     0.0771 |     1      |
| regression     |       0.2 |    0.2 | +agents vs +rag     | r2        |      0.015  | [0.0000; 0.0449]  |       1      |     0.3632 |     0.1667 |
| regression     |       0.2 |    0.5 | +rag vs baseline    | r2        |      0.0584 | [0.0154; 0.1061]  |       0.1562 |     0.0834 |     0.8333 |
| regression     |       0.2 |    0.5 | +agents vs baseline | r2        |      0.0774 | [0.0173; 0.1442]  |       0.1562 |     0.0922 |     0.8333 |
| regression     |       0.2 |    0.5 | +agents vs +rag     | r2        |      0.019  | [0.0000; 0.0569]  |       1      |     0.3632 |     0.1667 |
| regression     |       0.2 |    1   | +rag vs baseline    | r2        |      0.0258 | [0.0034; 0.0502]  |       0.1562 |     0.1114 |     0.8333 |
| regression     |       0.2 |    1   | +agents vs baseline | r2        |      0.0528 | [0.0034; 0.1296]  |       0.1562 |     0.2137 |     0.8333 |
| regression     |       0.2 |    1   | +agents vs +rag     | r2        |      0.027  | [0.0000; 0.0809]  |       0.5    |     0.3632 |     0.3333 |

## Тест Диболда–Мариано по наборам (поэлементные потери)

| problem_type   |   holdout |   fraction | a       | b        |   datasets |   significant |   a_better_significant |
|:---------------|----------:|-----------:|:--------|:---------|-----------:|--------------:|-----------------------:|
| classification |       0.2 |        0.1 | +rag    | baseline |         14 |             9 |                      4 |
| classification |       0.2 |        0.1 | +agents | baseline |         14 |             8 |                      4 |
| classification |       0.2 |        0.1 | +agents | +rag     |         14 |             1 |                      0 |
| classification |       0.2 |        0.2 | +rag    | baseline |         14 |             7 |                      5 |
| classification |       0.2 |        0.2 | +agents | baseline |         14 |             8 |                      5 |
| classification |       0.2 |        0.2 | +agents | +rag     |         14 |             2 |                      0 |
| classification |       0.2 |        0.5 | +rag    | baseline |         14 |             3 |                      3 |
| classification |       0.2 |        0.5 | +agents | baseline |         14 |             4 |                      3 |
| classification |       0.2 |        0.5 | +agents | +rag     |         14 |             3 |                      0 |
| classification |       0.2 |        1   | +rag    | baseline |         14 |             1 |                      1 |
| classification |       0.2 |        1   | +agents | baseline |         14 |             2 |                      1 |
| classification |       0.2 |        1   | +agents | +rag     |         14 |             3 |                      0 |
| regression     |       0.2 |        0.1 | +rag    | baseline |          6 |             2 |                      2 |
| regression     |       0.2 |        0.1 | +agents | baseline |          6 |             2 |                      2 |
| regression     |       0.2 |        0.1 | +agents | +rag     |          6 |             0 |                      0 |
| regression     |       0.2 |        0.2 | +rag    | baseline |          6 |             6 |                      6 |
| regression     |       0.2 |        0.2 | +agents | baseline |          6 |             6 |                      6 |
| regression     |       0.2 |        0.2 | +agents | +rag     |          6 |             0 |                      0 |
| regression     |       0.2 |        0.5 | +rag    | baseline |          6 |             3 |                      3 |
| regression     |       0.2 |        0.5 | +agents | baseline |          6 |             3 |                      3 |
| regression     |       0.2 |        0.5 | +agents | +rag     |          6 |             0 |                      0 |
| regression     |       0.2 |        1   | +rag    | baseline |          6 |             3 |                      3 |
| regression     |       0.2 |        1   | +agents | baseline |          6 |             3 |                      3 |
| regression     |       0.2 |        1   | +agents | +rag     |          6 |             0 |                      0 |

## Агентные метрики (+agents)

```json
{
  "runs": 80,
  "avg_iterations": 1.3375,
  "hitl_rate": 0.0875,
  "avg_time_s": 8.0758875,
  "avg_tokens": 4141.175,
  "avg_llm_calls": 2.675,
  "selection_modes": {
    "native": 62,
    "carried": 18
  },
  "models": {
    "fit_logistic_regression": 52,
    "fit_ridge_regression": 19,
    "fit_random_forest": 8,
    "fit_tabular_xgboost": 1
  },
  "llm_models_by_run": {
    "mock:mock-policy-v1": 80
  }
}
```
