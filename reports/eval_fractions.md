# Отчёт run_eval (2026-11-06 22:28)

Наборов: 30; режимы: baseline, +rag, +agents; доли данных: [0.1, 0.2, 0.5, 1.0]; holdout: [0.2]

## classification

| mode     |   holdout |   fraction |   m_roc_auc |   m_f1 |   m_accuracy |   m_recall |   m_brier |   time_s |
|:---------|----------:|-----------:|------------:|-------:|-------------:|-----------:|----------:|---------:|
| +agents  |       0.2 |        0.1 |      0.8304 | 0.6155 |       0.7822 |     0.6199 |    0.1564 |  13.1279 |
| +agents  |       0.2 |        0.2 |      0.8581 | 0.6956 |       0.7931 |     0.7446 |    0.1483 |  10.9155 |
| +agents  |       0.2 |        0.5 |      0.8768 | 0.7133 |       0.8096 |     0.7609 |    0.1346 |   9.0517 |
| +agents  |       0.2 |        1   |      0.8882 | 0.7309 |       0.8173 |     0.7756 |    0.1308 |   7.9821 |
| +rag     |       0.2 |        0.1 |      0.8451 | 0.6693 |       0.7769 |     0.6977 |    0.156  |   0.1788 |
| +rag     |       0.2 |        0.2 |      0.865  | 0.7079 |       0.8011 |     0.753  |    0.1438 |   0.1688 |
| +rag     |       0.2 |        0.5 |      0.8848 | 0.7256 |       0.8178 |     0.7685 |    0.1293 |   0.194  |
| +rag     |       0.2 |        1   |      0.8933 | 0.7401 |       0.8233 |     0.7851 |    0.1244 |   0.2746 |
| baseline |       0.2 |        0.1 |      0.8107 | 0.6378 |       0.7566 |     0.6673 |    0.1727 |   0.0567 |
| baseline |       0.2 |        0.2 |      0.8324 | 0.6654 |       0.7728 |     0.7072 |    0.1649 |   0.0544 |
| baseline |       0.2 |        0.5 |      0.861  | 0.7022 |       0.7933 |     0.7586 |    0.1496 |   0.0628 |
| baseline |       0.2 |        1   |      0.8815 | 0.723  |       0.8067 |     0.7828 |    0.1356 |   0.0747 |

## regression

| mode     |   holdout |   fraction |   m_r2 |   m_mae |   m_rmse |   time_s |
|:---------|----------:|-----------:|-------:|--------:|---------:|---------:|
| +agents  |       0.2 |        0.1 | 0.5801 |  5.3169 |   6.6127 |   3.5969 |
| +agents  |       0.2 |        0.2 | 0.6125 |  5.1901 |   6.4176 |   3.6184 |
| +agents  |       0.2 |        0.5 | 0.7271 |  4.5811 |   5.7891 |   3.052  |
| +agents  |       0.2 |        1   | 0.749  |  4.3834 |   5.5679 |   3.1188 |
| +rag     |       0.2 |        0.1 | 0.5819 |  5.3243 |   6.6092 |   0.0807 |
| +rag     |       0.2 |        0.2 | 0.6121 |  5.1822 |   6.4181 |   0.0779 |
| +rag     |       0.2 |        0.5 | 0.7252 |  4.5931 |   5.7944 |   0.0873 |
| +rag     |       0.2 |        1   | 0.7465 |  4.3875 |   5.5749 |   0.0868 |
| baseline |       0.2 |        0.1 | 0.4851 |  5.8267 |   7.2175 |   0.0357 |
| baseline |       0.2 |        0.2 | 0.5232 |  6.0693 |   7.5704 |   0.0348 |
| baseline |       0.2 |        0.5 | 0.6598 |  4.9901 |   6.3954 |   0.0389 |
| baseline |       0.2 |        1   | 0.7279 |  4.5284 |   5.7812 |   0.0409 |

## Статистические тесты (парные по наборам)

| задача         |   holdout |   доля | сравнение           | метрика   |   Δ среднее | 95% CI            |   Wilcoxon p |   t-test p |   win rate |
|:---------------|----------:|-------:|:--------------------|:----------|------------:|:------------------|-------------:|-----------:|-----------:|
| classification |       0.2 |    0.1 | +rag vs baseline    | roc_auc   |      0.0343 | [-0.0021; 0.0651] |       0.0016 |     0.0632 |     0.8095 |
| classification |       0.2 |    0.1 | +agents vs baseline | roc_auc   |      0.0197 | [-0.0146; 0.0519] |       0.0421 |     0.2807 |     0.7619 |
| classification |       0.2 |    0.1 | +agents vs +rag     | roc_auc   |     -0.0147 | [-0.0464; 0.0137] |       0.4527 |     0.3684 |     0.1429 |
| classification |       0.2 |    0.2 | +rag vs baseline    | roc_auc   |      0.0326 | [0.0103; 0.0525]  |       0.0012 |     0.0089 |     0.9048 |
| classification |       0.2 |    0.2 | +agents vs baseline | roc_auc   |      0.0257 | [-0.0012; 0.0499] |       0.0127 |     0.0638 |     0.8571 |
| classification |       0.2 |    0.2 | +agents vs +rag     | roc_auc   |     -0.0069 | [-0.0216; 0.0028] |       0.7    |     0.3381 |     0.0952 |
| classification |       0.2 |    0.5 | +rag vs baseline    | roc_auc   |      0.0237 | [0.0136; 0.0343]  |       0.0004 |     0.0003 |     0.8571 |
| classification |       0.2 |    0.5 | +agents vs baseline | roc_auc   |      0.0158 | [-0.0007; 0.0303] |       0.0158 |     0.0643 |     0.8095 |
| classification |       0.2 |    0.5 | +agents vs +rag     | roc_auc   |     -0.008  | [-0.0221; 0.0004] |       0.679  |     0.2189 |     0.0476 |
| classification |       0.2 |    1   | +rag vs baseline    | roc_auc   |      0.0118 | [-0.0001; 0.0215] |       0.001  |     0.053  |     0.8571 |
| classification |       0.2 |    1   | +agents vs baseline | roc_auc   |      0.0067 | [-0.0079; 0.0182] |       0.0142 |     0.3411 |     0.8095 |
| classification |       0.2 |    1   | +agents vs +rag     | roc_auc   |     -0.0051 | [-0.0157; 0.0013] |       0.679  |     0.3215 |     0.0476 |
| regression     |       0.2 |    0.1 | +rag vs baseline    | r2        |      0.0968 | [0.0208; 0.1855]  |       0.0156 |     0.0628 |     0.7778 |
| regression     |       0.2 |    0.1 | +agents vs baseline | r2        |      0.095  | [0.0208; 0.1798]  |       0.0156 |     0.0605 |     0.7778 |
| regression     |       0.2 |    0.1 | +agents vs +rag     | r2        |     -0.0019 | [-0.0069; 0.0014] |       1      |     0.4629 |     0.1111 |
| regression     |       0.2 |    0.2 | +rag vs baseline    | r2        |      0.0889 | [0.0098; 0.1812]  |       0.0156 |     0.0864 |     0.7778 |
| regression     |       0.2 |    0.2 | +agents vs baseline | r2        |      0.0893 | [0.0115; 0.1817]  |       0.0156 |     0.0821 |     0.7778 |
| regression     |       0.2 |    0.2 | +agents vs +rag     | r2        |      0.0004 | [-0.0035; 0.0048] |       1      |     0.8356 |     0.1111 |
| regression     |       0.2 |    0.5 | +rag vs baseline    | r2        |      0.0654 | [0.0008; 0.1446]  |       0.0781 |     0.1287 |     0.6667 |
| regression     |       0.2 |    0.5 | +agents vs baseline | r2        |      0.0673 | [0.0008; 0.1502]  |       0.0781 |     0.1332 |     0.6667 |
| regression     |       0.2 |    0.5 | +agents vs +rag     | r2        |      0.0019 | [0.0000; 0.0056]  |       0.5    |     0.3466 |     0.2222 |
| regression     |       0.2 |    1   | +rag vs baseline    | r2        |      0.0186 | [-0.0108; 0.0517] |       0.2188 |     0.3028 |     0.5556 |
| regression     |       0.2 |    1   | +agents vs baseline | r2        |      0.021  | [-0.0108; 0.0586] |       0.2188 |     0.2986 |     0.5556 |
| regression     |       0.2 |    1   | +agents vs +rag     | r2        |      0.0024 | [0.0000; 0.0073]  |       0.5    |     0.3466 |     0.2222 |

## Тест Диболда–Мариано по наборам (поэлементные потери)

| problem_type   |   holdout |   fraction | a       | b        |   datasets |   significant |   a_better_significant |
|:---------------|----------:|-----------:|:--------|:---------|-----------:|--------------:|-----------------------:|
| classification |       0.2 |        0.1 | +rag    | baseline |         21 |             8 |                      6 |
| classification |       0.2 |        0.1 | +agents | baseline |         21 |             9 |                      7 |
| classification |       0.2 |        0.1 | +agents | +rag     |         21 |             2 |                      1 |
| classification |       0.2 |        0.2 | +rag    | baseline |         21 |             6 |                      6 |
| classification |       0.2 |        0.2 | +agents | baseline |         21 |             6 |                      5 |
| classification |       0.2 |        0.2 | +agents | +rag     |         21 |             2 |                      0 |
| classification |       0.2 |        0.5 | +rag    | baseline |         21 |             6 |                      6 |
| classification |       0.2 |        0.5 | +agents | baseline |         21 |             7 |                      6 |
| classification |       0.2 |        0.5 | +agents | +rag     |         21 |             1 |                      0 |
| classification |       0.2 |        1   | +rag    | baseline |         21 |             4 |                      4 |
| classification |       0.2 |        1   | +agents | baseline |         21 |             5 |                      4 |
| classification |       0.2 |        1   | +agents | +rag     |         21 |             1 |                      0 |
| regression     |       0.2 |        0.1 | +rag    | baseline |          9 |             2 |                      2 |
| regression     |       0.2 |        0.1 | +agents | baseline |          9 |             2 |                      2 |
| regression     |       0.2 |        0.1 | +agents | +rag     |          9 |             0 |                      0 |
| regression     |       0.2 |        0.2 | +rag    | baseline |          9 |             4 |                      4 |
| regression     |       0.2 |        0.2 | +agents | baseline |          9 |             4 |                      4 |
| regression     |       0.2 |        0.2 | +agents | +rag     |          9 |             0 |                      0 |
| regression     |       0.2 |        0.5 | +rag    | baseline |          9 |             2 |                      2 |
| regression     |       0.2 |        0.5 | +agents | baseline |          9 |             2 |                      2 |
| regression     |       0.2 |        0.5 | +agents | +rag     |          9 |             0 |                      0 |
| regression     |       0.2 |        1   | +rag    | baseline |          9 |             2 |                      2 |
| regression     |       0.2 |        1   | +agents | baseline |          9 |             2 |                      2 |
| regression     |       0.2 |        1   | +agents | +rag     |          9 |             0 |                      0 |

## Агентные метрики (+agents)

```json
{
  "runs": 120,
  "avg_iterations": 1.3583333333333334,
  "hitl_rate": 0.14166666666666666,
  "avg_time_s": 8.161175,
  "avg_tokens": 4282.591666666666,
  "avg_llm_calls": 2.716666666666667,
  "selection_modes": {
    "native": 96,
    "carried": 23,
    "forced": 1
  },
  "models": {
    "fit_logistic_regression": 53,
    "fit_ridge_regression": 28,
    "fit_tabular_xgboost": 20,
    "fit_random_forest": 19
  }
}
```
