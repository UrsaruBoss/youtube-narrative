# Results Snapshot

- Rows: **177,430**
- Text column: **text_en** (non-empty: 173,845)
- Stance neutral share: **92.64%**
- Tactics neutral share: **90.30%**

## Stance distribution

| label        |   count |   pct |
|:-------------|--------:|------:|
| neutral      |  164376 | 92.64 |
| pro_russia   |    9983 |  5.63 |
| anti_russia  |    2700 |  1.52 |
| anti_ukraine |     352 |  0.2  |
| peace_call   |      16 |  0.01 |
| anti_west    |       2 |  0    |
| pro_west     |       1 |  0    |

## Tactics distribution

| label          |   count |   pct |
|:---------------|--------:|------:|
| neutral        |  160218 | 90.3  |
| propaganda     |   12251 |  6.9  |
| conspiracy     |    4850 |  2.73 |
| dehumanization |     111 |  0.06 |

## Stance score stats (by label)

| stance_label   |   count |   mean |   median |    min |    max |
|:---------------|--------:|-------:|---------:|-------:|-------:|
| neutral        |  164376 | 0.2535 |   0.2208 | 0      | 0.9927 |
| pro_russia     |    9983 | 0.5713 |   0.543  | 0.3601 | 0.9858 |
| anti_russia    |    2700 | 0.5352 |   0.4934 | 0.3601 | 0.9961 |
| anti_ukraine   |     352 | 0.6398 |   0.6135 | 0.3601 | 0.9873 |
| peace_call     |      16 | 0.7578 |   0.7314 | 0.5508 | 0.9712 |
| anti_west      |       2 | 0.7942 |   0.7942 | 0.6499 | 0.9385 |
| pro_west       |       1 | 0.4878 |   0.4878 | 0.4878 | 0.4878 |

## Tactic score stats (by label)

| tactic_label   |   count |   mean |   median |    min |    max |
|:---------------|--------:|-------:|---------:|-------:|-------:|
| neutral        |  160218 | 0.3302 |   0.3303 | 0      | 0.999  |
| propaganda     |   12251 | 0.5253 |   0.4976 | 0.4399 | 0.9873 |
| conspiracy     |    4850 | 0.6506 |   0.6221 | 0.4399 | 0.9956 |
| dehumanization |     111 | 0.4556 |   0.4565 | 0.4412 | 0.5527 |

## Stance × Tactic (counts)

| stance       |   conspiracy |   dehumanization |   neutral |   propaganda |
|:-------------|-------------:|-----------------:|----------:|-------------:|
| anti_russia  |          256 |                3 |      2039 |          402 |
| anti_ukraine |           27 |                2 |       288 |           35 |
| anti_west    |            0 |                0 |         1 |            1 |
| neutral      |         3698 |               98 |    150016 |        10564 |
| peace_call   |            2 |                0 |        10 |            4 |
| pro_russia   |          867 |                8 |      7863 |         1245 |
| pro_west     |            0 |                0 |         1 |            0 |

## Stance × Tactic (percent of total)

| stance       |   conspiracy |   dehumanization |   neutral |   propaganda |
|:-------------|-------------:|-----------------:|----------:|-------------:|
| anti_russia  |        0.144 |            0.002 |     1.149 |        0.227 |
| anti_ukraine |        0.015 |            0.001 |     0.162 |        0.02  |
| anti_west    |        0     |            0     |     0.001 |        0.001 |
| neutral      |        2.084 |            0.055 |    84.549 |        5.954 |
| peace_call   |        0.001 |            0     |     0.006 |        0.002 |
| pro_russia   |        0.489 |            0.005 |     4.432 |        0.702 |
| pro_west     |        0     |            0     |     0.001 |        0     |

## Top examples (saved as CSV)

- stance: **pro_russia** → `reports\top_stance_pro_russia.csv`
- stance: **anti_russia** → `reports\top_stance_anti_russia.csv`
- stance: **anti_ukraine** → `reports\top_stance_anti_ukraine.csv`
- stance: **peace_call** → `reports\top_stance_peace_call.csv`
- stance: **anti_west** → `reports\top_stance_anti_west.csv`
- stance: **pro_west** → `reports\top_stance_pro_west.csv`
- tactic: **propaganda** → `reports\top_tactic_propaganda.csv`
- tactic: **conspiracy** → `reports\top_tactic_conspiracy.csv`
- tactic: **dehumanization** → `reports\top_tactic_dehumanization.csv`
