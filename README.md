# Anime-Hit-Predictor

**Tsuki no Kokyu, Ichi no Kata: Yami Duki, Yoi no Miya**
*(Moon Breathing, First Form: Dark Moon, Evening Palace)*

Kokushibo's my favorite Demon Slayer character and yeah I know this has nothing to do with ML, I just think the move name is cool. Anyway, here's the actual project.

## What this is

A model that predicts whether an anime will become a "breakout hit" **before it even airs**, using studio history, director track record, genres, tags, source material, and release timing. Trained on anime from 2000-2024.

Not a toy notebook. Time-based splits, hyperparameter tuning, SHAP explainability, and I stress-tested it against real 2025 anime the model never saw during training.

## Why this isn't just a random classifier

- Split by **year**, not randomly (train ≤2021, val=2022, test ≥2023). Random splits leak future info and make models look better than they are.
- Used **SHAP** to explain *why* the model predicts what it predicts, not just raw feature importance.
- Tuned with `RandomizedSearchCV` + early stopping instead of default params.
- Ran the final model on **real 2025 anime** to sanity check it against reality, not just held-out rows from the same dataset.

## Tech stack

Python, LightGBM, XGBoost, SHAP, scikit-learn, pandas, matplotlib.

## Results (unseen 2023-2024 test data)

| Metric | Score |
|---|---|
| PR-AUC | **0.614** |
| Precision (Hit) | 0.614 |
| Recall (Hit) | 0.486 |
| F1 (Hit) | 0.543 |
| Accuracy | 0.85 |

Random baseline PR-AUC here is 0.185 (only ~18% of test anime are actual hits), so this model beats random guessing by **3.3x**.

*Most populars.*
<img width="1992" height="590" alt="most_popular" src="https://github.com/user-attachments/assets/6b592387-e002-4cfb-af16-5092ce17e3df" />


## What actually predicts a hit

<img width="989" height="590" alt="featured_importance_lgbm" src="https://github.com/user-attachments/assets/a77e460e-2dd7-44cd-bd52-9962736008bf" />

*Top 15 features from the tuned LightGBM model. Release timing, director reputation, and studio reputation dominate.*

<img width="780" height="940" alt="tuned_lgbm_anime" src="https://github.com/user-attachments/assets/a0717a89-0ff5-411c-893f-1b6b386653c4" />

*SHAP plot showing direction of impact, not just magnitude. Male Protagonist, studio score, and Tragedy/Ensemble Cast tags push hardest toward a "hit" prediction.*

<img width="645" height="453" alt="studio_by_breakout_hits" src="https://github.com/user-attachments/assets/8b9d3a17-360a-4ba4-9bc3-3acf773e191f" />

*Top 10 studios by breakout hit rate. Big names like Kyoto Animation and ufotable don't always win, smaller studios like Studio Bind and Orange top the list.*

## Real-world validation (the fun part)

Ran the trained model on actual 2025 anime it never trained on:

| Anime | Prediction | Probability | Actual |
|---|---|---|---|
| Dororo | Hit | 97.58% | ✅ Hit |
| Orb: On the Movements of the Earth | Hit | 90.77% | ✅ Hit |
| Lazarus | Hit | 83.31% | ⚠️ Mixed (91% RT score, but mixed fan reception) |
| BanG Dream! Ave Mujica | Not Hit | 26.06% | ✅ Flop |
| One-Punch Man S3 | Not Hit | 44.08% | ✅ Flop |

**4/5 correct.** Lazarus is the interesting miss, it got great critical scores but never became a true breakout, showing the model's limits when a show is "good" but not "phenomenon" level.

## Setup

```bash
pip install pandas numpy lightgbm xgboost scikit-learn shap matplotlib
jupyter notebook Anime_ML.ipynb
```

## Notes

Solo project, built from scratch, dataset scraped/compiled independently. Threshold locked at 0.78 (optimized for precision on validation data).
