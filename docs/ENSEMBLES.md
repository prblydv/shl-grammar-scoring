# 1. Ensemble mechanics

An **ensemble** combines predictions from several models. Here, a second-stage learner uses those predictions to estimate corrections. A **residual** is the difference between a true score and a prediction. The equations and saved parameters below specify the archived result.

1. **Start from aligned predictions.** The latest archive contains 15 metadata exports representing 14 unique OOF runs. Each input must match all 769 labels, fold/source-group assignments, and 216 test IDs. Reproduce base pooled RMSE, reject nonfinite/out-of-range values, and deduplicate identical OOF runs before stacking. The earlier retained joint component uses **12** columns, while the later hurdle uses **14**; these inventories are not interchangeable.

2. **Linear and quadratic corrections.** Elastic Net and Ridge constrain the weights assigned to correlated models. Later models learn a residual around the first prediction column, rather than relearning the entire score. Quadratic designs add a score-level term and interactions between predicted level and inter-model disagreement. This lets the ensemble adjust the correction across the score range with regularization.

3. **The retained joint-quality component.** The retained joint model uses 12 base predictions. It combines a quadratic feature kernel with a multiscale radial basis function (RBF) kernel. The quadratic terms model score-dependent corrections; the RBF terms allow smooth nonlinear corrections. RBF scales are 0.3, 1, and 3 times the selected bandwidth. Each model is scaled using errors from the training partition. The selected full fit uses alpha 100, gamma 0.3, new-column scale 1, and kernel weight 10. The anchor is the first base-model prediction. The learner estimates a regularized correction to that anchor and clips the result to [0,5]. Kernel centering uses only training statistics. The original implementation is [here](../scripts/optimize_all_metadata_ensembles_20261015.py); selected fold/full-fit settings are [archived](../reports/joint_fits.json).

4. **The experimental hurdle component.** A logistic classifier estimates `q(x) = P(y > 0 | x)` from the 14 base prediction columns. Separately, a polynomial/Ridge model fits the residual `y − x[0]` on positive-labelled training rows. Its positive prediction is the anchor plus this residual estimate. The final hurdle output is `q(x) × clip(positive_prediction, 0, 5)`. For the final fit on all labelled rows, the selected settings are logistic C=10, polynomial degree=1, and Ridge alpha=10. Some validation folds select different settings. Every scaler and learner fits within its training partition. See [implementation](../scripts/research_hurdle_rmse_20261015.py) and [fit records](../reports/hurdle_fits.json).

5. **Exact archived public result.**

   ```text
   prediction = 0.60 * hurdle_14_model_prediction
              + 0.40 * prior_12_model_joint_quality_prediction
   ```

   The winning file in the archive is `03_hurdle60_joint40.csv`, SHA256 `a6f2ff3bcc7b9833b68882c0564daeb00c1bf812b00a59d0277a61ef0bb64d6c`. The score/rank record is [here](../reports/winner_manifest.json). This is an experimental leaderboard result with material generalization concerns, not a claim that the hurdle is the preferred deployable architecture.

6. **Public-informed selection.**

   | Hurdle / joint weight | Recorded public RMSE |
   |---|---:|
   | 25% / 75% | 0.3158 |
   | 35% / 65% | 0.3153 |
   | 45% / 55% | 0.3148 |
   | 60% / 40% | 0.3141 |

   These follow-up weights were chosen after public feedback. The 60% version's pooled meta-development RMSE is 0.47758949, versus 0.47718415 for 35%. Larger weights remain untested in this record. The trajectory does not justify extrapolating to a pure hurdle or claiming private gains.

7. **What this does not establish.** A learned zero classifier is not a filename lookup, but it may still capture unusual dataset composition rather than transferable grammar. The base predictions were saved in advance, so the full model stack was not retrained inside each outer validation split. The source archive also contains the earlier 11-model non-hurdle public result of 0.3166; the retained 12-model joint component in the formula above is not that same file. See the [generalization discussion](GENERALIZATION.md).
