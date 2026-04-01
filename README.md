# Snowflake CR XGB Distributor: `train_fn_per_worker`

## What Changed

`XGBEstimator` now accepts `train_fn_per_worker` — a custom training function that runs on each distributed worker. This follows the same contract as [Ray's XGBoostTrainer `train_loop_per_worker`](https://docs.ray.io/en/latest/train/api/doc/ray.train.xgboost.XGBoostTrainer.html), giving you full access to OSS `xgb.train()` while the distributor still handles multi-node orchestration and data sharding.

Previously, the only way to configure training was through the `params` dict and top-level `XGBEstimator` arguments. Anything that required `DMatrix`-level control (sample weights, base margin, feature weights, group structure) or `xgb.train()`-level control (custom objectives, custom eval metrics) was not available in distributed mode. Note: callbacks were already supported as a top-level `XGBEstimator` argument.

## Usage

```python
from snowflake.ml.modeling.distributors.xgboost import XGBEstimator, XGBScalingConfig

def my_train_fn(config: dict):
    import xgboost as xgb
    import ray.train

    shard = ray.train.get_dataset_shard("train")
    data = shard.materialize().to_pandas()

    dtrain = xgb.DMatrix(data[feature_cols], label=data[label_col], weight=...)
    xgb.train(config, dtrain, num_boost_round=200, callbacks=[...], obj=..., feval=...)

estimator = XGBEstimator(
    n_estimators=200,
    objective="multi:softprob",
    scaling_config=XGBScalingConfig(num_workers=-1, num_cpu_per_worker=-1, use_gpu=False),
    train_fn_per_worker=my_train_fn,
)
booster = estimator.fit(dataset=connector, input_cols=[...], label_col="LABEL")
```

## What This Unlocks

All OSS XGBoost functionality that was previously blocked in distributed mode:

| Capability | How |
|---|---|
| **Sample weights** | `DMatrix(weight=...)` |
| **Base margin** | `dtrain.set_base_margin(...)` |
| **Feature weights** | `dtrain.set_info(feature_weights=...)` |
| **Group structure (LTR)** | `dtrain.set_group(...)` |
| **Custom objective** | `xgb.train(..., obj=my_obj_fn)` |
| **Custom eval metric** | `xgb.train(..., feval=my_eval_fn)` |
| **Warm-start / incremental** | `xgb.train(..., xgb_model=existing_booster)` |
| **Interaction constraints** | Via params inside `xgb.train()` |
| **Monotone constraints** | Via params inside `xgb.train()` |

## Demo Notebook

`xgb_distributor_weighted_demo.ipynb` — demonstrates sample weights for long-tailed multi-class classification with a 3-way comparison:

1. **Baseline**: Standard `XGBEstimator` (no weights)
2. **CR Distributor + `train_fn_per_worker`**: Effective-number-of-samples weighting via `DMatrix(weight=...)`
3. **Local OSS XGBoost**: Same weighting scheme run locally as a control

## Requirements

- `mlruntimes_service >= 2.5.7` on the compute pool runtime
- **Note**: Workspace Notebooks currently run `mlruntimes_service 2.2.0`, which does **not** include `train_fn_per_worker`. You must use `@remote` to dispatch training to a compute pool running a compatible runtime version.

## Infrastructure

| Resource | Value |
|---|---|
| Database | `DB_CR_XGB_TEST` |
| Schema | `SCHEMA_DISTRIBUTOR` |
| Compute Pool | `XGB_DIST_CPU_POOL` |
| Role | `ML_PERF_ROLE` |
