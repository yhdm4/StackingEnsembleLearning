"""文件内容与用途：4.4：重新加载4.3模型，使用十折模型均值评估固定测试集。"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from main import (
    METHODS,
    BASE_NAMES,
    load_base_groups,
    predict_base_feature_matrix,
    write_csv,
    load_mlp,
    predict_nn_stacking,
    predict_mlp,
    prediction_row,
    log,
    write_json,
    read_json,
    synchronized_time,
)


def run_section44(name, run_dir):
    """仅加载 4.3 保存的模型预测固定测试集，第一层对每类十个模型取均值。"""
    folder = Path(run_dir) / "datasets" / name
    if not all((folder / "section43" / m / "complete.json").exists() for m in METHODS):
        raise ValueError(f"{name} 请先完成 4.3")
    metrics_path = folder / "section44" / "metrics.csv"
    existing_rows = {}
    if metrics_path.exists():
        existing = pd.read_csv(metrics_path, encoding="utf-8-sig")
        existing_rows = {str(row.method): row.to_dict() for _, row in existing.iterrows()}
    prep = joblib.load(folder / "preprocess.joblib")
    test = pd.read_csv(folder / "test.csv", encoding="utf-8-sig")
    ids = np.load(folder / "train_arrays.npz")["test_ids"]
    x = prep["scaler_x"].transform(test[prep["feature_names"]].to_numpy(dtype=float))
    groups = load_base_groups(folder)
    base_start = synchronized_time()
    base = predict_base_feature_matrix(groups, x)
    base_seconds = synchronized_time() - base_start
    # 独立核对均值实现，防止测试阶段误用单折预测。
    for j, n in enumerate(BASE_NAMES):
        expected = sum(m.predict(x) for m in groups[n]) / 10
        # XGBoost 输出 float32；两种求和顺序会有约 1e-7 的舍入差异。
        np.testing.assert_allclose(base[:, j], expected, rtol=1e-6, atol=1e-6)
    del groups
    joint = np.column_stack([x, base])
    y = test[prep["target"]].to_numpy(dtype=float)
    rows = []
    write_csv(
        folder / "section44" / "base_mean_features.csv",
        pd.DataFrame(base, columns=BASE_NAMES).assign(row_id=ids),
    )
    for method in METHODS:
        prediction_path = folder / "section44" / method / "predictions.csv"
        if method in existing_rows and prediction_path.exists():
            rows.append(existing_rows[method])
            continue
        src = folder / "section43" / method
        features = x if method.startswith("E2E") else joint
        start = synchronized_time()
        if method == "NN-Stacking":
            model, info = load_mlp(src)
            p = predict_nn_stacking(model, x, base, info["actual_batch_size"])
        elif method in ["Proposed", "E2E-MLP"]:
            model, info = load_mlp(src)
            p = predict_mlp(model, features, info["actual_batch_size"])
        elif method == "Convex-Stacking":
            p = base @ np.asarray(read_json(src / "model.json")["weights"])
            model = None
        else:
            model = joblib.load(src / "model.joblib")
            p = model.predict(features)
        prediction_seconds = synchronized_time() - start
        feature_set = "raw" if method.startswith("E2E") else "raw_plus_fold_mean"
        if method == "NN-Stacking":
            feature_set = "raw_to_weights_plus_fold_mean"
        if method == "Convex-Stacking":
            feature_set = "fold_mean_convex_weights"
        row = prediction_row(name, method, "test", feature_set, ids, y, p, prep, prediction_path)
        row.update(prediction_seconds=prediction_seconds, shared_base_prediction_seconds=0.0 if method.startswith("E2E") else base_seconds)
        rows.append(row)
        log(f'{name} 4.4 {method} R2={row["R2"]:.6f} RMSE={row["RMSE"]:.6f}')
        del model
    write_csv(folder / "section44" / "metrics.csv", pd.DataFrame(rows))
    write_json(
        folder / "section44" / "complete.json",
        dict(methods=METHODS, source="reloaded_saved_models", mean_verified=True),
    )


if __name__ == "__main__":
    from main import main

    main("44")
