"""文件内容与用途：4.3：训练十二种回归方法，保存模型并评估训练集拟合。"""

from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from main import (
    METHODS,
    prepare_dataset,
    make_meta_models,
    make_base_models,
    read_json,
    log,
    train_nn_stacking,
    mlp_config,
    predict_nn_stacking,
    save_mlp,
    write_csv,
    load_mlp,
    train_mlp,
    predict_mlp,
    save_joblib,
    write_json,
    prediction_row,
    fit_convex,
    select_epochs,
)


def run_section43(name, run_dir):
    """训练十二种方法并评估训练拟合；凸组合使用严格OOF预测求全局权重。"""
    folder, a = prepare_dataset(name, run_dir)
    prep = joblib.load(folder / "preprocess.joblib")
    selected_epochs = select_epochs(name, folder)
    joint = np.column_stack([a["x"], a["oof"]])
    estimators = make_meta_models()
    base = make_base_models()
    estimators.update({"E2E-XGB": base["XGBoost"], "E2E-RF": base["RandomForest"], "E2E-SVM": base["SVM"]})
    rows = []
    for method in METHODS:
        dest = folder / "section43" / method
        complete = dest / "complete.json"
        if complete.exists():
            rows.append(read_json(complete)["row"])
            continue
        dest.mkdir(parents=True, exist_ok=True)
        x = a["x"] if method.startswith("E2E") else joint
        feature_set = "raw" if method.startswith("E2E") else "raw_plus_oof"
        if method == "NN-Stacking":
            feature_set = "raw_to_weights_plus_oof"
        log(f"{name} 4.3 {method} START")
        start = time.perf_counter()
        config = mlp_config(name)
        if method in selected_epochs:
            config.update(epochs=selected_epochs[method], selection="fixed_endpoint")
        if method == "NN-Stacking":
            model, info, curve = train_nn_stacking(
                a["x"],
                a["oof"],
                a["y"],
                config,
                progress_name=f"{name} {method}",
                checkpoint_dir=dest,
            )
            p = predict_nn_stacking(model, a["x"], a["oof"], info["actual_batch_size"])
            save_mlp(dest, model, info)
            write_csv(dest / "loss_curve.csv", curve)
            loaded, loaded_info = load_mlp(dest)
            reloaded = predict_nn_stacking(loaded, a["x"], a["oof"], loaded_info["actual_batch_size"])
            np.testing.assert_allclose(p, reloaded, rtol=1e-6, atol=1e-6)
            del loaded
        elif method == "Convex-Stacking":
            original = a["oof"] * prep["scaler_y"].scale_[0] + prep["scaler_y"].mean_[0]
            convex = fit_convex(original, a["y_orig"])
            convex["weights"] = convex["weights"].tolist()
            write_json(dest / "model.json", convex)
            p = a["oof"] @ np.asarray(convex["weights"])
            np.testing.assert_allclose(p, a["oof"] @ np.asarray(read_json(dest / "model.json")["weights"]))
            feature_set = "oof_convex_weights"
            model = None
            write_csv(dest / "weights.csv", pd.DataFrame({"base_model": ["SVM", "RandomForest", "XGBoost"], "weight": convex["weights"]}))
        elif method in ["Proposed", "E2E-MLP"]:
            model, info, curve = train_mlp(
                x, a["y"], config, progress_name=f"{name} {method}", checkpoint_dir=dest
            )
            p = predict_mlp(model, x, info["actual_batch_size"])
            save_mlp(dest, model, info)
            write_csv(dest / "loss_curve.csv", curve)
            loaded, loaded_info = load_mlp(dest)
            np.testing.assert_allclose(
                p, predict_mlp(loaded, x, loaded_info["actual_batch_size"]), rtol=1e-6, atol=1e-6
            )
            del loaded
        else:
            model = estimators[method].fit(x, a["y"])
            p = model.predict(x)
            save_joblib(dest / "model.joblib", model)
            np.testing.assert_allclose(
                p, joblib.load(dest / "model.joblib").predict(x), rtol=1e-10, atol=1e-10
            )
            params = {
                k: ("NaN" if isinstance(v, float) and np.isnan(v) else v)
                for k, v in model.get_params().items()
            }
            write_json(dest / "model.json", dict(params=params, input_dim=x.shape[1]))
        row = prediction_row(
            name,
            method,
            "train",
            feature_set,
            a["train_ids"],
            a["y_orig"],
            p,
            prep,
            dest / "predictions.csv",
            time.perf_counter() - start,
        )
        write_json(complete, dict(row=row, model_reloaded_train_predictions_verified=True))
        rows.append(row)
        log(f'{name} 4.3 {method} DONE R2={row["R2"]:.6f} RMSE={row["RMSE"]:.6f}')
        del model
    write_csv(folder / "section43" / "metrics.csv", pd.DataFrame(rows))


if __name__ == "__main__":
    from main import main

    main("43")
