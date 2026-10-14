"""文件内容与用途：4.1：比较纯 Sigmoid MLP、LR 与 Ridge，保存和绘制五折训练损失。"""

from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from main import (
    ROOT,
    DATASETS,
    ABBR,
    SEED,
    LinearRegression,
    Ridge,
    prepare_dataset,
    save_joblib,
    prediction_row,
    train_mlp,
    mlp_config,
    save_mlp,
    predict_mlp,
    load_mlp,
    write_csv,
    write_json,
    log,
)


def run_pure_input(name, run_dir, base_only=False):
    """执行纯 Sigmoid MLP、LR、Ridge 的同输入比较，供 4.1 与 4.2 共用。"""
    folder, a = prepare_dataset(name, run_dir)
    dest = folder / ("section42_base_only" if base_only else "section41")
    if (dest / "complete.json").exists():
        return pd.read_csv(dest / "metrics.csv")
    dest.mkdir(parents=True, exist_ok=True)
    x = a["oof"] if base_only else np.column_stack([a["x"], a["oof"]])
    prep = joblib.load(folder / "preprocess.joblib")
    feature_set = "base_only" if base_only else "base_plus_raw"
    rows = []
    linear_losses = {}
    for label, est in {"LR": LinearRegression(), "Ridge": Ridge(alpha=1.0, random_state=SEED)}.items():
        start = time.perf_counter()
        est.fit(x, a["y"])
        p = est.predict(x)
        save_joblib(dest / label / "model.joblib", est)
        row = prediction_row(
            name,
            label,
            "train",
            feature_set,
            a["train_ids"],
            a["y_orig"],
            p,
            prep,
            dest / label / "predictions.csv",
            time.perf_counter() - start,
        )
        rows.append(row)
        linear_losses[label] = row["MSELoss"]
    model, info, curve = train_mlp(
        x,
        a["y"],
        mlp_config(name, pure=True),
        pure=True,
        progress_name=f"{name} {feature_set} Sigmoid",
        checkpoint_dir=dest,
    )
    save_mlp(dest / "MLP", model, info)
    p = predict_mlp(model, x, info["actual_batch_size"])
    loaded, _ = load_mlp(dest / "MLP")
    np.testing.assert_allclose(p, predict_mlp(loaded, x, info["actual_batch_size"]), rtol=1e-6, atol=1e-6)
    row = prediction_row(
        name,
        "MLP",
        "train",
        feature_set,
        a["train_ids"],
        a["y_orig"],
        p,
        prep,
        dest / "MLP" / "predictions.csv",
        info["fit_seconds"],
    )
    row.update(best_epoch=info["best_epoch"], best_eval_loss=info["best_eval_loss"])
    rows.insert(0, row)
    curve["LR_mse"] = linear_losses["LR"]
    curve["Ridge_mse"] = linear_losses["Ridge"]
    curve["best_mlp_mse"] = info["best_eval_loss"]
    write_csv(dest / "loss_curve.csv", curve)
    result = pd.DataFrame(rows)
    write_csv(dest / "metrics.csv", result)
    write_json(dest / "complete.json", dict(feature_set=feature_set, rows=3))
    log(f'{name} {feature_set} COMPLETE best_mlp={info["best_eval_loss"]:.9g} LR={linear_losses["LR"]:.9g}')
    return result


def run_section41(name, run_dir):
    """完成 4.1 联合输入比较；图像由本文件的五折绘图函数生成。"""
    return run_pure_input(name, run_dir, False)


def plot_curve_panel(
    ax, epochs: np.ndarray, mean: np.ndarray, std: np.ndarray, lr_mean: float, ridge_mean: float, mode: str
) -> None:
    """绘制完整历史、MLP尾段或线性基线放大面板，避免不同量级互相压缩。"""
    ax.plot(epochs, mean, color="#2166ac", lw=1.2, label="MLP mean MSE")
    ax.fill_between(
        epochs, np.maximum(0, mean - std), mean + std, color="#67a9cf", alpha=0.24, label="MLP ±1 SD"
    )
    ax.axhline(lr_mean, color="#b2182b", ls="--", lw=1.6, label=f"LR mean: {lr_mean:.6g}")
    ax.axhline(ridge_mean, color="#d89013", ls=(0, (2, 3)), lw=1.6, label=f"Ridge mean: {ridge_mean:.6g}")
    if mode == "tail":
        start = min(max(0, int(len(epochs) * 0.8)), max(0, len(epochs) - 2))
        values = np.concatenate([mean[start:] - std[start:], mean[start:] + std[start:]])
        low, high = float(np.nanmin(values)), float(np.nanmax(values))
        pad = max((high - low) * 0.12, 1e-8)
        ax.set_xlim(epochs[start], epochs[-1])
        ax.set_ylim(max(0, low - pad), high + pad)
        ax.set_title("Final 20% epochs: convergence detail")
    elif mode == "main":
        # 保留完整横轴；跳过最初1%的尖峰，用主曲线振幅占画面约2/3设纵轴。
        start = max(1, int(len(epochs)*0.01))
        low, high = float(np.nanmin(mean[start:])), float(np.nanmax(mean[start:]))
        span = max(high-low, abs(high)*0.01, 1e-8)
        bottom = max(0.0, low - span*0.12)
        ax.set_xlim(1, epochs[-1])
        ax.set_ylim(bottom, bottom + span*1.5)
    elif mode == "linear":
        low, high = sorted([lr_mean, ridge_mean])
        gap = max(high - low, max(abs(low), abs(high), 1.0) * 1e-4)
        ax.set_ylim(max(0, low - gap * 0.8), high + gap * 0.8)
        ax.set_title("Linear baselines: expanded scale")
    else:
        finite = np.concatenate([mean + std, [lr_mean, ridge_mean]])
        low, high = float(np.nanmin(finite)), float(np.nanmax(finite))
        pad = max((high - low) * 0.06, 1e-8)
        ax.set_xlim(1, epochs[-1])
        ax.set_ylim(max(0, low - pad), high + pad)
        ax.set_title("Full training history")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (standardized target)")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, loc="best")


def plot_linear_inset(ax, lr_mean: float, ridge_mean: float) -> None:
    """在主图内嵌放大坐标，清楚显示五折 LR 与 Ridge 平均 MSE 的微小差异。"""
    inset = ax.inset_axes([0.58, 0.50, 0.38, 0.24])
    values = np.array([lr_mean, ridge_mean], dtype=float)
    span = max(abs(lr_mean - ridge_mean), max(abs(lr_mean), abs(ridge_mean), 1.0) * 1e-8)
    inset.scatter([0, 1], values, s=24, color=["#b2182b", "#d89013"], zorder=3)
    inset.plot([0, 1], values, color="#555555", lw=0.7, alpha=0.65)
    inset.set_xlim(-0.35, 1.35)
    inset.set_ylim(values.min() - 2 * span, values.max() + 2 * span)
    inset.set_xticks([0, 1], ["LR", "Ridge"])
    inset.ticklabel_format(axis="y", style="plain", useOffset=False)
    inset.tick_params(labelsize=6)
    inset.grid(axis="y", alpha=0.22)
    inset.set_title("LR vs Ridge", fontsize=7)


def plot_fivefold_curves(source: Path, output: Path) -> pd.DataFrame:
    """对五折4.1曲线求均值和标准差，输出双面板单图及带线性放大框的总图。"""
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    (output / "summary").mkdir(parents=True, exist_ok=True)
    diagnostics = []
    combined = []
    for dataset in DATASETS:
        table = pd.read_csv(source / f"{dataset}.csv", encoding="utf-8-sig")
        if set(table.outer_fold) != set(range(5)):
            raise ValueError(f"{dataset} 损失曲线缺少折号")
        curves = [table.loc[table.outer_fold == fold].reset_index(drop=True) for fold in range(5)]
        epochs = curves[0]["epoch"].to_numpy()
        if any(not np.array_equal(epochs, curve["epoch"].to_numpy()) for curve in curves[1:]):
            raise RuntimeError(f"{dataset} 的五折epoch不一致")
        values = np.vstack([curve["eval_mse"].to_numpy() for curve in curves])
        mean, std = values.mean(axis=0), values.std(axis=0, ddof=1)
        lr_mean = float(np.mean([curve["LR_mse"].iloc[0] for curve in curves]))
        ridge_mean = float(np.mean([curve["Ridge_mse"].iloc[0] for curve in curves]))
        tail = max(2, int(len(epochs) * 0.1))
        slope = float(np.polyfit(epochs[-tail:], mean[-tail:], 1)[0])
        diagnostics.append(
            {
                "dataset": dataset,
                "epochs": int(epochs[-1]),
                "final_mean_mse": float(mean[-1]),
                "final_sd_mse": float(std[-1]),
                "last_10pct_slope_per_epoch": slope,
                "LR_mean_mse": lr_mean,
                "Ridge_mean_mse": ridge_mean,
            }
        )
        combined.append((dataset, epochs, mean, std, lr_mean, ridge_mean))
        with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42}):
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
            for ax, mode in zip(axes, ("full", "tail")):
                plot_curve_panel(ax, epochs, mean, std, lr_mean, ridge_mean, mode)
            plot_linear_inset(axes[1], lr_mean, ridge_mean)
            fig.suptitle(f"{ABBR[dataset]} — five 80:20 splits; OOF + raw inputs; AdamW", fontsize=13)
            for extension in ("png", "pdf"):
                fig.savefig(figure_dir / f"{dataset}_loss_fivefold.{extension}", dpi=240)
            plt.close(fig)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(3, 2, figsize=(13, 13), layout="constrained")
        for ax, (dataset, epochs, mean, std, lr_mean, ridge_mean) in zip(axes.flat, combined):
            plot_curve_panel(ax, epochs, mean, std, lr_mean, ridge_mean, "main")
            ax.set_title(f"{ABBR[dataset]} — full epoch range")
            plot_linear_inset(ax, lr_mean, ridge_mean)
        fig.suptitle("Five-fold mean training loss with ±1 SD", fontsize=14)
        for extension in ("png", "pdf"):
            fig.savefig(figure_dir / f"all_six_convergence_fivefold.{extension}", dpi=240)
        plt.close(fig)
    result = pd.DataFrame(diagnostics)
    write_csv(output / "summary" / "section41_curve_diagnostics.csv", result)
    return result


if __name__ == "__main__":
    from main import main

    main("41")
