# -*- coding: utf-8 -*-
"""文件内容与用途：四节实验共用的数据处理、模型、统计计算和命令行入口。

用途：为 4.1–4.4 提供同一个实现；所有数据和结果均相对于本文件所在目录。
预处理沿用历史口径：在每次80%训练集清洗/标准化后做基学习器十折 OOF。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import time
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor
import xgboost

ROOT = Path(__file__).resolve().parent
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_FILE = None
DATASETS = [
    "AlignedData_aligned",
    "Concrete_Data",
    "parkinsons_updrs_regression",
    "wine_quality_combined",
    "airfoil_self_noise",
    "superconductivity",
]
ABBR = dict(zip(DATASETS, ["AD", "CO", "PU", "WQ", "AS", "SC"]))
BASE_NAMES = ["SVM", "RandomForest", "XGBoost"]
METHODS = [
    "Proposed",
    "Meta-LR",
    "Meta-Ridge",
    "Meta-XGB",
    "Meta-RF",
    "Meta-SVM",
    "E2E-XGB",
    "E2E-RF",
    "E2E-SVM",
    "E2E-MLP",
]
METHODS += ["NN-Stacking", "Convex-Stacking"]
# 单折候选轮数已经完成比较；4.3/4.4正式五折统一使用下面冻结的epochs。
# 保留空映射可阻止外层折再次各自选择不同轮数，保证三种神经方法和五个外层折口径一致。
EPOCH_CANDIDATES = {}
CACHE_ROOT = None
# 4.3/4.4采用单折探索后冻结的统一参数；五折及三种神经方法共用相同训练预算。
FIXED_MLP_CONFIGS = {
    "AlignedData_aligned": dict(
        hidden_dims=[1536, 1024, 512, 256],
        dropout=0.02,
        weight_decay=0.000001,
        lr=0.0003,
        batch_size=512,
        epochs=2500,
    ),
    "Concrete_Data": dict(
        hidden_dims=[384, 192, 96], dropout=0.05, weight_decay=0.00001, lr=0.0008, batch_size=32, epochs=750
    ),
    "parkinsons_updrs_regression": dict(
        hidden_dims=[1280, 1024, 512, 256],
        dropout=0.40,
        weight_decay=0.0002,
        lr=0.0004,
        batch_size=256,
        epochs=1200,
    ),
    "wine_quality_combined": dict(
        hidden_dims=[768, 512, 256, 128], dropout=0.35, weight_decay=0.0003, lr=0.0003, batch_size=64, epochs=150
    ),
    "airfoil_self_noise": dict(
        hidden_dims=[512, 256, 128],
        dropout=0.03,
        weight_decay=0.000001,
        lr=0.0003,
        batch_size=32,
        epochs=1800,
    ),
    "superconductivity": dict(
        hidden_dims=[2048, 1536, 768],
        dropout=0.10,
        weight_decay=0.0001,
        lr=0.0003,
        batch_size=256,
        epochs=280,
        use_lr_scheduler=False,
    ),
}
# 4.1/4.2 共用隐藏层、学习率和批大小；训练轮数与最终损失曲线一致。
FIT_CONFIGS = {
    "AlignedData_aligned": dict(lr=0.0004, batch_size=512, epochs=2000),
    "Concrete_Data": dict(lr=0.0008, batch_size=64, epochs=2000),
    "parkinsons_updrs_regression": dict(lr=0.0004, batch_size=256, epochs=2000),
    "wine_quality_combined": dict(lr=0.0003, batch_size=32, epochs=2000),
    "airfoil_self_noise": dict(lr=0.0003, batch_size=32, epochs=2000),
    "superconductivity": dict(lr=0.0003, batch_size=256, epochs=4000),
}
# 纯Sigmoid比较有独立结构，避免调4.3的网络时意外改变4.1/4.2。
FIT_HIDDEN_DIMS = {
    "AlignedData_aligned": [1024, 768, 384, 192],
    "Concrete_Data": [256, 128, 64],
    "parkinsons_updrs_regression": [1280, 1024, 512, 256],
    "wine_quality_combined": [512, 496, 192],
    "airfoil_self_noise": [512, 256, 128],
    "superconductivity": [2048, 1024, 512],
}


import importlib
import itertools
import math
import sys
from typing import Iterable
from scipy.stats import friedmanchisquare, rankdata

# 直接运行主程序时让四个实验共享同一份模块状态和模型类。
if __name__ == "__main__":
    sys.modules["main"] = sys.modules[__name__]


def feature_columns(frame):
    """选择实际输入与目标；AD使用匿名列，SC使用官方完整81个输入。"""
    if "y" in frame:
        expected = [f"x{i}" for i in range(1, len(frame.columns))] + ["y"]
        if list(frame.columns) != expected:
            raise ValueError("匿名数据列必须依次为 x1, x2, ..., y")
        return list(frame.columns[:-1]), "y"
    target = "critical_temp" if "critical_temp" in frame else frame.columns[-1]
    excluded = {target}
    return [column for column in frame if column not in excluded], target


def grouped_data(name):
    """按完全相同输入分组，目标取组内均值，保留首次出现顺序及原行映射。"""
    raw = pd.read_csv(ROOT / "data" / f"{name}.csv", encoding="utf-8-sig")
    columns, target = feature_columns(raw)
    if not np.isfinite(raw[columns + [target]].to_numpy(dtype=float)).all():
        raise ValueError(f"{name} 含缺失或非有限值")
    groups = raw.groupby(columns, sort=False, dropna=False)
    ids = groups.ngroup().to_numpy()
    frame = raw.loc[~raw.duplicated(columns), columns + [target]].reset_index(drop=True)
    summary = raw.assign(group_id=ids).groupby("group_id", sort=True)[target].agg(["mean", "size", "min", "max"])
    frame[target] = summary["mean"].to_numpy()
    audit = dict(raw_rows=len(raw), grouped_rows=len(frame), features=len(columns),
                 duplicate_input_rows=len(raw)-len(frame), conflicting_groups=int((summary["min"] != summary["max"]).sum()),
                 rule="exact_input_group_mean_target_first_appearance", data_sha256=sha256(ROOT / "data" / f"{name}.csv"))
    return frame, raw, ids, audit


def run_path(output, fold):
    """优先复用指定的历史模型目录，否则使用结果目录内的缓存。"""
    return Path(str(CACHE_ROOT) + str(fold)) if CACHE_ROOT else Path(output) / ".cache" / f"fold{fold}"


def synchronized_time():
    """等显卡任务结束后读计时器，避免异步计算少计耗时。"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def fit_convex(predictions, target):
    """枚举三权重单纯形的所有面，求非负且和为1的全局最小二乘解。"""
    best = None
    # 只有三个基模型，枚举七个非空面可同时覆盖边界最优解，无迭代收敛误差。
    for size in range(1, 4):
        for active in itertools.combinations(range(3), size):
            p = predictions[:, active]
            gram, rhs = p.T @ p, p.T @ target
            system = np.block([[gram, np.ones((size, 1))], [np.ones((1, size)), np.zeros((1, 1))]])
            w = np.linalg.lstsq(system, np.r_[rhs, 1.0], rcond=None)[0][:size]
            if np.min(w) < -1e-8 or not np.isclose(w.sum(), 1, atol=1e-7):
                continue
            w = np.maximum(w, 0); w /= w.sum()
            weights = np.zeros(3); weights[list(active)] = w
            mse = float(np.mean((predictions @ weights - target)**2))
            if best is None or mse < best[0]:
                best = mse, weights
    if best is None:
        raise RuntimeError("凸组合求解失败")
    return dict(weights=best[1], mse=best[0], solver="enumerate_all_simplex_faces", intercept=0.0)


def split_indices(n, outer_fold):
    """固定随机种子生成五组互斥测试集，每组使用约80%训练、20%测试。"""
    folds = list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(np.arange(n)))
    return folds[int(outer_fold)]


def init_run(run_dir, outer_fold, datasets):
    """记录数据、全部代码、参数与环境指纹；条件变化时拒绝复用训练缓存。"""
    run_dir = Path(run_dir)
    protocol = dict(
        seed=SEED,
        version=4,
        evaluation_status="post_selection_descriptive_fivefold_fold0_test_used_in_tuning",
        deduplication="exact_input_group_mean_target_before_outer_split",
        epoch_candidates=EPOCH_CANDIDATES,
        outer_split="KFold5_outer_80_20",
        outer_fold=outer_fold,
        inner_folds=10,
        preprocessing_scope="whole_outer_training_set_before_inner_KFold",
        checkpoint_selection="minimum_training_eval_MSE",
        optimizer="AdamW",
        mlp=FIXED_MLP_CONFIGS,
        pure_mlp=FIT_CONFIGS,
        pure_hidden_dims=FIT_HIDDEN_DIMS,
        code={p.name: sha256(p) for p in sorted(ROOT.glob("*.py"))},
        data={name: sha256(ROOT / "data" / f"{name}.csv") for name in DATASETS},
        versions=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            pandas=pd.__version__,
            torch=torch.__version__,
            sklearn=sklearn.__version__,
            xgboost=xgboost.__version__,
        ),
    )
    path = run_dir / "protocol.json"
    if path.exists() and read_json(path) != protocol:
        raise ValueError("数据、代码、参数、数据集选择或环境已变化，请使用新的 --output 目录")
    write_json(path, protocol)
    return run_dir


def log(message):
    """输出带时间的即时进度，便于查看长时间实验日志。"""
    line = time.strftime("%Y-%m-%d %H:%M:%S") + " " + str(message)
    print(line, flush=True)
    if LOG_FILE is not None:
        with LOG_FILE.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def sha256(path):
    """分块计算文件哈希，用于识别数据、代码和模型是否发生变化。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    """先写临时文件再原子替换，避免中断留下半个配置或完成标记。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    """按 UTF-8 读取实验元数据。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path, frame):
    """以带 BOM 的 UTF-8 原子保存数据表，兼容 Windows 表格软件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def save_joblib(path, obj):
    """压缩并原子保存 sklearn 模型或预处理器。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    joblib.dump(obj, temporary, compress=3)
    os.replace(temporary, path)


def set_seed(seed=SEED):
    """固定 Python、NumPy 和 PyTorch 随机数，限制 CPU 线程并关闭不确定的卷积选择。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_base_models():
    """返回正式实验使用的三种第一层算法及其冻结参数。"""
    return {
        "SVM": SVR(C=100, gamma="auto"),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=1),
        "XGBoost": XGBRegressor(
            n_estimators=250, learning_rate=0.05, max_depth=6, random_state=SEED, verbosity=0, n_jobs=1
        ),
    }


def make_meta_models():
    """生成五种二层基线，非线性模型与相应第一层算法使用相同固定参数。"""
    bases = make_base_models()
    return {
        "Meta-LR": LinearRegression(),
        "Meta-Ridge": Ridge(alpha=1.0, random_state=SEED),
        "Meta-XGB": bases["XGBoost"],
        "Meta-RF": bases["RandomForest"],
        "Meta-SVM": bases["SVM"],
    }


def clean_training(frame, columns, target):
    """仅对80%训练集依次做 IQR(0.8) 与 3σ 清洗，返回保留位置及拟合的标准化器。"""
    y = frame[target].to_numpy(dtype=float)
    q1, q3 = np.percentile(y, [25, 75])
    low, high = q1 - 0.8 * (q3 - q1), q3 + 0.8 * (q3 - q1)
    kept = np.flatnonzero((y >= low) & (y <= high))
    mean, std = float(y[kept].mean()), float(y[kept].std())
    kept = kept[(y[kept] >= mean - 3 * std) & (y[kept] <= mean + 3 * std)]
    x = frame[columns].to_numpy(dtype=float)[kept]
    if len(kept) < 10 or not np.isfinite(x).all() or not np.isfinite(y[kept]).all():
        raise ValueError("训练集有效样本不足十个或存在非有限值")
    sx, sy = StandardScaler(), StandardScaler()
    xs = sx.fit_transform(x)
    ys = sy.fit_transform(y[kept, None]).ravel()
    prep = dict(
        scaler_x=sx,
        scaler_y=sy,
        feature_names=columns,
        target=target,
        iqr_bounds=[float(low), float(high)],
        sigma_bounds=[mean - 3 * std, mean + 3 * std],
    )
    return kept, xs, ys, prep


def train_fold_models(estimator, x, y, fold_dir=None):
    """每折只预测未参与本折训练的样本，保存每折模型并验证每行恰好预测一次。"""
    models, oof = [], np.full(len(y), np.nan)
    seen, fold_ids = np.zeros(len(y), dtype=int), np.full(len(y), -1)
    for fold, (tr, va) in enumerate(KFold(n_splits=10, shuffle=True, random_state=SEED).split(x)):
        assert not np.intersect1d(tr, va).size
        path = Path(fold_dir) / f"fold_{fold:02d}.joblib" if fold_dir else None
        if path and path.exists():
            model = joblib.load(path)
        else:
            model = clone(estimator).fit(x[tr], y[tr])
            if path:
                save_joblib(path, model)
        oof[va] = model.predict(x[va])
        seen[va] += 1
        fold_ids[va] = fold
        models.append(model)
        log(f"fold {fold+1}/10 ready; train={len(tr)} holdout={len(va)}")
    assert (seen == 1).all() and np.isfinite(oof).all()
    return models, oof, fold_ids


def predict_base_feature_matrix(groups, x):
    """对测试或新样本取每种基算法十个模型的均值；输出三列且不使用目标标签。"""
    result = []
    for name in BASE_NAMES:
        if len(groups[name]) != 10:
            raise ValueError(f"{name} 必须恰好有十个折模型")
        result.append(np.column_stack([m.predict(x) for m in groups[name]]).mean(axis=1))
    return np.column_stack(result)


def prepare_dataset(name, run_dir):
    """在独立运行目录生成固定划分、训练预处理及可恢复的第一层 OOF 缓存。"""
    folder = Path(run_dir) / "datasets" / name
    folder.mkdir(parents=True, exist_ok=True)
    ready = folder / "cache_complete.json"
    if ready.exists():
        return folder, dict(np.load(folder / "train_arrays.npz"))
    start = synchronized_time()
    frame, raw, group_ids, audit = grouped_data(name)
    write_csv(folder / "group_mapping.csv", pd.DataFrame(dict(raw_row=np.arange(len(raw)), group_id=group_ids)))
    write_json(folder / "data_audit.json", audit)
    columns, target = feature_columns(frame)
    protocol = read_json(Path(run_dir) / "protocol.json")
    train_ids, test_ids = split_indices(len(frame), protocol.get("outer_fold"))
    train = frame.iloc[train_ids].reset_index(drop=True)
    test = frame.iloc[test_ids].reset_index(drop=True)
    kept, x, y, prep = clean_training(train, columns, target)
    write_csv(folder / "train.csv", train)
    write_csv(folder / "test.csv", test)
    save_joblib(folder / "preprocess.joblib", prep)
    preprocessing_seconds = synchronized_time() - start
    base_start = synchronized_time()
    oof_cols, fold_ids = [], None
    for base_name, estimator in make_base_models().items():
        log(f"{name} BASE {base_name}")
        group, oof, current_fold_ids = train_fold_models(estimator, x, y, folder / "base_models" / base_name)
        if fold_ids is not None:
            np.testing.assert_array_equal(fold_ids, current_fold_ids)
        fold_ids = current_fold_ids
        oof_cols.append(oof)
        del group
    oof = np.column_stack(oof_cols)
    arrays = dict(
        x=x,
        y=y,
        y_orig=train[target].to_numpy(dtype=float)[kept],
        oof=oof,
        train_ids=train_ids[kept],
        outer_train_ids=train_ids,
        test_ids=test_ids,
        fold_ids=fold_ids,
    )
    np.savez_compressed(folder / "train_arrays.npz", **arrays)
    write_csv(
        folder / "oof_features.csv",
        pd.DataFrame(
            dict(
                row_id=arrays["train_ids"],
                fold=fold_ids,
                SVM=oof[:, 0],
                RandomForest=oof[:, 1],
                XGBoost=oof[:, 2],
            )
        ),
    )
    write_csv(
        folder / "split_rows.csv",
        pd.DataFrame(
            dict(
                row_id=np.concatenate([train_ids, test_ids]),
                split=["train"] * len(train_ids) + ["test"] * len(test_ids),
            )
        ),
    )
    write_json(
        ready,
        dict(
            raw_rows=len(raw),
            grouped_rows=len(frame),
            preprocessing_seconds=preprocessing_seconds,
            base_fit_seconds=synchronized_time()-base_start,
            train_before_clean=len(train),
            train_after_clean=len(kept),
            test_rows=len(test),
            features=len(columns),
            folds=10,
        ),
    )
    log(f"{name} CACHE READY train={len(kept)} test={len(test)} features={len(columns)}")
    return folder, arrays


class AdaptiveMLP(nn.Module):
    """4.3/4.4 的回归网络：全连接、BatchNorm、LeakyReLU、Dropout。"""

    def __init__(self, input_dim, hidden_dims, dropout):
        """根据冻结参数创建隐藏层及标量回归输出层。"""
        super().__init__()
        layers = []
        for size in hidden_dims:
            layers += [nn.Linear(input_dim, size), nn.BatchNorm1d(size), nn.LeakyReLU(), nn.Dropout(dropout)]
            input_dim = size
        self.net = nn.Sequential(*layers, nn.Linear(input_dim, 1))

    def forward(self, x):
        """将输入特征映射为一维预测。"""
        return self.net(x).squeeze(-1)


def combine_nn_stacking_outputs(network_output, base_predictions):
    """按论文公式计算三个无约束权重乘基预测之和，再加样本修正项。"""
    if network_output.ndim != 2 or network_output.shape[1] != len(BASE_NAMES) + 1:
        raise ValueError("NN-Stacking网络必须输出三个权重和一个修正项")
    if base_predictions.ndim != 2 or base_predictions.shape[1] != len(BASE_NAMES):
        raise ValueError("NN-Stacking必须接收三列基模型预测")
    weights = network_output[:, : len(BASE_NAMES)]
    correction = network_output[:, len(BASE_NAMES)]
    return (weights * base_predictions).sum(dim=1) + correction


class UNNSWithCorrection(nn.Module):
    """NN-Stacking基线：只由原始特征生成样本权重与修正项，再组合三列基预测。"""

    def __init__(self, raw_input_dim, hidden_dims, dropout):
        """沿用当前数据集的隐藏层、BatchNorm、LeakyReLU和Dropout配置。"""
        super().__init__()
        layers = []
        input_dim = raw_input_dim
        for size in hidden_dims:
            layers += [nn.Linear(input_dim, size), nn.BatchNorm1d(size), nn.LeakyReLU(), nn.Dropout(dropout)]
            input_dim = size
        self.weight_network = nn.Sequential(*layers, nn.Linear(input_dim, len(BASE_NAMES) + 1))

    def forward(self, raw_features, base_predictions):
        """权重网络只读取原始特征；基预测只在最终线性组合处参与计算。"""
        return combine_nn_stacking_outputs(self.weight_network(raw_features), base_predictions)


class PureSigmoidMLP(nn.Module):
    """4.1/4.2 的纯全连接 Sigmoid 网络，不包含 Dropout 和 BatchNorm。"""

    def __init__(self, input_dim, hidden_dims):
        """按给定隐藏层大小建立纯 Sigmoid 网络。"""
        super().__init__()
        layers = []
        for size in hidden_dims:
            layers += [nn.Linear(input_dim, size), nn.Sigmoid()]
            input_dim = size
        self.net = nn.Sequential(*layers, nn.Linear(input_dim, 1))

    def forward(self, x):
        """返回每个样本的标准化目标预测值。"""
        return self.net(x).squeeze(-1)


def predict_mlp(model, x, batch_size):
    """在评估模式下分批预测，禁用 Dropout 并复用 BatchNorm 统计量。"""
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            features = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=DEVICE)
            result.append(model(features).cpu().numpy())
    return np.concatenate(result)


def evaluate_loss(model, x, y, batch_size):
    """按最终实验口径分批以 float32 计算完整训练集 MSELoss，供权重选择与绘图。"""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb, yb = x[start : start + batch_size], y[start : start + batch_size]
            total += nn.functional.mse_loss(model(xb), yb).item() * len(xb)
    return total / len(x)


def predict_nn_stacking(model, raw_features, base_predictions, batch_size):
    """分批计算NN-Stacking预测，保持权重网络与三列基预测的输入边界。"""
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(raw_features), batch_size):
            raw = torch.as_tensor(
                raw_features[start : start + batch_size], dtype=torch.float32, device=DEVICE
            )
            base = torch.as_tensor(
                base_predictions[start : start + batch_size], dtype=torch.float32, device=DEVICE
            )
            result.append(model(raw, base).cpu().numpy())
    return np.concatenate(result)


def evaluate_nn_stacking_loss(model, raw_features, base_predictions, y, batch_size):
    """在完整训练集上计算NN-Stacking的标准化MSE，供同一检查点规则选择。"""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(y), batch_size):
            raw = raw_features[start : start + batch_size]
            base = base_predictions[start : start + batch_size]
            target = y[start : start + batch_size]
            total += nn.functional.mse_loss(model(raw, base), target).item() * len(target)
    return total / len(y)


def train_nn_stacking(raw_features, base_predictions, y, config, progress_name, checkpoint_dir=None, validation=None):
    """仅用训练集原始特征、严格OOF基预测和标签训练UNNS加修正项网络。"""
    set_seed()
    raw = torch.as_tensor(raw_features, dtype=torch.float32, device=DEVICE)
    base = torch.as_tensor(base_predictions, dtype=torch.float32, device=DEVICE)
    target = torch.as_tensor(y, dtype=torch.float32, device=DEVICE)
    model = UNNSWithCorrection(raw_features.shape[1], config["hidden_dims"], config["dropout"]).to(DEVICE)
    batch = int(config["batch_size"])
    if len(y) % batch == 1:
        batch += 1
    loader = DataLoader(TensorDataset(raw, base, target), batch_size=batch, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = (
        None
        if not config.get("use_lr_scheduler", True)
        else torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=40)
    )
    best_loss, best_epoch = float("inf"), 0
    best = deepcopy(model.state_dict())
    curves = []
    start = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total = 0.0
        for raw_batch, base_batch, target_batch in loader:
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(raw_batch, base_batch), target_batch)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(target_batch)
        train_loss = total / len(y)
        if scheduler:
            scheduler.step(train_loss)
        do_eval = epoch % 10 == 0 or epoch == config["epochs"]
        eval_loss = evaluate_nn_stacking_loss(model, raw, base, target, batch) if do_eval else None
        if do_eval and eval_loss < best_loss:
            best_loss, best_epoch = eval_loss, epoch
            best = {key: value.detach().clone() for key, value in model.state_dict().items()}
        validation_mse = None
        if validation is not None and epoch in config.get("candidates", []):
            validation_mse = float(np.mean((predict_nn_stacking(model, validation[0], validation[1], batch) - validation[2])**2))
        if config.get("selection") == "fixed_endpoint" and epoch == config["epochs"]:
            best_loss, best_epoch = eval_loss, epoch
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
        curves.append(
            dict(
                epoch=epoch,
                validation_mse=validation_mse,
                train_batch_mse=train_loss,
                eval_mse=eval_loss,
                learning_rate=optimizer.param_groups[0]["lr"],
            )
        )
        if epoch == 1 or epoch % 100 == 0 or epoch == config["epochs"]:
            log(
                f'{progress_name} epoch {epoch}/{config["epochs"]}; best={best_loss:.9g}; elapsed={time.perf_counter()-start:.1f}s'
            )
            if checkpoint_dir:
                write_csv(Path(checkpoint_dir) / "curve_in_progress.csv", pd.DataFrame(curves))
    model.load_state_dict(best)
    info = dict(
        config=config,
        pure=False,
        optimizer="AdamW",
        input_dim=raw_features.shape[1],
        raw_input_dim=raw_features.shape[1],
        base_model_order=BASE_NAMES,
        correction=True,
        model_type="nn_stacking_unns_correction",
        weights_constraint="none",
        best_epoch=best_epoch,
        best_eval_loss=best_loss,
        actual_batch_size=batch,
        fit_seconds=time.perf_counter() - start,
        selection=config.get("selection", "training_eval_mse"),
    )
    return model, info, pd.DataFrame(curves)


def train_mlp(x, y, config, pure=False, progress_name="MLP", checkpoint_dir=None, validation=None):
    """只用训练数据通过 AdamW 优化，保留最低训练评估损失及逐轮日志。"""
    set_seed()
    xs = torch.as_tensor(x, dtype=torch.float32, device=DEVICE)
    ys = torch.as_tensor(y, dtype=torch.float32, device=DEVICE)
    model = (
        PureSigmoidMLP(x.shape[1], config["hidden_dims"])
        if pure
        else AdaptiveMLP(x.shape[1], config["hidden_dims"], config["dropout"])
    ).to(DEVICE)
    batch = int(config["batch_size"])
    # BatchNorm 不支持单个训练样本；合并可能出现的最后一个单样本批，不丢样本。
    if not pure and len(y) % batch == 1:
        batch += 1
    loader = DataLoader(TensorDataset(xs, ys), batch_size=batch, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    # 定向轮数扫描使用固定学习率时可关闭调度器，确保主代码能复现选中的检查点。
    scheduler = (
        None
        if pure or not config.get("use_lr_scheduler", True)
        else torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=40)
    )
    best_loss, best_epoch = float("inf"), 0
    best = deepcopy(model.state_dict())
    curves = []
    start = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        train_loss = total / len(y)
        if scheduler:
            scheduler.step(train_loss)
        do_eval = pure or epoch % 10 == 0 or epoch == config["epochs"]
        eval_loss = evaluate_loss(model, xs, ys, batch) if do_eval else None
        if do_eval and eval_loss < best_loss:
            best_loss, best_epoch = eval_loss, epoch
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
        validation_mse = None
        if validation is not None and epoch in config.get("candidates", []):
            validation_mse = float(np.mean((predict_mlp(model, validation[0], batch) - validation[1])**2))
        if config.get("selection") == "fixed_endpoint" and epoch == config["epochs"]:
            best_loss, best_epoch = eval_loss, epoch
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
        curves.append(
            dict(
                epoch=epoch,
                validation_mse=validation_mse,
                train_batch_mse=train_loss,
                eval_mse=eval_loss,
                learning_rate=opt.param_groups[0]["lr"],
            )
        )
        if epoch == 1 or epoch % 100 == 0 or epoch == config["epochs"]:
            log(
                f'{progress_name} epoch {epoch}/{config["epochs"]}; best={best_loss:.9g}; elapsed={time.perf_counter()-start:.1f}s'
            )
            if checkpoint_dir:
                write_csv(Path(checkpoint_dir) / "curve_in_progress.csv", pd.DataFrame(curves))
    model.load_state_dict(best)
    info = dict(
        config=config,
        pure=pure,
        optimizer="AdamW",
        input_dim=x.shape[1],
        best_epoch=best_epoch,
        best_eval_loss=best_loss,
        actual_batch_size=batch,
        fit_seconds=time.perf_counter() - start,
        selection=config.get("selection", "training_eval_mse"),
    )
    return model, info, pd.DataFrame(curves)


def select_epochs(name, folder):
    """仅在外层训练数据内部选轮数；重新拟合预处理与基模型，完全隔离验证标签。"""
    if name not in EPOCH_CANDIDATES:
        return {}
    dest = Path(folder) / "epoch_selection"
    ready = dest / "selection.json"
    if ready.exists():
        return read_json(ready)["selected"]
    dest.mkdir(parents=True, exist_ok=True)
    candidates = EPOCH_CANDIDATES[name]
    train = pd.read_csv(Path(folder) / "train.csv", encoding="utf-8-sig")
    columns, target = feature_columns(train)
    tr, va = train_test_split(np.arange(len(train)), test_size=0.2, random_state=SEED)
    arrays_path = dest / "validation_arrays.npz"
    if arrays_path.exists():
        a = dict(np.load(arrays_path))
    else:
        kept, x, y, prep = clean_training(train.iloc[tr], columns, target)
        vx = prep["scaler_x"].transform(train.iloc[va][columns].to_numpy(dtype=float))
        vy = prep["scaler_y"].transform(train.iloc[va][[target]].to_numpy(dtype=float)).ravel()
        ps, vps = [], []
        start = synchronized_time()
        for key, estimator in make_base_models().items():
            models, oof, _ = train_fold_models(estimator, x, y, dest / "base_models" / key)
            ps.append(oof); vps.append(np.column_stack([m.predict(vx) for m in models]).mean(axis=1))
        a = dict(x=x, y=y, oof=np.column_stack(ps), vx=vx, vy=vy, voof=np.column_stack(vps), train_ids=tr[kept], validation_ids=va)
        np.savez_compressed(arrays_path, **a)
        save_joblib(dest / "preprocess.joblib", prep)
        write_json(dest / "protocol.json", dict(validation_fraction=0.2, seed=SEED, base_fit_seconds=synchronized_time()-start, test_used=False))
    selected = {}
    for method in ["Proposed", "E2E-MLP", "NN-Stacking"]:
        curve_path = dest / f"{method}.csv"
        if curve_path.exists():
            curve = pd.read_csv(curve_path)
        else:
            config = mlp_config(name)
            config.update(epochs=max(candidates), candidates=candidates)
            if method == "NN-Stacking":
                model, _, curve = train_nn_stacking(a["x"], a["oof"], a["y"], config,
                    f"{name} validation {method}", validation=(a["vx"], a["voof"], a["vy"]))
            else:
                x = a["x"] if method == "E2E-MLP" else np.column_stack([a["x"], a["oof"]])
                vx = a["vx"] if method == "E2E-MLP" else np.column_stack([a["vx"], a["voof"]])
                model, _, curve = train_mlp(x, a["y"], config, progress_name=f"{name} validation {method}", validation=(vx, a["vy"]))
            write_csv(curve_path, curve)
            del model
        scores = curve.loc[curve.epoch.isin(candidates)].sort_values(["validation_mse", "epoch"])
        if len(scores) != len(candidates) or scores.validation_mse.isna().any():
            raise ValueError("候选轮数验证结果不完整")
        selected[method] = int(scores.iloc[0].epoch)
    write_json(ready, dict(selected=selected, candidates=candidates, criterion="internal_validation_MSE_ties_shorter", test_used=False))
    return selected


def run_dedup_diagnostic(name, run_dir):
    """CO/SC固定轮数比较合并与重复训练；共享唯一测试输入，组内样本不会跨OOF折。"""
    if name not in EPOCH_CANDIDATES:
        return
    folder = Path(run_dir) / "datasets" / name
    dest = folder / "dedup_diagnostic"
    ready = dest / "complete.json"
    if ready.exists():
        return
    grouped, raw, group_ids, _ = grouped_data(name)
    formal = dict(np.load(folder / "train_arrays.npz"))
    test = grouped.iloc[formal["test_ids"]]
    columns, target = feature_columns(grouped)
    rows = []
    for regime in ["group_mean", "repeated_rows"]:
        branch = dest / regime
        branch.mkdir(parents=True, exist_ok=True)
        cache = branch / "arrays.npz"
        if regime == "group_mean":
            a = formal
            prep = joblib.load(folder / "preprocess.joblib")
            groups = load_base_groups(folder)
        elif cache.exists():
            a = dict(np.load(cache))
            prep = joblib.load(branch / "preprocess.joblib")
            groups = load_base_groups(branch)
        else:
            mask = np.isin(group_ids, formal["outer_train_ids"])
            repeated = raw.loc[mask].reset_index(drop=True)
            kept, x, y, prep = clean_training(repeated, columns, target)
            gids = group_ids[mask][kept]
            unique = pd.unique(gids)
            groups, ps = {}, []
            for key, estimator in make_base_models().items():
                oof = np.full(len(y), np.nan); models = []
                for k, (tr, va) in enumerate(KFold(10, shuffle=True, random_state=SEED).split(unique)):
                    train_mask, val_mask = np.isin(gids, unique[tr]), np.isin(gids, unique[va])
                    path = branch / "base_models" / key / f"fold_{k:02d}.joblib"
                    model = joblib.load(path) if path.exists() else clone(estimator).fit(x[train_mask], y[train_mask])
                    save_joblib(path, model)
                    oof[val_mask] = model.predict(x[val_mask]); models.append(model)
                assert np.isfinite(oof).all()
                ps.append(oof); groups[key] = models
            a = dict(x=x, y=y, oof=np.column_stack(ps))
            np.savez_compressed(cache, **a); save_joblib(branch / "preprocess.joblib", prep)
        tx = prep["scaler_x"].transform(test[columns].to_numpy(dtype=float))
        tp = predict_base_feature_matrix(groups, tx)
        del groups
        for method in ["Proposed", "E2E-MLP", "NN-Stacking"]:
            model_dir = branch / method
            complete = model_dir / "complete.json"
            if complete.exists():
                rows.append(read_json(complete)); continue
            config = mlp_config(name); config["selection"] = "fixed_endpoint"
            if method == "NN-Stacking":
                model, info, curve = train_nn_stacking(a["x"], a["oof"], a["y"], config, f"{name} {regime} fixed {method}")
                pred = predict_nn_stacking(model, tx, tp, info["actual_batch_size"])
            else:
                x = a["x"] if method == "E2E-MLP" else np.column_stack([a["x"], a["oof"]])
                testx = tx if method == "E2E-MLP" else np.column_stack([tx, tp])
                model, info, curve = train_mlp(x, a["y"], config, progress_name=f"{name} {regime} fixed {method}")
                pred = predict_mlp(model, testx, info["actual_batch_size"])
            save_mlp(model_dir, model, info); write_csv(model_dir / "loss_curve.csv", curve)
            row = prediction_row(name, method, "test", regime, formal["test_ids"], test[target].to_numpy(), pred, prep, model_dir / "predictions.csv", info["fit_seconds"])
            row.update(epochs=config["epochs"], regime=regime)
            write_json(complete, row); rows.append(row)
            del model
    write_csv(dest / "metrics.csv", pd.DataFrame(rows))
    write_json(ready, dict(rows=len(rows), common_unique_test=True, repeated_inner_folds_group_safe=True, formal_ranking=False))


def save_mlp(folder, model, info):
    """使用文件句柄保存 PyTorch 权重，兼容含中文的 Windows 路径。"""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    pending = folder / "model.pt.pending"
    with pending.open("wb") as stream:
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, stream)
    os.replace(pending, folder / "model.pt")
    write_json(folder / "model.json", info)


def load_mlp(folder):
    """从本目录的结构参数及权重重新创建 MLP，供独立预测及重载核验。"""
    folder = Path(folder)
    info = read_json(folder / "model.json")
    c = info["config"]
    if info.get("model_type") == "nn_stacking_unns_correction":
        model = UNNSWithCorrection(info["raw_input_dim"], c["hidden_dims"], c["dropout"]).to(DEVICE)
    else:
        model = (
            PureSigmoidMLP(info["input_dim"], c["hidden_dims"])
            if info["pure"]
            else AdaptiveMLP(info["input_dim"], c["hidden_dims"], c["dropout"])
        ).to(DEVICE)
    with (folder / "model.pt").open("rb") as stream:
        model.load_state_dict(torch.load(stream, map_location=DEVICE, weights_only=True))
    model.eval()
    return model, info


def metrics(y, p):
    """在原始目标尺度计算四项指标，沿用最终实验 MAPE 的近零保护规则。"""
    if not np.isfinite(p).all():
        raise ValueError("预测包含非有限值")
    safe = np.where(np.abs(y) < 1e-8, 1e-8, y)
    return dict(
        R2=float(r2_score(y, p)),
        MAPE=float(np.mean(np.abs((y - p) / safe)) * 100),
        MAE=float(np.mean(np.abs(y - p))),
        RMSE=float(np.sqrt(np.mean((y - p) ** 2))),
    )


def prediction_row(name, method, split, feature_set, ids, y, scaled_pred, prep, output, fit_seconds=0.0):
    """保存逐样本预测并生成可反算指标行，明确训练与测试的输入口径。"""
    pred = prep["scaler_y"].inverse_transform(np.asarray(scaled_pred).reshape(-1, 1)).ravel()
    write_csv(output, pd.DataFrame(dict(row_id=ids, y_true=y, y_pred=pred, pred_scaled=scaled_pred)))
    y_scaled = prep["scaler_y"].transform(np.asarray(y).reshape(-1, 1)).ravel()
    return dict(
        dataset=name,
        method=method,
        split=split,
        feature_set=feature_set,
        n=len(y),
        **metrics(y, pred),
        MSELoss=float(np.mean((y_scaled - scaled_pred) ** 2)),
        fit_seconds=fit_seconds,
    )


def mlp_config(name, pure=False):
    """返回独立参数副本，纯网络复用冻结的隐藏层维度。"""
    config = deepcopy(FIXED_MLP_CONFIGS[name])
    if pure:
        config.update(FIT_CONFIGS[name], hidden_dims=deepcopy(FIT_HIDDEN_DIMS[name]), dropout=0.0, weight_decay=0.0)
    return config


def load_base_groups(folder):
    """加载第一层三类模型，每类恰好十个，与训练缓存的折顺序一致。"""
    return {
        name: [joblib.load(Path(folder) / "base_models" / name / f"fold_{k:02d}.joblib") for k in range(10)]
        for name in BASE_NAMES
    }


SECTION_METRICS = {
    "41": ["R2", "MAPE", "MAE", "RMSE", "MSELoss"],
    "42": ["MSELoss"],
    "43": ["R2", "MAPE", "MAE", "RMSE", "MSELoss"],
    "44": ["R2", "MAPE", "MAE", "RMSE", "MSELoss"],
}

# 4.3和4.4的综合排名只使用论文表格展示的四项指标；MSELoss仅用于统一统计检验口径。
RANK_METRICS = {
    "41": ["R2", "MAPE", "MAE", "RMSE", "MSELoss"],
    "42": ["MSELoss"],
    "43": ["R2", "MAPE", "MAE", "RMSE"],
    "44": ["R2", "MAPE", "MAE", "RMSE"],
}


_NULL_CACHE = {}


def method_key(frame: pd.DataFrame, section: str) -> pd.Series:
    """为4.2把模型与输入方式合成唯一名称，其余实验直接使用方法名。"""
    if section == "42":
        return frame["method"].astype(str) + "|" + frame["feature_set"].astype(str)
    return frame["method"].astype(str)


def summarize_values(frame: pd.DataFrame, section: str) -> pd.DataFrame:
    """按数据集与方法计算五折均值和样本标准差，并保留原始方法/输入名称。"""
    group = ["dataset", "method_key"]
    numeric = SECTION_METRICS[section]
    result = frame.groupby(group)[numeric].agg(["mean", "std"]).reset_index()
    result.columns = ["_".join(part for part in column if part) for column in result.columns]
    return result.rename(columns={"dataset_": "dataset", "method_key_": "method_key"})


def rank_values(frame: pd.DataFrame, section: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """先在每个数据集和80:20划分内排名，再汇总为每数据集及全部六数据集平均排名。"""
    ranked = frame.copy()
    rank_columns = []
    for metric in RANK_METRICS[section]:
        column = f"{metric}_rank"
        ranked[column] = ranked.groupby(["dataset", "outer_fold"])[metric].rank(
            method="average", ascending=metric != "R2"
        )
        rank_columns.append(column)
    ranked["mean_rank"] = ranked[rank_columns].mean(axis=1)
    by_dataset = ranked.groupby(["dataset", "method_key"], as_index=False)[
        rank_columns + ["mean_rank"]
    ].mean()
    overall = by_dataset.groupby("method_key", as_index=False)[rank_columns + ["mean_rank"]].mean()
    overall["overall_rank"] = overall["mean_rank"].rank(method="min")
    return by_dataset, overall.sort_values(["overall_rank", "mean_rank", "method_key"])


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    """对一组原始P值执行逐步拒绝式Holm多重比较校正。"""
    values = np.asarray(list(p_values), dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for position, index in enumerate(order):
        running = max(running, (count - position) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def primary_metric(section: str) -> str:
    """四项实验统一以训练目标MSELoss作为主统计指标。"""
    return "MSELoss"


def validate_metrics(frame: pd.DataFrame, section: str) -> None:
    """拒绝重复、缺折、缺方法、非有限指标，防止静默平均不完整实验。"""
    expected = (
        ["MLP", "LR", "Ridge"]
        if section == "41"
        else (
            [f"{m}|{f}" for m in ("MLP", "LR", "Ridge") for f in ("base_only", "base_plus_raw")]
            if section == "42"
            else METHODS
        )
    )
    if set(frame.dataset) != set(DATASETS) or set(frame.outer_fold) != set(range(5)):
        raise ValueError(f"4.{section[-1]} 数据集或折号不完整")
    if not frame.method_key.equals(method_key(frame, section)):
        raise ValueError("method_key与方法/输入名称不一致")
    if len(frame) != len(DATASETS) * 5 * len(expected):
        raise ValueError("数据集、折号和方法的组合不完整")
    if frame.duplicated(["dataset", "outer_fold", "method_key"]).any():
        raise ValueError("出现重复的数据集-折-方法记录")
    for _, group in frame.groupby(["dataset", "outer_fold"]):
        if set(group.method_key) != set(expected):
            raise ValueError("某个折的方法不完整")
    if not np.isfinite(frame[SECTION_METRICS[section]].to_numpy()).all():
        raise ValueError("存在缺失或非有限指标")


def rank_statistics(ranks: np.ndarray) -> tuple[float, float, float]:
    """由逐行名次计算含并列修正的Friedman Q、Kendall W及共同的秩和离差平方和。"""
    blocks, methods = ranks.shape
    centered = ranks - (methods + 1) / 2
    denominator = float(np.square(centered).sum())
    score = float(np.square(centered.sum(axis=0)).sum())
    if denominator == 0:
        return 0.0, float("nan"), 0.0
    return (methods - 1) * score / denominator, score / (blocks * denominator), score


def permutation_tail(ranks: np.ndarray, draws: int, seed: int) -> dict:
    """逐块置换方法标签；小空间穷举，完全同序且无并列时精确计数，其余使用固定种子的Monte Carlo。"""
    blocks, methods = ranks.shape
    _, concordance, observed = rank_statistics(ranks)
    space = math.factorial(methods) ** (blocks - 1)
    no_ties = all(len(np.unique(row)) == methods for row in ranks)
    if not np.isfinite(concordance):
        return {
            "p": 1.0,
            "calibration": "all_tied_no_information",
            "draws": 0,
            "seed": seed,
            "extreme_count": 0,
            "permutation_space": str(space),
        }
    if no_ties and np.isclose(concordance, 1.0, rtol=0, atol=1e-12):
        return {
            "p": 1.0 / space,
            "calibration": "exact_maximum_no_ties",
            "draws": 0,
            "seed": seed,
            "extreme_count": 1,
            "permutation_space": str(space),
        }
    # 共同重标记不会改变秩和平方，因此固定第一行，其余行独立置换。
    canonical = np.sort(ranks, axis=1) - (methods + 1) / 2
    key = (tuple(map(tuple, canonical)), draws, seed)
    exact = space <= 200000
    if key not in _NULL_CACHE:
        if exact:
            permutations = list(itertools.permutations(range(methods)))
            scores = []
            for choices in itertools.product(permutations, repeat=blocks - 1):
                sums = canonical[0].copy()
                for row, order in zip(canonical[1:], choices):
                    sums += row[list(order)]
                scores.append(float(np.square(sums).sum()))
            null = np.asarray(scores)
        else:
            rng = np.random.default_rng(seed)
            pieces = []
            for start in range(0, draws, 10000):
                size = min(10000, draws - start)
                sums = np.broadcast_to(canonical[0], (size, methods)).copy()
                for row in canonical[1:]:
                    order = np.argsort(rng.random((size, methods)), axis=1)
                    sums += row[order]
                pieces.append(np.square(sums).sum(axis=1))
            null = np.concatenate(pieces)
        _NULL_CACHE[key] = np.sort(null)
    null = _NULL_CACHE[key]
    extreme = int(len(null) - np.searchsorted(null, observed - 1e-10, side="left"))
    p_value = extreme / len(null) if exact else (extreme + 1) / (len(null) + 1)
    return {
        "p": p_value,
        "calibration": "exact_enumeration" if exact else "monte_carlo_plus_one",
        "draws": len(null),
        "seed": seed,
        "extreme_count": extreme,
        "permutation_space": str(space),
        "mc_resolution": None if exact else 1 / (len(null) + 1),
        "mc_standard_error_approx": None if exact else math.sqrt(p_value * (1 - p_value) / (len(null) + 1)),
    }


def statistical_test(frame: pd.DataFrame, section: str, metric: str, draws: int, seed: int) -> dict:
    """目标一：以六数据集五折均值形成的名次检验算法总体差异；不用小样本卡方近似作结论。"""
    means = frame.groupby(["dataset", "method_key"], as_index=False)[metric].mean()
    means["rank"] = means.groupby("dataset")[metric].rank(method="average", ascending=metric != "R2")
    table = means.pivot(index="dataset", columns="method_key", values="rank").reindex(DATASETS)
    methods = list(table.columns)
    statistic, _, _ = rank_statistics(table.to_numpy())
    permutation = permutation_tail(table.to_numpy(), draws, seed)
    scipy_q, approximate_p = friedmanchisquare(*[table[name].to_numpy() for name in methods])
    if not np.isclose(statistic, scipy_q):
        raise RuntimeError("Friedman Q与SciPy独立公式不一致")
    omnibus = {
        "section": section,
        "metric": metric,
        "primary_metric": metric == primary_metric(section),
        "unit": "six dataset-level means; each mean contains five 80:20 splits",
        "methods": methods,
        "n_datasets": len(DATASETS),
        "friedman_statistic": float(statistic),
        "friedman_p": permutation["p"],
        "chi_square_p_reference_only": float(approximate_p),
        "permutation": permutation,
        "null_hypothesis": "method labels exchangeable within each independent dataset block",
        "dataset_rank_matrix": table.to_dict(orient="index"),
    }
    return omnibus


def split_concordance(
    frame: pd.DataFrame, section: str, metric: str, draws: int, seed: int
) -> tuple[list[dict], list[dict]]:
    """目标二：逐数据集量化五折方法排名一致性；置换P值仅为忽略折间依赖的条件探索性参照。"""
    concordance_rows, dispersion_rows = [], []
    for dataset in DATASETS:
        table = (
            frame.loc[frame.dataset == dataset]
            .pivot(index="outer_fold", columns="method_key", values=metric)
            .sort_index()
        )
        values = table.to_numpy()
        ranks = rankdata(-values if metric == "R2" else values, axis=1, method="average")
        q_value, w_value, _ = rank_statistics(ranks)
        permutation = permutation_tail(ranks, draws, seed)
        concordance_rows.append(
            {
                "section": section,
                "dataset": dataset,
                "abbr": ABBR[dataset],
                "metric": metric,
                "primary_metric": metric == primary_metric(section),
                "n_folds": 5,
                "n_methods": len(table.columns),
                "kendall_w": w_value,
                "rank_q": q_value,
                "p_conditional": permutation["p"],
                "calibration": permutation["calibration"],
                "draws": permutation["draws"],
                "seed": seed,
                "extreme_count": permutation["extreme_count"],
                "permutation_space": permutation["permutation_space"],
                "inference_status": "exploratory_only_overlapping_training_folds",
                "null_hypothesis": "independent random method-label orders conditional on each fold rank multiset",
            }
        )
        for column, method in enumerate(table.columns):
            scores, positions = values[:, column], ranks[:, column]
            dispersion_rows.append(
                {
                    "section": section,
                    "dataset": dataset,
                    "abbr": ABBR[dataset],
                    "metric": metric,
                    "primary_metric": metric == primary_metric(section),
                    "method_key": method,
                    "mean": float(scores.mean()),
                    "sample_sd": float(scores.std(ddof=1)),
                    "min": float(scores.min()),
                    "max": float(scores.max()),
                    "rank_mean": float(positions.mean()),
                    "rank_sample_sd": float(positions.std(ddof=1)),
                    "rank_min": float(positions.min()),
                    "rank_max": float(positions.max()),
                    **{f"fold_{fold}_value": float(scores[fold]) for fold in range(5)},
                    **{f"fold_{fold}_rank": float(positions[fold]) for fold in range(5)},
                }
            )
    return concordance_rows, dispersion_rows


def save_section_summary(frame, section, output):
    """保存单节五折指标、均值、样本标准差与平均排名，供论文表格直接引用。"""
    summary = Path(output) / "summary"
    write_csv(summary / f"section{section}_all_folds.csv", frame)
    write_csv(summary / f"section{section}_mean_std.csv", summarize_values(frame, section))
    by_dataset, overall = rank_values(frame, section)
    write_csv(summary / f"section{section}_dataset_average_ranks.csv", by_dataset)
    write_csv(summary / f"section{section}_overall_average_ranks.csv", overall)


def summarize_results(source, output, permutations=999999, concordance_permutations=199999):
    """先确认四节共990行完整，再重算全部汇总和统计。"""
    omnibus, concordances, dispersions = [], [], []
    counts, input_hashes = {}, {}
    for section in ("41", "42", "43", "44"):
        validate_metrics(pd.read_csv(Path(source) / f"section{section}_all_folds.csv", encoding="utf-8-sig"), section)
    for section in ("41", "42", "43", "44"):
        path = Path(source) / f"section{section}_all_folds.csv"
        frame = pd.read_csv(path, encoding="utf-8-sig")
        validate_metrics(frame, section)
        input_hashes[path.name] = sha256(path)
        counts[section] = len(frame)
        save_section_summary(frame, section, output)
        for metric in SECTION_METRICS[section]:
            omnibus.append(statistical_test(frame, section, metric, permutations, 20260914))
            concordance, dispersion = split_concordance(
                frame, section, metric, concordance_permutations, 20260915
            )
            concordances.extend(concordance)
            dispersions.extend(dispersion)
    # 主要指标和补充指标分别校正，保持最终论文预先规定的分组。
    for primary in (True, False):
        selected = [row for row in omnibus if row["primary_metric"] == primary]
        for row, adjusted in zip(selected, holm_adjust(row["friedman_p"] for row in selected)):
            row.update(
                p_holm=float(adjusted),
                significant_0_05=bool(adjusted < 0.05),
                multiplicity_family="four_primary_tests" if primary else "twelve_secondary_tests",
            )
    concordance_frame = pd.DataFrame(concordances)
    for primary in (True, False):
        mask = concordance_frame.primary_metric == primary
        concordance_frame.loc[mask, "p_conditional_holm"] = holm_adjust(
            concordance_frame.loc[mask, "p_conditional"]
        )
        concordance_frame.loc[mask, "multiplicity_family"] = (
            "24_primary_conditional_tests" if primary else "72_secondary_conditional_tests"
        )
    write_json(Path(output) / "statistics" / "friedman.json", omnibus)
    write_csv(Path(output) / "statistics" / "kendall_concordance.csv", concordance_frame)
    write_csv(Path(output) / "statistics" / "split_dispersion.csv", pd.DataFrame(dispersions))
    write_json(
        Path(output) / "verification.json",
        dict(
            counts=counts,
            input_sha256=input_hashes,
            seed=20260914,
            friedman_permutations=permutations,
            concordance_permutations=concordance_permutations,
        ),
    )
    log(f"汇总完成：{counts}；输出到 {output}")


def collect_section(section, output, datasets):
    """收集本次运行各数据集五折结果；缺折时保留缓存，完整时才生成汇总。"""
    frames = []
    for fold in range(5):
        for name in datasets:
            path = run_path(output, fold) / "datasets" / name / f"section{section}" / "metrics.csv"
            if not path.exists():
                return False
            frame = pd.read_csv(path, encoding="utf-8-sig")
            frame["outer_fold"] = fold
            frame["method_key"] = method_key(frame, section)
            frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    if datasets != DATASETS:
        return False
    validate_metrics(frame, section)
    write_csv(Path(output) / "summary" / f"section{section}_all_folds.csv", frame)
    if section == "41":
        for name in datasets:
            curves = []
            for fold in range(5):
                path = run_path(output, fold) / "datasets" / name / "section41" / "loss_curve.csv"
                curve = pd.read_csv(path, encoding="utf-8-sig")
                curve.insert(0, "outer_fold", fold)
                curves.append(curve)
            write_csv(Path(output) / "curves" / f"{name}.csv", pd.concat(curves, ignore_index=True))
    return True


def output_directory(value):
    """检查复现输出路径，保护仓库中已发布的数据、代码和最终结果不被覆盖。"""
    path = Path(value).resolve()
    if path == ROOT or path in ROOT.parents:
        raise ValueError("输出目录不能是仓库根目录或它的上级目录")
    if ROOT in path.parents:
        relative = path.relative_to(ROOT).parts
        if relative[0] != "results":
            raise ValueError("仓库内输出请使用 results/reproduced；其他输出请指定仓库外目录")
    return path


def main(default_section="all"):
    """统一运行四项实验、五折统计或绘图；复现结果单独保存，可按相同命令续跑。"""
    global LOG_FILE, CACHE_ROOT
    parser = argparse.ArgumentParser(description="论文4.1–4.4复现：默认运行全部六数据集的五折实验。")
    parser.add_argument("--section", choices=["all", "41", "42", "43", "44"], default=default_section)
    parser.add_argument("--dataset", choices=["all"] + DATASETS, default="all")
    parser.add_argument("--fold", choices=["all", "0", "1", "2", "3", "4"], default="all")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--cache-prefix", type=Path, help="复用历史折目录前缀，后接0到4")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--summarize", action="store_true", help="从最终折级指标重算统计，不训练")
    actions.add_argument("--plot", action="store_true", help="从最终五折曲线重绘4.1，不训练")
    parser.add_argument("--input", type=Path, help="统计时为summary目录，绘图时为curves目录")
    args = parser.parse_args()
    CACHE_ROOT = args.cache_prefix
    output = output_directory(args.output)
    if args.summarize:
        summarize_results(args.input or ROOT / "results" / "summary", output)
        return
    if args.plot:
        from train_4_1 import plot_fivefold_curves

        plot_fivefold_curves(args.input or ROOT / "results" / "curves", output)
        return
    if args.input:
        parser.error("--input 仅与 --summarize 或 --plot 一起使用")
    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    folds = range(5) if args.fold == "all" else [int(args.fold)]
    sections = ["43", "44", "41", "42"] if args.section == "all" else [args.section]
    modules = {"41": "train_4_1", "42": "ablation_4_2", "43": "train_4_3", "44": "predict_4_4"}
    set_seed()
    for fold in folds:
        run = init_run(run_path(output, fold), fold, datasets)
        LOG_FILE = run / "run.log"
        for section in sections:
            execute = getattr(importlib.import_module(modules[section]), f"run_section{section}")
            for name in datasets:
                log(f"4.{section[-1]} / fold {fold} / {name}")
                execute(name, run)
        if args.section == "all":
            for name in datasets:
                run_dedup_diagnostic(name, run)
    LOG_FILE = None
    # 4.2已执行4.1对照，一并收集它的结果。
    for section in dict.fromkeys(sections + (["41"] if "42" in sections else [])):
        if not collect_section(section, output, datasets):
            log(f'4.{section[-1]} 尚未完成全部五折；当前结果保留在 {output / ".cache"}')
    if datasets == DATASETS:
        if all((output / "summary" / f"section{s}_all_folds.csv").exists() for s in modules):
            summarize_results(output / "summary", output)
        if all((output / "curves" / f"{name}.csv").exists() for name in DATASETS):
            from train_4_1 import plot_fivefold_curves

            plot_fivefold_curves(output / "curves", output)
    log(f"请求的实验已完成；输出到 {output}")


if __name__ == "__main__":
    main()
