"""ML pipeline: clustering, classification, and evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.analysis.export_utils import qp_sign_case, write_excel_csv
from src.analysis.industry_filters import apply_profile, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def _label_regime(row: pd.Series, cfg: dict) -> str:
    th = cfg.get("regime_thresholds", {})
    p_min = th.get("price_led_min_price_growth", 0.005)
    q_min = th.get("quantity_led_min_quantity_growth", 0.005)
    if row["avg_price_growth"] > p_min and row["avg_quantity_growth"] < 0:
        return "price_led"
    if row["avg_quantity_growth"] > q_min and row["avg_price_growth"] < p_min:
        return "quantity_led"
    return "balanced"


def _cluster_label(centroid: pd.Series) -> tuple[str, str]:
    ap, aq, vp, vq = (
        centroid["avg_price_growth"],
        centroid["avg_quantity_growth"],
        centroid["vol_price_growth"],
        centroid["vol_quantity_growth"],
    )
    vol = (vp + vq) / 2
    if vol > 0.03:
        return "high_volatility", f"High volatility (avg vol ~{vol:.3f})"
    if ap > aq and ap > 0.005:
        return "price_dominant", f"Higher avg price growth ({ap:.4f}) than quantity ({aq:.4f})"
    if aq > ap and aq > 0.005:
        return "quantity_dominant", f"Higher avg quantity growth ({aq:.4f}) than price ({ap:.4f})"
    return "stable_growth", f"Moderate, balanced growth (P={ap:.4f}, Q={aq:.4f})"


def _build_cluster_legend(features: pd.DataFrame, x_cols: list[str]) -> pd.DataFrame:
    rows = []
    for cid in sorted(features["cluster_id"].unique()):
        sub = features[features["cluster_id"] == cid]
        centroid = sub[x_cols].mean()
        label, desc = _cluster_label(centroid)
        rows.append(
            {
                "cluster_id": int(cid),
                "cluster_label": label,
                "description": desc,
                "n_industries": len(sub),
                "avg_price_growth": float(centroid["avg_price_growth"]),
                "avg_quantity_growth": float(centroid["avg_quantity_growth"]),
                "vol_price_growth": float(centroid["vol_price_growth"]),
                "vol_quantity_growth": float(centroid["vol_quantity_growth"]),
            }
        )
    return pd.DataFrame(rows)


def _industry_features(
    bea_growth: pd.DataFrame,
    bea_industries: pd.DataFrame,
    profile: str,
    period_start: str,
    period_end: str,
) -> pd.DataFrame:
    panel = apply_profile(
        bea_growth,
        bea_industries,
        profile,
        period_start=period_start,
        period_end=period_end,
    )
    panel = panel.dropna(subset=["price_growth", "quantity_growth"])

    features = (
        panel.groupby(["line_id", "industry_name"], as_index=False)
        .agg(
            avg_price_growth=("price_growth", "mean"),
            avg_quantity_growth=("quantity_growth", "mean"),
            vol_price_growth=("price_growth", "std"),
            vol_quantity_growth=("quantity_growth", "std"),
            avg_qp_ratio=("qp_ratio", "mean"),
        )
        .fillna(0.0)
    )
    features["qp_sign_case"] = [
        qp_sign_case(r.avg_quantity_growth, r.avg_price_growth)
        for r in features.itertuples()
    ]
    return features


def run_ml_pipeline(
    bea_growth: pd.DataFrame,
    bea_industries: pd.DataFrame,
    output_dir: Path | None = None,
    profile: str = "excl_trust_funds",
    period_start: str = "2019-Q1",
    period_end: str = "2026-Q1",
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config()

    features = _industry_features(bea_growth, bea_industries, profile, period_start, period_end)
    features["growth_regime"] = features.apply(lambda r: _label_regime(r, cfg), axis=1)

    x_cols = [
        "avg_price_growth",
        "avg_quantity_growth",
        "vol_price_growth",
        "vol_quantity_growth",
        "avg_qp_ratio",
    ]
    x = features[x_cols].to_numpy()
    n = len(features)
    n_clusters = min(4, max(2, n // 15)) if n >= 4 else max(1, n)

    if n < 2:
        features["cluster_id"] = 0
        features["cluster_label"] = "insufficient_data"
        silhouette = 0.0
        report = {}
        matrix = [[0]]
        confusion_labels = []
        importances = dict.fromkeys(x_cols, 0.0)
    else:
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        features["cluster_id"] = kmeans.fit_predict(x_scaled)
        silhouette = float(silhouette_score(x_scaled, features["cluster_id"])) if n_clusters > 1 else 0.0

        legend = _build_cluster_legend(features, x_cols)
        label_map = dict(zip(legend["cluster_id"], legend["cluster_label"], strict=True))
        features["cluster_label"] = features["cluster_id"].map(label_map)

        y = features["growth_regime"].to_numpy()
        split_kwargs: dict = {"test_size": 0.25, "random_state": 42}
        if len(set(y)) > 1 and min(pd.Series(y).value_counts()) >= 2:
            split_kwargs["stratify"] = y
        x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, **split_kwargs)
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        matrix = confusion_matrix(y_test, y_pred, labels=sorted(features["growth_regime"].unique()))
        confusion_labels = sorted(features["growth_regime"].unique())
        importances = dict(zip(x_cols, clf.feature_importances_.tolist(), strict=True))

    cluster_legend = _build_cluster_legend(features, x_cols) if n >= 2 else pd.DataFrame()

    results = {
        "profile": profile,
        "period": f"{period_start} to {period_end}",
        "n_industries": int(len(features)),
        "n_clusters": int(n_clusters) if n >= 2 else 1,
        "silhouette_score": silhouette,
        "classification_report": report,
        "confusion_matrix": matrix.tolist() if n >= 2 else [],
        "confusion_labels": confusion_labels if n >= 2 else [],
        "feature_importances": importances if n >= 2 else dict.fromkeys(x_cols, 0.0),
        "cluster_counts": features["cluster_id"].value_counts().to_dict(),
        "regime_counts": features["growth_regime"].value_counts().to_dict(),
    }

    suffix = f"_{profile}"
    write_excel_csv(features, output_dir / f"ml_industry_features{suffix}.csv")
    with (output_dir / f"ml_evaluation{suffix}.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    if not cluster_legend.empty:
        write_excel_csv(cluster_legend, output_dir / f"cluster_legend{suffix}.csv")

    labels = features[["line_id", "growth_regime", "cluster_id", "cluster_label"]].copy()
    labels["source"] = "BEA"
    labels["profile"] = profile
    return results, labels, cluster_legend
