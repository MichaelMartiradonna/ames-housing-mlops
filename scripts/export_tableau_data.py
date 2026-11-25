"""Export real Ames sales data and held-out model predictions for Tableau Public."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data import load_dataset, split_dataset
from src.evaluate import regression_metrics
from src.interface import CATEGORY_LABELS
from src.serving import load_serving_model


def quality_tier(score: int) -> str:
    if score <= 4:
        return "1-4 Below Average"
    if score <= 6:
        return "5-6 Average"
    if score <= 8:
        return "7-8 Above Average"
    return "9-10 Excellent"


def main() -> int:
    config = load_config()
    frame = load_dataset(config, verify_hash=True)
    splits = split_dataset(frame, config)
    model, _ = load_serving_model()

    test_preds = np.asarray(model.predict(splits.X_test), dtype=float)
    train_median = float(np.median(splits.y_train))
    baseline_preds = np.full(len(splits.y_test), train_median, dtype=float)

    model_metrics = regression_metrics(splits.y_test, test_preds)
    baseline_metrics = regression_metrics(splits.y_test, baseline_preds)
    reduction_pct = (1.0 - model_metrics["mae"] / baseline_metrics["mae"]) * 100.0

    # Verify exact parity with the saved evaluation report before writing outputs.
    assert len(frame) == 2930, f"Expected 2930 rows, got {len(frame)}"
    assert len(splits.y_test) == 586, f"Expected 586 holdout rows, got {len(splits.y_test)}"
    assert round(model_metrics["mae"]) == 17765, f"Unexpected MAE: {model_metrics['mae']}"
    assert round(baseline_metrics["mae"]) == 63823, f"Unexpected baseline MAE: {baseline_metrics['mae']}"

    split_labels = pd.Series("Train", index=frame.index, dtype="object")
    split_labels.loc[splits.X_validation.index] = "Validation"
    split_labels.loc[splits.X_test.index] = "Held-out Test"

    pred_series = pd.Series(np.nan, index=frame.index, dtype="float64")
    pred_series.loc[splits.X_test.index] = test_preds

    baseline_series = pd.Series(np.nan, index=frame.index, dtype="float64")
    baseline_series.loc[splits.X_test.index] = train_median

    actual = frame["SalePrice"].astype(float)
    signed_error = pred_series - actual
    abs_error = signed_error.abs()
    pct_abs_error = abs_error / actual
    baseline_abs_error = (baseline_series - actual).abs()

    out = pd.DataFrame({
        "PID": frame["PID"],
        "Split": split_labels,
        "Neighborhood_Code": frame["Neighborhood"],
        "Neighborhood": frame["Neighborhood"].map(lambda c: CATEGORY_LABELS.get(c, c)),
        "Year_Built": frame["Year Built"].astype(int),
        "Decade_Built": ((frame["Year Built"].astype(int) // 10) * 10).astype(str) + "s",
        "Year_Sold": frame["Yr Sold"].astype(int),
        "Overall_Qual": frame["Overall Qual"].astype(int),
        "Quality_Tier": frame["Overall Qual"].astype(int).map(quality_tier),
        "Gr_Liv_Area": frame["Gr Liv Area"].astype(int),
        "Total_Bsmt_SF": frame["Total Bsmt SF"].fillna(0).astype(int),
        "Lot_Area": frame["Lot Area"].astype(int),
        "Full_Bath": frame["Full Bath"].astype(int),
        "Bedroom_AbvGr": frame["Bedroom AbvGr"].astype(int),
        "Garage_Cars": frame["Garage Cars"].fillna(0).astype(int),
        "House_Style": frame["House Style"].map(lambda c: CATEGORY_LABELS.get(c, c)),
        "Building_Type": frame["Bldg Type"].map(lambda c: CATEGORY_LABELS.get(c, c)),
        "Central_Air": frame["Central Air"].map(lambda c: CATEGORY_LABELS.get(c, c)),
        "Actual_Sale_Price": actual.round(0).astype(int),
        "Price_Per_SqFt": (actual / frame["Gr Liv Area"].astype(float)).round(2),
        "Predicted_Sale_Price": pred_series.round(2),
        "Baseline_Median_Price": baseline_series.round(2),
        "Signed_Error": signed_error.round(2),
        "Abs_Error": abs_error.round(2),
        "Pct_Abs_Error": pct_abs_error.round(4),
        "Baseline_Abs_Error": baseline_abs_error.round(2),
    })

    bi_dir = ROOT / "bi"
    bi_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = bi_dir / "ames_tableau_dataset.csv"
    out.to_csv(dataset_path, index=False)

    # Neighborhood-level summary for quick reference and verification
    all_agg = (
        out.groupby(["Neighborhood_Code", "Neighborhood"], as_index=False)
        .agg(
            Total_Sales=("PID", "count"),
            Median_Sale_Price=("Actual_Sale_Price", "median"),
            Mean_Sale_Price=("Actual_Sale_Price", "mean"),
            Median_Year_Built=("Year_Built", "median"),
            Mean_Overall_Qual=("Overall_Qual", "mean"),
        )
    )
    holdout = out[out["Split"] == "Held-out Test"]
    test_agg = (
        holdout.groupby("Neighborhood_Code", as_index=False)
        .agg(
            Holdout_Homes=("PID", "count"),
            Holdout_MAE=("Abs_Error", "mean"),
            Baseline_MAE=("Baseline_Abs_Error", "mean"),
            Mean_Signed_Error=("Signed_Error", "mean"),
            Median_Pct_Abs_Error=("Pct_Abs_Error", "median"),
        )
    )
    summary = all_agg.merge(test_agg, on="Neighborhood_Code", how="left")
    summary["Median_Sale_Price"] = summary["Median_Sale_Price"].round(0).astype(int)
    summary["Mean_Sale_Price"] = summary["Mean_Sale_Price"].round(0).astype(int)
    summary["Median_Year_Built"] = summary["Median_Year_Built"].round(0).astype(int)
    summary["Mean_Overall_Qual"] = summary["Mean_Overall_Qual"].round(2)
    summary["Holdout_Homes"] = summary["Holdout_Homes"].fillna(0).astype(int)
    summary["Holdout_MAE"] = summary["Holdout_MAE"].round(0)
    summary["Baseline_MAE"] = summary["Baseline_MAE"].round(0)
    summary["Mean_Signed_Error"] = summary["Mean_Signed_Error"].round(0)
    summary["Median_Pct_Abs_Error"] = (summary["Median_Pct_Abs_Error"] * 100.0).round(1)
    summary = summary.sort_values("Median_Sale_Price", ascending=False)

    summary_path = bi_dir / "ames_neighborhood_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Exported {len(out)} rows to {dataset_path}")
    print(f"Exported {len(summary)} neighborhoods to {summary_path}")
    print(f"Verified Holdout (n={len(holdout)}): Model MAE=${model_metrics['mae']:,.2f} | "
          f"Baseline MAE=${baseline_metrics['mae']:,.2f} ({reduction_pct:.1f}% lower) | R2={model_metrics['r2']:.3f}")
    print("\nTop 5 neighborhoods by median sale price:")
    print(summary[["Neighborhood", "Total_Sales", "Median_Sale_Price", "Holdout_Homes", "Holdout_MAE"]].head(5).to_string(index=False))
    print("\nTop 5 neighborhoods by holdout MAE (min 5 holdout homes):")
    high_err = summary[summary["Holdout_Homes"] >= 5].sort_values("Holdout_MAE", ascending=False)
    print(high_err[["Neighborhood", "Holdout_Homes", "Median_Sale_Price", "Holdout_MAE", "Baseline_MAE"]].head(5).to_string(index=False))
    print("\nLowest 5 neighborhoods by holdout MAE (min 5 holdout homes):")
    low_err = summary[summary["Holdout_Homes"] >= 5].sort_values("Holdout_MAE", ascending=True)
    print(low_err[["Neighborhood", "Holdout_Homes", "Median_Sale_Price", "Holdout_MAE", "Baseline_MAE"]].head(5).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
