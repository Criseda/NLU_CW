import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_confusion_matrix(cm_dict, output_path):
    """Plot a nice confusion matrix heatmap."""
    # cm_dict corresponds to {"tp": ..., "fp": ..., "fn": ..., "tn": ...}
    cm = [[cm_dict["tn"], cm_dict["fp"]], 
          [cm_dict["fn"], cm_dict["tp"]]]
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred Neg (0)', 'Pred Pos (1)'],
                yticklabels=['Actual Neg (0)', 'Actual Pos (1)'])
    plt.title('Confusion Matrix on Dev Set')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved Confusion Matrix to {output_path}")

def plot_calibration_curve(calib_dict, output_path):
    """Plot the calibration curve."""
    fraction_pos = calib_dict["fraction_positive"]
    mean_pred = calib_dict["mean_predicted"]
    
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.plot(mean_pred, fraction_pos, "s-", label=f"Stacked Model (ECE={calib_dict['ece']:.4f})")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve (Reliability Diagram)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved Calibration Curve to {output_path}")

def print_metrics_table(results):
    """Print a clean Markdown-style table of the results for the poster."""
    print("\n" + "="*50)
    print("📊 FINAL DEV SET EVALUATION RESULTS")
    print("="*50)
    
    df_metrics = pd.DataFrame(
        [
            {"Metric": k, "Score": v, "95% CI Limit (Lower)": results["bootstrap_ci"][k]["lower"], "95% CI Limit (Upper)": results["bootstrap_ci"][k]["upper"]}
            for k, v in results["metrics"].items()
        ]
    )
    print("\n--- Core Metrics ---")
    print(df_metrics.to_markdown(index=False))
    
    print("\n--- Base Model Comparison ---")
    base_df = pd.DataFrame.from_dict(results["base_models"], orient='index')
    print(base_df.to_markdown())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", default="src/solution1/results/eval_results.json", help="Path to evaluation JSON")
    parser.add_argument("--output_dir", default="src/solution1/results/plots", help="Directory to save plots")
    args = parser.parse_args()
    
    json_file = Path(args.json_path)
    if not json_file.exists():
        print(f"❌ Error: Could not find {json_file}")
        return
        
    with open(json_file, "r") as f:
        results = json.loads(f.read())
        
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualisations
    plot_confusion_matrix(results["confusion_matrix"], out_dir / "confusion_matrix.png")
    plot_calibration_curve(results["calibration"], out_dir / "calibration_curve.png")
    
    # Print formatted text table
    print_metrics_table(results)
    
if __name__ == "__main__":
    main()
