import argparse
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Merge predictions from multiple models via weighted soft-voting.")
    parser.add_argument("--small-csv", default="outputs/solution2/small_model/small_probs_dev.csv", help="Predictions from the MiniLM model")
    parser.add_argument("--deberta-original-csv", default="outputs/solution2/big_model/big_probs_dev.csv", help="Predictions from original DeBERTa")
    parser.add_argument("--deberta-7931-csv", default="outputs/solution2/big_model/big_probs_7931.csv", help="Predictions from DeBERTa checkpoint 7931")
    parser.add_argument("--deberta-803-csv", default="outputs/solution2/big_model/big_probs_803.csv", help="Predictions from DeBERTa checkpoint 803")
    parser.add_argument("--deberta-8180-csv", default="outputs/solution2/big_model/big_probs_8180.csv", help="Predictions from our best DeBERTa checkpoint 8180")
    parser.add_argument("--split", default="dev", help="Dataset split (e.g., dev, test)")
    args = parser.parse_args()

    # Load and prep base small model predictions
    print(f"[ensemble] Loading {args.small_csv}...")
    df_minilm = pd.read_csv(args.small_csv)[["pair_id", "prob"]]
    df_minilm.rename(columns={"prob": "prob_minilm"}, inplace=True)

    # Load and prep original DeBERTa predictions
    print(f"[ensemble] Loading {args.deberta_original_csv}...")
    df_deberta_original = pd.read_csv(args.deberta_original_csv)[["pair_id", "prob"]]
    df_deberta_original.rename(columns={"prob": "prob_deberta_original"}, inplace=True)

    # Load and prep DeBERTa 7931 predictions
    print(f"[ensemble] Loading {args.deberta_7931_csv}... (Model 7931)")
    df_deberta_7931 = pd.read_csv(args.deberta_7931_csv)[["pair_id", "prob"]]
    df_deberta_7931.rename(columns={"prob": "prob_deberta_7931"}, inplace=True)

    # Load and prep DeBERTa 803 predictions
    print(f"[ensemble] Loading {args.deberta_803_csv}... (Model 803)")
    df_deberta_803 = pd.read_csv(args.deberta_803_csv)[["pair_id", "prob"]]
    df_deberta_803.rename(columns={"prob": "prob_deberta_803"}, inplace=True)

    # Load and prep DeBERTa 8180 predictions (the highest performing single model)
    print(f"[ensemble] Loading {args.deberta_8180_csv}... (Model 8180)")
    df_deberta_8180 = pd.read_csv(args.deberta_8180_csv)[["pair_id", "prob"]]
    df_deberta_8180.rename(columns={"prob": "prob_deberta_8180"}, inplace=True)

    # Sequentially merge all dataframes on 'pair_id' to align predictions
    df = df_minilm.merge(df_deberta_original, on="pair_id")
    df = df.merge(df_deberta_7931, on="pair_id")
    df = df.merge(df_deberta_803, on="pair_id")
    df = df.merge(df_deberta_8180, on="pair_id")

    if len(df) == 0:
        raise ValueError("Merged dataframe is empty. Ensure pair_ids match across all CSVs.")

    # Weighted Soft Voting:
    # We drop the weaker models (7931 and Original) by setting their weight to 0.00 to avoid dragging down the score.
    # The strongest model (8180) gets 50%, the runner-up (803) gets 40%, and the small MiniLM gets 10% for diversity.
    df["prob_ensemble"] = (
        (df["prob_deberta_8180"] * 0.50) +
        (df["prob_deberta_803"] * 0.40) + 
        (df["prob_deberta_7931"] * 0.00) + 
        (df["prob_deberta_original"] * 0.00) + 
        (df["prob_minilm"] * 0.10)
    )
    
    # Threshold at 0.5 for final binary classification
    df["pred_ensemble"] = (df["prob_ensemble"] > 0.5).astype(int)

    # Export merged probabilities for analysis
    out_dir = "outputs/solution2/ensemble"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"ensemble_probs_{args.split}.csv")
    df.to_csv(out_path, index=False)
    print(f"[ensemble] Saved full ensemble log (5 models) to {out_path}")

    # Export the final predictions required by the local scorer
    scorer_out_path = os.path.join(out_dir, f"ENSEMBLE_AV_{args.split}.csv")
    df[["pred_ensemble"]].to_csv(scorer_out_path, index=False, header=["prediction"])
    print(f"[ensemble] Saved scorer-ready format to {scorer_out_path}")

if __name__ == "__main__":
    main()
