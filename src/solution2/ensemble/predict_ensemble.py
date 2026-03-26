import argparse
import os
import pandas as pd

def main() -> None:
    parser = argparse.ArgumentParser(description="Merge predictions from multiple models via weighted soft-voting.")
    # Core Models
    parser.add_argument("--small-csv", default="outputs/solution2/small_model/small_probs_dev.csv", help="Predictions from the MiniLM model")
    parser.add_argument("--deberta-8180-csv", default="outputs/solution2/big_model/big_probs_8180.csv", help="Best DeBERTa checkpoint 8180")
    parser.add_argument("--deberta-803-csv", default="outputs/solution2/big_model/big_probs_803.csv", help="DeBERTa checkpoint 803")
    
    # New Vanilla Models
    parser.add_argument("--roberta-csv", default="outputs/roberta_probs_dev.csv", help="RoBERTa vanilla probabilities")
    parser.add_argument("--xlnet-csv", default="outputs/xlnetvanilla_probs_dev.csv", help="XLNet vanilla probabilities")
    parser.add_argument("--electra-csv", default="outputs/electra_probs_dev.csv", help="ELECTRA vanilla probabilities")
    
    parser.add_argument("--split", default="dev", help="Dataset split (e.g., dev, test)")
    args = parser.parse_args()

    # 1. Load Core Models
    print(f"[ensemble] Loading Core Models...")
    df_8180 = pd.read_csv(args.deberta_8180_csv)[["pair_id", "prob"]].rename(columns={"prob": "p_8180"})
    df_803 = pd.read_csv(args.deberta_803_csv)[["prob"]].rename(columns={"prob": "p_803"})
    df_small = pd.read_csv(args.small_csv)[["prob"]].rename(columns={"prob": "p_small"})

    # 2. Load New Vanilla Models (assuming row alignment as they lack pair_id)
    print(f"[ensemble] Loading New Vanilla Models...")
    df_roberta = pd.read_csv(args.roberta_csv)["probability"].to_frame("p_roberta")
    df_xlnet = pd.read_csv(args.xlnet_csv)["probability"].to_frame("p_xlnet")
    df_electra = pd.read_csv(args.electra_csv)["probability"].to_frame("p_electra")

    # 3. Concatenate all predictions
    # We use df_8180 as the anchor because it contains the pair_ids
    df = pd.concat([df_8180, df_803, df_small, df_roberta, df_xlnet, df_electra], axis=1)

    if df.isnull().values.any():
        print("[WARNING] Missing values detected in merged ensemble. Check row alignments.")

    # 4. Weighted Soft Voting (Exhaustive Optimization: 0.8615 Macro F1)
    # Weights: 8180(21.6%), 803(5.8%), Small(12.6%), RoBERTa(30.5%), XLNet(13.9%), ELECTRA(15.6%)
    df["prob_ensemble"] = (
        (df["p_8180"] * 0.2161) +
        (df["p_803"] * 0.0583) + 
        (df["p_small"] * 0.1262) + 
        (df["p_roberta"] * 0.3047) + 
        (df["p_xlnet"] * 0.1387) + 
        (df["p_electra"] * 0.1561)
    )
    
    # 5. Threshold at 0.5 for final binary classification
    df["pred_ensemble"] = (df["prob_ensemble"] > 0.5).astype(int)

    # Export merged probabilities for analysis
    out_dir = "outputs/solution2/ensemble"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"ensemble_probs_expanded_{args.split}.csv")
    df.to_csv(out_path, index=False)
    print(f"[ensemble] Saved expanded ensemble log (6 models) to {out_path}")

    # Export the final predictions required by the local scorer
    scorer_out_path = os.path.join(out_dir, f"ENSEMBLE_EXPANDED_AV_{args.split}.csv")
    df[["pred_ensemble"]].to_csv(scorer_out_path, index=False, header=["prediction"])
    print(f"[ensemble] Saved scorer-ready format to {scorer_out_path}")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
