import pandas as pd
import argparse
import os
import sys
from inference import threatInference

def main():
    parser = argparse.ArgumentParser(description="Predict attack types for an entire CSV log file")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV log")
    parser.add_argument("--output", type=str, default="threat_analysis_results.csv", help="Path to output CSV")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
        sys.exit(1)

    try:
        # Initialize engine
        engine = threatInference()
        
        # Load data
        print(f"Reading {args.input}...")
        df = pd.read_csv(args.input)
        
        # Run batch prediction
        print("Analyzing logs with LSTM engine...")
        predictions = engine.predict_dataframe(df)
        
        # Append results and save
        df["predicted_attack"] = predictions
        
        print(f"Saving analysis to {args.output}...")
        df.to_csv(args.output, index=False)
        print("\n" + "="*35)
        print("   BATCH ANALYSIS COMPLETE")
        print("="*35)
        print(f"Total Log Entries: {len(df)}")
        if "predicted_attack" in df.columns:
            summary = df["predicted_attack"].value_counts()
            print("\nThreat Summary:")
            for attack, count in summary.items():
                if attack != "Pending Sequence...":
                    print(f" - {attack:<15}: {count}")
        print("="*35 + "\n")

    except Exception as e:
        print(f"Error during batch processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
