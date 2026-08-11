import os
import pandas as pd
from datasets import load_dataset

def fetch_and_translate_supply_chain_data(num_rows=100, output_path="data/supply_chain_text.txt"):
    print(f"[INFO] Fetching {num_rows} rows from SupplyGraph dataset...")
    try:
        # Load the dataset from Hugging Face
        dataset = load_dataset("miminmoons/olist-ecommerce-for-delivery-and-review-prediction", split="train")
        df = dataset.to_pandas().head(num_rows)
        
        paragraphs = []
        for index, row in df.iterrows():
            # Translate relational rows into plain English paragraphs
            col_strings = [f"{col} is {val}" for col, val in row.items() if pd.notnull(val)]
            paragraph = " ".join(col_strings) + "."
            paragraphs.append(paragraph)
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(paragraphs))
            
        print(f"[INFO] Successfully translated and saved {num_rows} paragraphs to {output_path}")
        return paragraphs
        
    except Exception as e:
        print(f"[ERROR] Failed to fetch or translate dataset: {str(e)}")
        return []

if __name__ == "__main__":
    fetch_and_translate_supply_chain_data(num_rows=100)
