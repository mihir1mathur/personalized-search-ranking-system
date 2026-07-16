# Data

This folder holds the datasets used by the project. The raw and processed data
files are **not committed** to the repository (they are large and are ignored in
`.gitignore`); only this README is tracked so the folder structure is preserved.

## Expected contents (local only)

- `shopping_queries_dataset_examples.parquet` — raw ESCI query–product examples
- `shopping_queries_dataset_products.parquet` — raw ESCI product catalog
- `shopping_queries_dataset_sources.csv` — raw ESCI source metadata
- `processed/sample_esci_50k.parquet` — the cleaned, balanced 50k sample used
  throughout Weeks 0–5

## How to obtain the data

The project is built on the public **Amazon ESCI Shopping Queries Dataset**
(<https://github.com/amazon-science/esci-data>). Download the raw files into this
folder, then run the Week 0 preprocessing step to create the processed sample:

```bash
python src/dataset_preprocessing.py
```
