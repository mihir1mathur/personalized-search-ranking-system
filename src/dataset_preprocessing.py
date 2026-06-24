"""
Week 0: Dataset Verification + Beginner Understanding
=====================================================

Project 3 - Personalized Search Ranking System
Dataset: Amazon ESCI Shopping Queries Dataset

WHAT THIS SCRIPT DOES (in plain words):
This is the very first step of the project. Before we build any fancy search
or machine-learning models, we must first *understand* and *trust* our data.

So this script only does "safe" beginner work:
  1. Loads the three dataset files.
  2. Prints basic facts about them (shape, columns, first rows, missing values).
  3. Joins the "examples" table with the "products" table.
  4. Cleans the data and builds one combined text column for each product.
  5. Converts the ESCI relevance labels (E/S/C/I) into numbers (3/2/1/0).
  6. Creates a smaller balanced 50,000-row sample so future weeks run fast.
  7. Saves the sample and a human-readable summary report.

IMPORTANT: Week 0 does NOT build BM25, FAISS, transformers, or train any model.
That work comes in later weeks. Week 0 is only about verifying and understanding.

HOW TO RUN:
    python src/dataset_preprocessing.py
"""

# ---------------------------------------------------------------------------
# Step 0: Imports and helper setup
# ---------------------------------------------------------------------------
# We wrap the pandas import in a try/except so that if the user does not have
# the right libraries installed, we give them a friendly, clear message
# instead of a scary error.
import os
import sys

try:
    import pandas as pd
except ImportError:
    print("ERROR: The 'pandas' library is not installed.")
    print("Pandas is the tool we use to read and work with tables of data.")
    print("Please install it by running this command in your terminal:")
    print("    pip install pandas")
    sys.exit(1)


# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
# We build paths relative to the project root so the script works no matter
# what folder you run it from. __file__ is this script's location
# (.../src/dataset_preprocessing.py), so its parent's parent is the
# project root (.../Project 3).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

EXAMPLES_PATH = os.path.join(DATA_DIR, "shopping_queries_dataset_examples.parquet")
PRODUCTS_PATH = os.path.join(DATA_DIR, "shopping_queries_dataset_products.parquet")
SOURCES_PATH = os.path.join(DATA_DIR, "shopping_queries_dataset_sources.csv")

SAMPLE_OUT_PATH = os.path.join(PROCESSED_DIR, "sample_esci_50k.parquet")
SUMMARY_OUT_PATH = os.path.join(RESULTS_DIR, "week0_dataset_summary.txt")
# A tiny 100-row CSV of example rows (easy to open in Excel / a text editor).
SAMPLE_100_CSV_PATH = os.path.join(RESULTS_DIR, "sample_esci_100.csv")

# We collect summary lines here so we can both print them AND save them to a
# report file at the end.
SUMMARY_LINES = []


def report(line=""):
    """Print a line to the screen AND remember it for the summary report file."""
    print(line)
    SUMMARY_LINES.append(str(line))


def section(title):
    """Print a clearly visible section header (purely for readability)."""
    bar = "=" * 70
    report("")
    report(bar)
    report(title)
    report(bar)


# ---------------------------------------------------------------------------
# Step 1: Loading helpers with friendly error handling
# ---------------------------------------------------------------------------
def load_parquet(path, friendly_name):
    """
    Load a .parquet file safely.

    A .parquet file is just an efficient way to store a big table on disk.
    To read it, pandas needs a helper engine called 'pyarrow' (or 'fastparquet').
    If neither is installed, we explain how to fix that.
    """
    if not os.path.exists(path):
        report(f"ERROR: Could not find the {friendly_name} file at:")
        report(f"    {path}")
        report("Please make sure the dataset files are inside the 'data' folder.")
        sys.exit(1)

    try:
        df = pd.read_parquet(path)
    except ImportError:
        # This happens when neither pyarrow nor fastparquet is installed.
        report("ERROR: Reading parquet files needs an extra engine library.")
        report("Please install one of these (pyarrow is recommended):")
        report("    pip install pyarrow")
        report("    pip install fastparquet")
        sys.exit(1)
    except Exception as e:
        report(f"ERROR: Something went wrong reading {friendly_name}: {e}")
        sys.exit(1)

    return df


def load_csv(path, friendly_name):
    """Load a .csv file safely. A CSV is a simple comma-separated text table."""
    if not os.path.exists(path):
        report(f"ERROR: Could not find the {friendly_name} file at:")
        report(f"    {path}")
        report("Please make sure the dataset files are inside the 'data' folder.")
        sys.exit(1)

    try:
        df = pd.read_csv(path)
    except Exception as e:
        report(f"ERROR: Something went wrong reading {friendly_name}: {e}")
        sys.exit(1)

    return df


def describe_dataset(df, name):
    """Print the beginner facts about one dataset: shape, columns, head, missing."""
    section(f"DATASET: {name}")

    # "Shape" = (number of rows, number of columns).
    report(f"Shape (rows, columns): {df.shape}")

    # Column names tell us what information each table holds.
    report("")
    report("Column names:")
    for col in df.columns:
        report(f"  - {col}")

    # The first 5 rows give us a quick "eyeball" of what the data looks like.
    report("")
    report("First 5 rows:")
    # to_string keeps the table readable in the text report.
    report(df.head(5).to_string())

    # Missing values: empty cells can break later steps, so we count them now.
    report("")
    report("Missing value counts per column:")
    report(df.isnull().sum().to_string())


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------
def main():
    section("WEEK 0: DATASET VERIFICATION + BEGINNER UNDERSTANDING")
    report("Goal: verify, understand, join, sample, and document the ESCI dataset.")
    report("(No BM25, FAISS, transformers, or model training in Week 0.)")

    # Make sure the output folders exist before we try to save anything.
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # -----------------------------------------------------------------------
    # Step 1: Load the three dataset files.
    # -----------------------------------------------------------------------
    report("")
    report("Loading datasets... (the products file is large, please be patient)")
    examples = load_parquet(EXAMPLES_PATH, "examples parquet")
    products = load_parquet(PRODUCTS_PATH, "products parquet")
    sources = load_csv(SOURCES_PATH, "sources csv")

    # -----------------------------------------------------------------------
    # Step 2: Describe each dataset (shape, columns, head, missing values).
    # -----------------------------------------------------------------------
    describe_dataset(examples, "examples (query -> product judgments)")
    describe_dataset(products, "products (product catalog/details)")
    describe_dataset(sources, "sources (where each query came from)")

    # -----------------------------------------------------------------------
    # Step 3: ESCI label distribution.
    # -----------------------------------------------------------------------
    # The ESCI label says how relevant a product is to a query:
    #   E = Exact, S = Substitute, C = Complement, I = Irrelevant.
    section("ESCI LABEL DISTRIBUTION (in the examples table)")
    if "esci_label" in examples.columns:
        label_counts = examples["esci_label"].value_counts(dropna=False)
        report(label_counts.to_string())
    else:
        report("WARNING: 'esci_label' column was not found.")
        report(f"Available columns are: {list(examples.columns)}")

    # -----------------------------------------------------------------------
    # Step 4: Unique queries and unique products.
    # -----------------------------------------------------------------------
    section("UNIQUE COUNTS")
    if "query" in examples.columns:
        report(f"Number of unique queries: {examples['query'].nunique()}")
    else:
        report("WARNING: 'query' column not found in examples.")
        report(f"Available columns are: {list(examples.columns)}")

    if "product_id" in products.columns:
        report(f"Number of unique products (in products table): "
               f"{products['product_id'].nunique()}")
    else:
        report("WARNING: 'product_id' column not found in products.")
        report(f"Available columns are: {list(products.columns)}")

    # -----------------------------------------------------------------------
    # Step 5: Check the join key (product_id) exists in BOTH tables.
    # -----------------------------------------------------------------------
    section("JOIN KEY CHECK")
    has_pid_examples = "product_id" in examples.columns
    has_pid_products = "product_id" in products.columns
    report(f"'product_id' in examples table? {has_pid_examples}")
    report(f"'product_id' in products table? {has_pid_products}")

    if not (has_pid_examples and has_pid_products):
        report("ERROR: Cannot join the tables without 'product_id' in both.")
        report(f"examples columns: {list(examples.columns)}")
        report(f"products columns: {list(products.columns)}")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # A NOTE ON ORDER (memory-friendly pipeline)
    # -----------------------------------------------------------------------
    # The products table is very large (full descriptions for ~1.8M products).
    # Joining ALL examples to ALL product text first would build a giant table
    # in memory and can crash a normal laptop. So we work smart:
    #   1) First do the cheap, lightweight steps on the small "examples" table:
    #      filter to US, convert labels to scores, drop missing, balance-sample.
    #   2) THEN join only the ~50k sampled rows to the heavy product text.
    # The end result is exactly the same cleaned 50k sample, just produced in a
    # way that fits comfortably in memory.
    RANDOM_STATE = 42  # fixed seed so the sample is identical every run

    # -----------------------------------------------------------------------
    # Step 6: Filter examples to US / English rows if a locale column exists.
    # -----------------------------------------------------------------------
    # For Week 0 we focus on English (US) data to keep things simple and to
    # match an English language model later. In this dataset the locale value
    # for US English is the lowercase string 'us'. We do this on the small
    # examples table (no product text yet), so it is fast and light.
    section("FILTERING TO US / ENGLISH (if possible)")
    if "product_locale" in examples.columns:
        before = examples.shape[0]
        us_mask = examples["product_locale"].astype(str).str.lower() == "us"
        if bool(us_mask.any()):
            examples = examples[us_mask].copy()
            report(f"Kept US/English example rows: {examples.shape[0]:,} (was {before:,})")
        else:
            report("No 'us' locale rows found; keeping all rows instead.")
            report(f"Locale values present: "
                   f"{examples['product_locale'].astype(str).str.lower().unique().tolist()}")
    else:
        report("No 'product_locale' column found in examples; skipping locale filter.")

    # -----------------------------------------------------------------------
    # Step 7: Convert ESCI labels into numeric relevance scores.
    # -----------------------------------------------------------------------
    # Models work with numbers, not letters. We use a graded relevance scale:
    #   E (Exact)      = 3  (best match)
    #   S (Substitute) = 2  (a reasonable alternative)
    #   C (Complement) = 1  (related but not what was asked)
    #   I (Irrelevant) = 0  (not relevant)
    # Higher number = more relevant. This lets us rank products later.
    section("CONVERTING ESCI LABELS TO NUMERIC RELEVANCE SCORES")
    esci_to_score = {"E": 3, "S": 2, "C": 1, "I": 0}
    report("Mapping used: E=3 (Exact), S=2 (Substitute), C=1 (Complement), I=0 (Irrelevant)")
    examples["relevance_score"] = examples["esci_label"].map(esci_to_score)

    # Any label that did not map (unexpected value) becomes NaN; drop those.
    before = examples.shape[0]
    examples = examples.dropna(subset=["relevance_score"])
    if examples.shape[0] != before:
        report(f"Dropped {before - examples.shape[0]} rows with unexpected ESCI labels.")
    examples["relevance_score"] = examples["relevance_score"].astype(int)
    report("Score distribution after mapping:")
    report(examples["relevance_score"].value_counts().sort_index().to_string())

    # -----------------------------------------------------------------------
    # Step 8: Drop rows missing the essential fields we have so far.
    # -----------------------------------------------------------------------
    # Before joining, every useful row needs a query, a product_id, and a label.
    # (product_text is added after the join and is checked again in Step 11.)
    section("REMOVING ROWS WITH MISSING ESSENTIAL FIELDS (pre-join)")
    pre_join_essential = [c for c in ["query", "product_id", "esci_label"]
                          if c in examples.columns]
    report(f"Checking these essential columns for missing values: {pre_join_essential}")
    before = examples.shape[0]
    examples = examples.dropna(subset=pre_join_essential)
    report(f"Rows after removing missing essentials: {examples.shape[0]:,} (was {before:,})")

    # -----------------------------------------------------------------------
    # Step 9: Create a balanced sample of up to 50,000 rows.
    # -----------------------------------------------------------------------
    # WHY SAMPLE? The full dataset is huge. A smaller sample lets us develop and
    # test future code quickly on a normal laptop.
    # WHY BALANCED? If one label dominates, models can get lazy and just guess
    # the common label. Taking a roughly equal number of each label gives every
    # relevance level fair representation.
    # We sample on the lightweight examples table FIRST, then attach product
    # text to only the chosen rows in Step 10.
    section("CREATING A BALANCED SAMPLE (UP TO 50,000 ROWS)")
    TARGET_TOTAL = 50_000
    labels_present = sorted(examples["relevance_score"].unique())
    n_groups = len(labels_present)
    report(f"Relevance levels present: {labels_present}")

    # How many rows we would *like* per label to reach the target evenly.
    per_label_target = TARGET_TOTAL // max(n_groups, 1)
    report(f"Aiming for about {per_label_target:,} rows per relevance level.")

    sampled_parts = []
    for score in labels_present:
        group = examples[examples["relevance_score"] == score]
        # If a group is smaller than the target, take all of it (no replacement).
        take = min(per_label_target, len(group))
        sampled_parts.append(group.sample(n=take, random_state=RANDOM_STATE))
        report(f"  Relevance {score}: available {len(group):,}, taking {take:,}")

    sample = pd.concat(sampled_parts, ignore_index=True)

    # If some labels were too small to hit the target, we may be under 50k.
    # Top up with extra random rows from the leftovers to get closer to 50k.
    if len(sample) < TARGET_TOTAL and len(examples) > len(sample):
        already = set(sample["example_id"]) if "example_id" in sample.columns else None
        if already is not None:
            leftovers = examples[~examples["example_id"].isin(already)]
        else:
            leftovers = examples
        need = min(TARGET_TOTAL - len(sample), len(leftovers))
        if need > 0:
            topup = leftovers.sample(n=need, random_state=RANDOM_STATE)
            sample = pd.concat([sample, topup], ignore_index=True)
            report(f"Topped up with {need:,} extra rows to get closer to {TARGET_TOTAL:,}.")

    # Shuffle so the labels are mixed together rather than grouped.
    sample = sample.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    report(f"Sampled example rows (before join): {len(sample):,}")

    # -----------------------------------------------------------------------
    # Step 10: Join the sampled examples with products.
    # -----------------------------------------------------------------------
    # WHY JOIN? The examples table tells us which (query, product) pairs were
    # judged and how relevant they are -- but it does NOT contain the product
    # text (title, description, etc.). The products table holds that text.
    # Joining glues the two together so each judged pair also has product text.
    #
    # The ESCI dataset is multi-language, and the same product_id can exist in
    # different locales. So the *correct* join key is BOTH 'product_locale' and
    # 'product_id' when both tables have a locale column. Otherwise we fall back
    # to joining on product_id only. We join only the ~50k sampled rows, so this
    # stays small and memory-friendly.
    section("JOINING SAMPLED EXAMPLES WITH PRODUCTS")
    join_keys = ["product_id"]
    if "product_locale" in sample.columns and "product_locale" in products.columns:
        join_keys = ["product_locale", "product_id"]
    report(f"Joining on key(s): {join_keys}")

    sample = sample.merge(products, on=join_keys, how="inner")
    report(f"Rows after join: {sample.shape[0]:,}")
    report(f"Columns after join: {list(sample.columns)}")

    # -----------------------------------------------------------------------
    # Step 11: Build the combined 'product_text' column.
    # -----------------------------------------------------------------------
    # WHY? A search/ranking model reads product *text*. Instead of juggling 5
    # separate columns, we combine them into one rich text field per product.
    # We only use the columns that actually exist in the data.
    section("BUILDING THE 'product_text' COLUMN")
    text_source_cols = [
        "product_title",
        "product_description",
        "product_bullet_point",
        "product_brand",
        "product_color",
    ]
    existing_text_cols = [c for c in text_source_cols if c in sample.columns]
    report(f"Columns combined into product_text: {existing_text_cols}")
    missing_text_cols = [c for c in text_source_cols if c not in sample.columns]
    if missing_text_cols:
        report(f"(These were requested but not present, so skipped: {missing_text_cols})")

    if not existing_text_cols:
        report("ERROR: None of the expected product text columns were found.")
        report(f"Available columns are: {list(sample.columns)}")
        sys.exit(1)

    # For each chosen column, turn missing values into empty strings, then glue
    # the columns together with a single space between them.
    parts = [sample[c].fillna("").astype(str) for c in existing_text_cols]
    product_text = parts[0]
    for p in parts[1:]:
        product_text = product_text.str.cat(p, sep=" ")
    # Collapse multiple spaces into one and trim the ends for tidy text.
    sample["product_text"] = (
        product_text.str.replace(r"\s+", " ", regex=True).str.strip()
    )
    report("Created 'product_text'. Example (first row, trimmed to 200 chars):")
    if sample.shape[0] > 0:
        report("  " + sample["product_text"].iloc[0][:200])

    # -----------------------------------------------------------------------
    # Step 12: Final check -- drop any rows now missing product_text.
    # -----------------------------------------------------------------------
    # A row is only useful if it has a query, product_id, esci_label, AND some
    # product text. We re-check the full essential list now that text exists.
    section("REMOVING ROWS WITH MISSING product_text")
    essential = ["query", "product_id", "esci_label", "product_text"]
    essential_present = [c for c in essential if c in sample.columns]
    before = sample.shape[0]
    # Treat empty product_text as missing too (a blank string is not useful).
    sample = sample[sample["product_text"].str.len() > 0]
    sample = sample.dropna(subset=essential_present)
    report(f"Final rows after removing missing product_text: "
           f"{sample.shape[0]:,} (was {before:,})")

    report(f"Final sample size: {len(sample):,} rows")
    report("Final sample relevance distribution:")
    report(sample["relevance_score"].value_counts().sort_index().to_string())

    # -----------------------------------------------------------------------
    # Step 13: Query length analysis.
    # -----------------------------------------------------------------------
    # WHAT IS QUERY LENGTH? It is simply how many WORDS a shopper typed in their
    # search. For example, "shoes" has length 1, while "red running shoes for
    # men" has length 5. We count words by splitting the query text on spaces.
    #
    # WHY DO SEARCH SYSTEMS ANALYZE QUERY LENGTH?
    #   - Short queries (1-2 words) are broad and ambiguous ("shoes"), so the
    #     system must guess intent and often return many possible matches.
    #   - Medium queries (3-5 words) are more specific ("red running shoes").
    #   - Long queries (6+ words) are very specific and often describe an exact
    #     need, which changes how a ranking model should weigh keywords.
    # Knowing this distribution helps us design and tune retrieval methods
    # (BM25, embeddings, etc.) for the kinds of queries users actually type.
    section("QUERY LENGTH ANALYSIS")

    # str.split() breaks the text on whitespace; len() counts the resulting words.
    sample["query_length"] = sample["query"].astype(str).str.split().apply(len)

    avg_len = float(sample["query_length"].mean())
    min_len = int(sample["query_length"].min())
    max_len = int(sample["query_length"].max())
    report(f"Average query length (words): {avg_len:.2f}")
    report(f"Minimum query length (words): {min_len}")
    report(f"Maximum query length (words): {max_len}")

    # Percentage of queries in each length "bucket". We divide each count by the
    # total number of rows and multiply by 100 to get a percentage.
    total_rows = len(sample)
    pct_1_2 = (sample["query_length"].between(1, 2).sum() / total_rows) * 100
    pct_3_5 = (sample["query_length"].between(3, 5).sum() / total_rows) * 100
    pct_6_plus = ((sample["query_length"] >= 6).sum() / total_rows) * 100
    report("")
    report("Query length distribution:")
    report(f"  1-2 words : {pct_1_2:.2f}%")
    report(f"  3-5 words : {pct_3_5:.2f}%")
    report(f"  6+  words : {pct_6_plus:.2f}%")

    # -----------------------------------------------------------------------
    # Step 14: Save the cleaned sample to disk.
    # -----------------------------------------------------------------------
    section("SAVING OUTPUTS")
    try:
        sample.to_parquet(SAMPLE_OUT_PATH, index=False)
        report(f"Saved cleaned sample to: {SAMPLE_OUT_PATH}")
    except ImportError:
        report("ERROR: Saving parquet needs the 'pyarrow' (or 'fastparquet') engine.")
        report("Install it with:  pip install pyarrow")
        sys.exit(1)
    except Exception as e:
        report(f"ERROR: Could not save the sample parquet: {e}")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 15: Save a tiny 100-row CSV of example rows.
    # -----------------------------------------------------------------------
    # WHY? A small, easy-to-open CSV (just the first 100 rows) is handy for:
    #   - debugging BM25 / retrieval code on a few real rows,
    #   - showing concrete retrieval examples,
    #   - explaining the data in interviews,
    #   - demonstrating the project to recruiters.
    # It is NOT used for training -- it is purely a convenient human-readable peek.
    try:
        sample.head(100).to_csv(SAMPLE_100_CSV_PATH, index=False, encoding="utf-8")
        report(f"Saved 100-row example CSV to: {SAMPLE_100_CSV_PATH}")
    except Exception as e:
        report(f"ERROR: Could not save the 100-row CSV: {e}")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 16: Save the summary report.
    # -----------------------------------------------------------------------
    # Note: we add this final line to the report contents before writing.
    SUMMARY_LINES.append("")
    SUMMARY_LINES.append("Week 0 Dataset Verification completed successfully.")
    try:
        with open(SUMMARY_OUT_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(SUMMARY_LINES))
        report(f"Saved summary report to: {SUMMARY_OUT_PATH}")
    except Exception as e:
        report(f"ERROR: Could not save the summary report: {e}")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 17: Final success message.
    # -----------------------------------------------------------------------
    print("")
    print("Week 0 Dataset Verification completed successfully.")


if __name__ == "__main__":
    main()
