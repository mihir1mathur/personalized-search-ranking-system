# Personalized Search Ranking System

A search and ranking system built on real e-commerce search data. The project
implements and evaluates information retrieval methods that take a shopper's
query and rank candidate products by relevance, starting from classic lexical
baselines and building toward modern semantic retrieval.

## Project Overview

Product search is the problem of returning the most relevant items from a large
catalog in response to a free-text query such as `wireless headphones` or
`blue yoga mat`. The core challenge is **ranking**: given many candidate
products, decide the order in which they should appear so that the items a user
most likely wants surface at the top.

This project sits at the intersection of three ideas:

- **Information retrieval** — efficiently finding relevant documents (products)
  from a large collection for a given query.
- **Ranking systems** — ordering those results so that more relevant items
  appear first, using graded relevance rather than a simple match/no-match.
- **Search relevance** — measuring *how well* a result actually satisfies the
  intent behind a query, and quantifying that quality with standard metrics.

The work is grounded in the **Amazon ESCI Shopping Queries Dataset**, a large
public benchmark of real shopping queries paired with products and human
relevance judgments.

## Dataset

The project uses the **Amazon ESCI Shopping Queries Dataset**. Key figures:

| Item | Count |
| --- | --- |
| Examples (query–product judgments) | ~2.6 million |
| Products | ~1.8 million |
| Unique queries | ~130,000 |

Each example pairs a query with a product and assigns one of four **ESCI
relevance labels**:

- **E — Exact**: the product is exactly what the query asked for.
  (`red running shoes` → a pair of red running shoes)
- **S — Substitute**: not exact, but a reasonable alternative that could still
  satisfy the shopper. (`red running shoes` → blue running shoes)
- **C — Complement**: a related item that goes *with* the searched item but is
  not the item itself. (`red running shoes` → running socks)
- **I — Irrelevant**: not related to the query at all.
  (`red running shoes` → a kitchen blender)

For modeling, these labels are mapped to a graded numeric relevance scale where
a higher number means a better match: **E = 3, S = 2, C = 1, I = 0**. This
ordering lets the system learn to place higher-scoring products at the top of
the results and enables ranking metrics like NDCG.

## Project Goals

- Build **lexical retrieval baselines** (TF-IDF and BM25).
- **Evaluate search quality** with standard ranking metrics.
- **Compare retrieval methods** fairly on the same data and metrics.
- **Study search relevance** and where keyword matching breaks down.
- Lay the groundwork to **develop semantic search systems** that go beyond
  exact keyword overlap.

## Project Pipeline

```
Dataset
    ↓
Preprocessing
    ↓
TF-IDF
    ↓
BM25
    ↓
Evaluation
    ↓
Failure Analysis
    ↓
Dense Retrieval (future)
    ↓
Re-ranking (future)
```

## Repository Structure

```
.
├── data/
│   ├── shopping_queries_dataset_examples.parquet
│   ├── shopping_queries_dataset_products.parquet
│   ├── shopping_queries_dataset_sources.csv
│   └── processed/
│       └── sample_esci_50k.parquet
├── src/
│   ├── dataset_preprocessing.py     # Load, clean, balance, and sample the dataset
│   ├── tfidf_retriever.py           # TF-IDF lexical retrieval baseline
│   ├── bm25_retriever.py            # BM25 lexical retrieval baseline
│   └── evaluate_retrieval.py        # Run retrieval and compute ranking metrics
├── evaluation/
│   └── metrics.py                   # Precision@K, Recall@K, NDCG@K
├── results/
│   ├── week0_dataset_summary.txt
│   ├── sample_esci_100.csv
│   ├── sample_query_analysis.txt
│   ├── query_length_distribution.png
│   ├── retrieval_results.csv
│   ├── week1_metrics.txt
│   ├── week1_failure_cases.txt
│   └── tfidf_vs_bm25_metrics.png
├── docs/
│   ├── Week0_Data_Understanding.md
│   └── Week1_Retrieval_Baselines.md
├── notes/
│   ├── Week0.txt
│   └── Week1.txt
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

## Week 0: Dataset Preprocessing

The first stage verifies and prepares the data before any modeling:

- **Filtering**: focus on the US / English subset of the dataset to keep the
  text consistent with an English-language retrieval setup.
- **Balancing**: build a sample of up to 50,000 rows with roughly equal numbers
  of each relevance level (3/2/1/0), so no single label dominates.
- **Preprocessing**: join the examples and products tables on product ID, map
  the ESCI labels to numeric scores, drop rows with missing essential fields,
  and combine the product title, description, bullet points, brand, and color
  into a single `product_text` field.
- **Query analysis**: measure query length (in words) and its distribution to
  understand how broad or specific user queries tend to be.

## Week 1: Retrieval Baselines

This stage builds two classic lexical retrieval baselines and evaluates them
with standard ranking metrics computed at a cutoff of 10 results.

- **TF-IDF**: ranks products by term-frequency / inverse-document-frequency
  similarity between the query and `product_text`.
- **BM25**: a probabilistic keyword-ranking function that improves on TF-IDF
  with term-frequency saturation and document-length normalization.
- **Precision@10**: of the top 10 retrieved products, the fraction that are
  relevant.
- **Recall@10**: of all relevant products, the fraction found in the top 10.
- **NDCG@10**: a graded ranking metric that rewards placing more relevant
  products higher in the list.

### Results

**TF-IDF**

| Metric | Score |
| --- | --- |
| Precision@10 | 0.0442 |
| Recall@10 | 0.3208 |
| NDCG@10 | 0.1940 |

**BM25**

| Metric | Score |
| --- | --- |
| Precision@10 | 0.0657 |
| Recall@10 | 0.4737 |
| NDCG@10 | 0.3434 |

BM25 outperforms TF-IDF on every metric, which is consistent with its stronger
handling of term frequency and document length.

## Failure Analysis

Even the stronger lexical baseline misses relevant products when the query and
the product text use different words for the same concept. Representative cases:

- **`pjs` vs `pajamas`** — an abbreviation the user types that never appears in
  the product text.
- **`charger cord` vs `cable`** — different vocabulary for the same kind of
  accessory.
- **`TV` vs `television`** — an acronym versus its full written form.

Keyword retrieval struggles here because TF-IDF and BM25 match on **exact token
overlap**. They have no notion that two different words can mean the same thing,
so synonyms, abbreviations, and acronyms cause relevant products to be scored as
unrelated. This vocabulary mismatch is precisely the gap that semantic retrieval
is designed to close.

## Future Work

The project is intentionally developed incrementally so that each retrieval stage can be evaluated against previously established baselines.

The following stages are planned to move beyond lexical matching toward semantic
understanding of queries and products:

- **Sentence Transformers** — encode queries and products into dense vector
  representations that capture meaning rather than surface words.
- **Dense Retrieval** — retrieve by vector similarity so semantically related
  items match even without shared keywords.
- **FAISS** — index dense vectors for fast approximate nearest-neighbor search
  over the full catalog.
- **Hybrid Retrieval** — combine lexical (BM25) and dense signals to get the
  precision of keywords and the recall of semantics.
- **Cross-Encoder Re-ranking** — re-score the top candidates with a model that
  reads the query and product jointly for a final, high-quality ordering.

## Installation

```bash
pip install -r requirements.txt
```

## Running the Project

Preprocess the dataset (Week 0):

```bash
python src/dataset_preprocessing.py
```

Run retrieval and evaluation (Week 1):

```bash
python src/evaluate_retrieval.py
```

## Additional Documentation

- **[Week 0 — Data Understanding](docs/Week0_Data_Understanding.md):** dataset
  understanding and preprocessing.
- **[Week 1 — Retrieval Baselines](docs/Week1_Retrieval_Baselines.md):**
  retrieval baselines and evaluation.

## Notes

The repository includes detailed learning notes (in [`notes/`](notes/)) that
explain the concepts, intuition, and evaluation process used during development.

## License

This project is released under the [MIT License](LICENSE).
