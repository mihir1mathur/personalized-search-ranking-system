# Week 0 — Dataset Understanding & Preprocessing

**Project:** Personalized Search Ranking System
**Dataset:** Amazon ESCI Shopping Queries Dataset

This document explains the goals, decisions, and observations of the data
understanding and preprocessing stage. It assumes no prior machine-learning
background.

---

## 1. What is this project trying to build?

We are building a **search ranking system**. When a shopper types a query such
as `wireless headphones` into a search box, the site returns a list of products.
The central question is:

> In what **order** should those products appear?

A good system places the products the shopper most likely wants at the **top**
and unrelated items at the bottom. That ordering job is called **ranking**, and
building a system that ranks products well for a given search is the goal of the
overall project.

Week 0 is the warm-up: we do **not** build the ranking system yet. We first make
sure we fully understand and trust the data.

---

## 2. What each dataset file means

The `data/` folder contains three files:

- **`shopping_queries_dataset_examples.parquet`** — the "answer key" of the
  dataset. Each row is a *(search query, product)* pair plus a label saying how
  relevant that product is to that query. In other words: *for this search, here
  is one product, and here is how good a match it is.*
- **`shopping_queries_dataset_products.parquet`** — the "product catalog." Each
  row is one product with its details: title, description, bullet points, brand,
  and color. The examples file only stores a product's ID, not its text; the
  product text lives here.
- **`shopping_queries_dataset_sources.csv`** — a small extra file describing
  where each query originally came from (for example, behavioral data or
  negations). It is mostly background information for this stage.

---

## 3. What does "query" mean?

A **query** is simply what a shopper types into the search box — for example
`running shoes`, `iphone charger`, or `blue yoga mat`. It is the question the
shopper is asking the search engine.

---

## 4. What does `product_id` mean?

Every product has a unique code that identifies it, called `product_id` (similar
to a barcode). The examples table uses this code to point at a product without
repeating all its text, and the products table uses the same code to store the
full details. Because both tables share this code, we can **join** them: match
each example to the correct product using `product_id`.

---

## 5. What do the ESCI labels mean?

ESCI is four letters, each describing how relevant a product is to a search
query:

| Label | Meaning | Example (query → product) |
| --- | --- | --- |
| **E — Exact** | Exactly what the shopper searched for. | `red running shoes` → a pair of red running shoes |
| **S — Substitute** | A reasonable alternative that could still satisfy the shopper. | `red running shoes` → blue running shoes |
| **C — Complement** | A related item that goes *with* the searched item but is not the item itself. | `red running shoes` → running socks |
| **I — Irrelevant** | Not related to the search at all. | `red running shoes` → a kitchen blender |

---

## 6. Why convert E/S/C/I into 3/2/1/0?

Models work with **numbers**, not letters, and we also want to capture that some
matches are "better" than others. So we use a graded scale where a bigger number
means more relevant:

```
E = 3   (best match)
S = 2   (good alternative)
C = 1   (related but not the item)
I = 0   (not relevant)
```

This ordering (`3 > 2 > 1 > 0`) lets us later teach a model to put higher-scoring
products at the top of the search results.

---

## 7. Why do we join examples with products?

The examples file tells us **which** product was judged and **how** relevant it
is, but it does not contain the product's text. The products file has the text
but no relevance labels.

By joining the two tables on `product_id` (and locale), each row ends up having
**both** the relevance label **and** the product text. That combined row is what
a ranking model needs to learn from.

---

## 8. Why do we create `product_text`?

A product's information is split across several columns: title, description,
bullet points, brand, and color. A search model reads text, so it is easier to
give it **one** rich text field per product instead of five separate ones.

We therefore concatenate those columns into a single column called
`product_text`. This single field describes the product fully and is what future
stages will search and rank over.

---

## 9. Why create a smaller 50,000-row sample?

The full dataset has millions of rows. Running experiments on all of it would be
slow and could overwhelm a normal laptop.

So we take a smaller sample of up to **50,000 rows**, made **balanced** — we try
to include a fair number of each relevance level (3, 2, 1, 0). Without balancing,
one label could dominate and a model might just learn to always guess that label.
The balanced sample keeps development fast and fair. The full data is kept too;
the sample is only for quick work.

---

## 10. Why no BM25, FAISS, or Transformers yet?

Those are the actual ranking/search tools built later:

- **BM25** — a classic keyword-matching ranking method.
- **FAISS** — a library for fast similarity search over vectors.
- **Transformers** — modern neural language models.

But building on data you do not understand or trust leads to hidden bugs and
wasted time. The whole point of this stage is to **first** verify, clean, and
understand the data. With a solid, trusted dataset in hand, the later stages
become much easier and safer. This is exactly how real applied work is sequenced.

---

## 11. What files were created?

After running the preprocessing script, you will have:

- **`src/dataset_preprocessing.py`** — the script that does all the
  preprocessing work.
- **`data/processed/sample_esci_50k.parquet`** — the cleaned, balanced 50k
  sample with `product_text` and the numeric `relevance_score` column.
- **`results/week0_dataset_summary.txt`** — a human-readable report of
  everything the script found (shapes, columns, label distribution, join
  results, and so on).

The folders `src`, `data/processed`, `results`, and `docs` are also created if
they did not already exist.

---

## 12. What should the next stage focus on?

The next stage (Week 1) focuses on the first **retrieval baselines**. The goal is
not to build the final search system immediately, but to:

1. Build a **TF-IDF** baseline.
2. Build a **BM25** baseline.
3. Compare both methods.
4. Evaluate using **Precision@K**, **Recall@K**, and **NDCG@K**.

These are the first keyword-based retrieval methods. Later stages introduce
Sentence-Transformer retrieval, dense embeddings, FAISS vector search, hybrid
retrieval, transformer re-ranking, and fine-tuning. The purpose of Week 1 is to
establish simple baselines that future retrieval models must beat.

---

## Key Week 0 observations

| Observation | Value |
| --- | --- |
| Examples | ~2.6 million |
| Products | ~1.8 million |
| US subset | ~1.8 million rows |
| Unique queries | 130,000+ |
| Unique products | 1.8 million+ |
| Label distribution | Strong imbalance: **E ≫ S ≫ I ≫ C** |
| Working sample | Balanced 50k sample created |

### Why these observations matter

- **Dataset size (2.6M examples / 1.8M products):** The data is far too large to
  load and join naively in memory. This is exactly why the pipeline filters and
  samples on the lightweight examples table **before** attaching the heavy
  product text. It also tells us future neural methods will need batching and
  efficient indexing (e.g. FAISS) to scale.
- **US subset (1.8M rows):** After focusing on English (US), we still have
  plenty of data. We are not starved for examples, so a 50k sample is a
  reasonable, fast working subset.
- **Unique queries (130k+):** Many distinct search intents exist, so a good
  ranking system must generalize across many query types, not memorize a few.
  This is why query-length analysis matters.
- **Unique products (1.8M+):** The catalog is large, so retrieval must
  efficiently narrow millions of products down to a short, relevant list — the
  core challenge of search.
- **Strong label imbalance (E ≫ S ≫ I ≫ C):** Most judged pairs are "Exact"
  matches and very few are "Complement". Training on the raw distribution could
  let a model score well just by predicting "Exact" all the time. This is the key
  reason we built a **balanced** 50k sample, and why evaluation must use ranking
  metrics (Precision@K, Recall@K, NDCG@K) rather than plain accuracy.
- **Balanced 50k sample:** Gives every relevance level (3/2/1/0) fair
  representation and keeps development fast and reproducible.
