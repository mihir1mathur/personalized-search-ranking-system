# Week 1 — Retrieval Baselines (TF-IDF + BM25)

**Project:** Personalized Search Ranking System

This document explains the first two retrieval methods built for the project and
how they are evaluated. It is written in beginner-friendly language; no
machine-learning background is required.

In Week 0 we cleaned the data and built a balanced 50,000-row sample. In Week 1
we build our **first two search methods** and measure how good they are. These
are **baselines**: simple, explainable starting points that more advanced models
(in later stages) will have to beat.

---

## 1. What is retrieval?

**Retrieval** means: given a search query, quickly find the most relevant items
from a large collection.

When you type `wireless headphones`, the system must look through millions of
products and pull out the handful that best match what you typed. That
"find the best matches" step is retrieval. In Week 1, the collection is ~48,000
unique products, and for each query we retrieve the **Top 10**.

---

## 2. What is TF-IDF?

TF-IDF stands for **Term Frequency – Inverse Document Frequency**. It is a way to
score how important each word is in a document.

- **Term Frequency (TF):** a word that appears often in a product description is
  probably important to that product.
- **Inverse Document Frequency (IDF):** a word that appears in almost *every*
  product (like `the` or `and`) is not useful for telling products apart, so it
  gets a low weight. A rare, specific word (like `noise-cancelling`) gets a high
  weight.

TF-IDF turns each product into a list of numbers (a **vector**), where each
number is the TF-IDF weight of a word. We do the same to the query, then compare
the query vector to every product vector using **cosine similarity** (below).

---

## 3. What is BM25?

BM25 is another keyword-matching formula, and it is the classic workhorse of real
search engines. It is like a smarter cousin of TF-IDF, with two key improvements:

- **Saturation:** seeing a word 20 times is not 20× better than seeing it once.
  After a few repeats, extra mentions barely help. This stops "keyword stuffing"
  from winning.
- **Length normalization:** long documents naturally contain more words, so they
  would unfairly match more queries. BM25 gently discounts long documents so a
  short, on-topic product can still beat a long, rambling one.

We use the [`rank_bm25`](https://pypi.org/project/rank-bm25/) library to compute
BM25 scores.

---

## 4. Why build baselines?

A **baseline** is a simple method we build first. It matters because:

- It gives us a number to beat. If a fancy neural model later scores **worse**
  than BM25, we know something is wrong.
- It is fast, cheap, and easy to explain.
- It often works surprisingly well, so it tells us how hard the problem really is
  before we invest in complex models.

In serious applied work, you **always** start with strong, simple baselines.
Skipping them is a classic beginner mistake.

---

## 5. What is cosine similarity?

Once TF-IDF turns text into vectors, we need a way to measure how "close" two
vectors are. **Cosine similarity** measures the **angle** between two vectors:

- If the query and a product point in the **same** direction (share important
  words), the angle is small and the cosine is close to **1** → very similar.
- If they share no important words, they point in different directions, the angle
  is large, and the cosine is near **0** → not similar.

Cosine similarity ignores document length (it only cares about direction), which
is why it pairs naturally with TF-IDF vectors.

---

## 6. What is Precision@K?

**Precision@10** asks: *"Of the 10 products I showed, how many were actually
relevant?"*

```
Precision@10 = (relevant items in the top 10) / 10
```

If 3 of the top 10 are relevant, `Precision@10 = 0.3`.

> **Note for this project:** the dataset has only a **few** judged relevant
> products per query (often just one). So even a perfect system cannot fill all
> 10 slots with relevant items, and Precision@10 will look low on purpose. That
> is expected here; **Recall@10** and **NDCG@10** are more informative for this
> data.

---

## 7. What is Recall@K?

**Recall@10** asks: *"Of all the products I know are relevant, how many did I
manage to put in the top 10?"*

```
Recall@10 = (relevant items in the top 10) / (all relevant items)
```

If a query has 2 known relevant products and 1 appears in the top 10, then
`Recall@10 = 0.5`. Recall tells us whether the truly relevant products are being
pulled into the results at all.

---

## 8. What is NDCG?

NDCG stands for **Normalized Discounted Cumulative Gain**. It is the most
important ranking metric here because it cares about **order**, not just
presence.

- **Gain:** each result contributes its relevance score (`3` = highly relevant …
  `0` = irrelevant).
- **Discounted:** results lower down the list are worth less. A relevant item at
  rank 1 helps more than at rank 10.
- **Normalized:** we divide by the score of the **best possible** ordering,
  giving a fair number between 0 and 1.

A high NDCG means the system not only found the relevant products but also ranked
the **most** relevant ones at the very top.

---

## 9. Why does BM25 often outperform TF-IDF?

Because of the two improvements from Section 3:

- **Saturation** stops products that repeat a keyword many times from unfairly
  dominating.
- **Length normalization** keeps long descriptions from winning just because they
  contain more words.

Together these usually make BM25 rank the genuinely relevant products a bit
higher, which lifts Recall@10 and NDCG@10.

In this Week 1 run, **BM25 clearly beat TF-IDF on all three metrics** (see
[`results/week1_metrics.txt`](../results/week1_metrics.txt) for the exact numbers
and `results/tfidf_vs_bm25_metrics.png` for the chart). This matches what
research and industry experience predict.

**Where TF-IDF still shines:**

- Short, clean documents, where length matters less.
- When you need similarity vectors for other tasks (clustering, quick similarity,
  or as features for another model).
- As a dead-simple, transparent baseline that ships in minutes.

---

## Results

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

BM25 outperforms TF-IDF on every metric.

---

## Limitations of keyword retrieval

TF-IDF and BM25 are powerful, but they share one big weakness: they match
**words**, not **meaning**. A product is only matched if it literally contains
the query's words. This causes them to fail in several common situations:

- **Synonyms** (different words, same meaning): `charger cord` vs
  `charging cable`. A keyword method does not know `cord` and `cable` mean the
  same thing, so it can miss the right product entirely.
- **Abbreviations** (short forms): `pjs` vs `pajamas`, or `TV` vs `television`.
  The letters do not overlap, so keyword search treats them as unrelated words.
- **Spelling mistakes** (typos): `tankless watter heatre` would not match
  `tankless water heater`, because the misspelled words are literally different
  strings. Keyword search has no built-in idea of "close spelling".
- **Semantic meaning** (intent and themes): a `horror` bedding theme, or the word
  `portable`, expresses an **intent**. Keyword search cannot reason about intent;
  it only counts shared words, so it returns generic or off-topic items.
- **Different wording** (paraphrases): `laptop` vs `notebook computer` describe
  the same thing in different words. Keyword search sees no overlap and misses it.

**Quick examples of pairs keyword search cannot connect:**

```
pjs            <->  pajamas
TV             <->  television
laptop         <->  notebook computer
charger cord   <->  charging cable
```

**Why BM25 and TF-IDF cannot understand meaning:** both methods represent text
purely by *which* words appear and how often / how rare they are. They have no
knowledge that two different words can mean the same thing, because they never
learn relationships between words — they only count and weight the exact tokens.
So if the shopper's words and the product's words differ, the methods are blind
to the match, no matter how related the words are in meaning. (See
`results/week1_failure_cases.txt` for five real queries where exactly this
happens.)

---

## Why the next stage is necessary

The limitations above are not bugs we can patch — they are built into how keyword
matching works. To go further, we need methods that understand **meaning**. That
is what Week 2 introduces.

- **Embeddings:** an embedding is a list of numbers (a vector) that captures the
  **meaning** of a piece of text, learned by a neural network from huge amounts
  of language. Words and phrases with similar meanings get similar vectors — even
  if they use completely different letters.
- **Dense vectors:** unlike sparse TF-IDF vectors (mostly zeros, one slot per
  word), embeddings are short, "dense" vectors where every number carries
  meaning. `pjs` and `pajamas` end up close together in this vector space because
  they mean the same thing.
- **Semantic retrieval:** to search, we embed the query and embed the products,
  then find the products whose vectors are **closest** in meaning to the query's
  vector. This matches by meaning, not by exact words.

**Concrete example:**

```
Query: holiday pjs

Keyword search (BM25 / TF-IDF):
    does NOT understand "pjs", and treats "holiday" only as the literal
    word — so it can miss the Christmas pajamas product.

Embedding (semantic) search:
    understands that "pjs" means "pajamas" and that "holiday" relates to
    "Christmas", so it can surface the right product.
```

In short, Week 2 attempts to solve the exact problems keyword search cannot:
synonyms, abbreviations, typos, paraphrases, and intent. The goal is to **beat**
the strong BM25 baseline built this week, measured with the same metrics
(Precision@10, Recall@10, NDCG@10).

---

## Files created in Week 1

- **`src/tfidf_retriever.py`** — the TF-IDF retriever (`TfidfRetriever` class).
- **`src/bm25_retriever.py`** — the BM25 retriever (`BM25Retriever` class).
- **`evaluation/metrics.py`** — Precision@K, Recall@K, and NDCG@K functions.
- **`src/evaluate_retrieval.py`** — the main script: builds both retrievers,
  evaluates them, and writes all the outputs below.
- **`results/retrieval_results.csv`** — Top-10 results per query for both methods
  (method, query, product_id, score, rank, relevance_score).
- **`results/week1_metrics.txt`** — the comparison table and written
  explanations.
- **`results/sample_query_analysis.txt`** — 10 random queries with Top-5 results
  from each method.
- **`results/week1_failure_cases.txt`** — 5 real queries where keyword retrieval
  struggled, with explanations of why and what semantic search may understand.
- **`results/tfidf_vs_bm25_metrics.png`** — bar chart comparing the two methods.
- **`results/query_length_distribution.png`** — histogram of how many words
  queries contain.

---

## What the next stage should focus on

Week 2 should move beyond keyword matching toward **meaning-based ("semantic")
retrieval**, which can match `laptop` with `notebook computer` even when the
exact words differ:

- **Sentence-Transformer embeddings** (dense vectors).
- **FAISS** for fast vector similarity search.
- Possibly **hybrid retrieval** (combine BM25 + embeddings).
- Later: **transformer re-ranking** and **fine-tuning**.

The goal of all of these is to **beat** the BM25 baseline established this week,
measured with the same metrics.
