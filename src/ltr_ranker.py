"""
ltr_ranker.py  --  LightGBM LambdaMART Learning-to-Rank model (Week 5)
======================================================================

WHAT THIS DOES (beginner view)
------------------------------
This wraps a LightGBM RANKER -- a gradient boosted decision tree ensemble
trained with the LambdaMART objective -- behind a tiny, clean interface that
matches the rest of the project.

WHY A *RANKER*, NOT A REGRESSOR OR CLASSIFIER?
A plain regressor would try to predict each product's relevance in isolation
("this is a 2.0") and be punished for getting the absolute number wrong. But we
do not care about absolute numbers -- we care about ORDER within one query.
LambdaMART optimizes exactly that: it looks at PAIRS of products within the
same query and nudges the model to put the more-relevant one higher, weighting
each pair by how much swapping them would change NDCG. That is why it is the
classic, still-competitive algorithm for search ranking.

THE "group" CONCEPT (this trips everyone up once):
Training data is a big flat table of (query, product) rows. The model must know
which rows belong to the SAME query so it only compares products WITHIN a query
(comparing a product for "iphone charger" against one for "dog food" is
meaningless). We tell it via a `group` array: group = [12, 8, 20, ...] means
"the first 12 rows are query 1, the next 8 are query 2, ...". The rows must be
sorted so each query's rows are contiguous.

LABEL GAIN:
Our relevance labels are 0,1,2,3 (irrelevant .. exact match). LightGBM's NDCG
uses gain = 2**label - 1 by default, which needs a label_gain table covering
the max label. We pass label_gain explicitly so labels 0..3 are all valid.

This class exposes: fit(), predict(), rank_candidates(), feature_importance(),
save(), and load() -- the same lightweight shape as the Week 1-4 components.
"""

import numpy as np

try:
    import lightgbm as lgb
except ImportError:
    raise ImportError(
        "Learning-to-Rank needs LightGBM.\n"
        "Install it with:\n"
        "    pip install lightgbm"
    )

try:
    from ltr_features import FEATURE_NAMES, NUM_FEATURES
except ImportError:  # allow running from the project root too
    from src.ltr_features import FEATURE_NAMES, NUM_FEATURES  # type: ignore


# Default hyper-parameters. Kept SMALL and fixed on purpose: Week 5 is about
# clean engineering and reuse, not a giant hyper-parameter search. These are
# sensible, well-known LambdaMART defaults that train in seconds on our data.
#
# Two choices deserve a note because they are what let a LEARNED ranker safely
# match/beat a single very strong feature (the cross-encoder):
#   * DOMAIN-PRIOR MONOTONE CONSTRAINTS (see MONOTONE_CONSTRAINTS below): a
#     feature that should only ever HELP relevance (cross-encoder score, exact
#     match, brand match, ...) is constrained to move the ranking score in that
#     direction only. This encodes what we know and prevents the model from
#     overfitting noisy inversions on a small training set.
#   * Strong regularization + early stopping (small leaves, high
#     min_child_samples, L2, and a validation carve-out) so the model does not
#     chase spurious splits on weak features (length, etc.).
DEFAULT_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [10],
    "boosting_type": "gbdt",
    "n_estimators": 600,        # upper bound; early stopping picks the best round
    "learning_rate": 0.03,      # small step size -> steadier, less overfit
    "num_leaves": 15,           # modest capacity (2^4 - 1) to resist overfitting
    "min_child_samples": 50,    # a leaf must cover >=50 rows (regularization)
    "subsample": 0.8,           # row sampling per tree (bagging)
    "subsample_freq": 1,        # apply subsampling every round
    "colsample_bytree": 0.8,    # feature sampling per tree
    "reg_lambda": 5.0,          # L2 regularization (stronger)
    "min_split_gain": 0.0,
    "max_bin": 1023,            # fine binning preserves the cross-encoder order
    "random_state": 42,         # reproducible training
    "n_jobs": -1,               # use all CPU cores
    "verbose": -1,              # keep LightGBM quiet
}

# Monotone constraints, ONE per feature in FEATURE_NAMES order:
#   +1 : ranking score must be NON-DECREASING in this feature (more = better)
#   -1 : ranking score must be NON-INCREASING (more = worse)
#    0 : no constraint (let the trees decide)
# These are domain priors, not tuning: a higher cross-encoder / hybrid / BM25 /
# embedding score, more query-title overlap, an exact match, or a brand/color
# match can only make a product MORE relevant; a worse stage-1 rank (bigger
# number) can only make it LESS relevant. Lengths and the score-gap have no
# clear prior direction, so they stay unconstrained.
_MONOTONE_BY_NAME = {
    "ce_score": 1,
    "ce_rank": -1,
    "hybrid_score": 1,
    "bm25_norm": 1,
    "emb_norm": 1,
    "bm25_emb_gap": 0,
    "hybrid_rank": -1,
    "query_len": 0,
    "title_len": 0,
    "text_len_log": 0,
    "title_overlap_count": 1,
    "title_overlap_ratio": 1,
    "title_exact_contains": 1,
    "brand_match": 1,
    "color_match": 1,
}
MONOTONE_CONSTRAINTS = [_MONOTONE_BY_NAME[name] for name in FEATURE_NAMES]

# Relevance labels are 0..3. label_gain[label] = gain used inside NDCG.
# 2**label - 1 is the standard graded-relevance gain (0,1,3,7).
LABEL_GAIN = [2 ** r - 1 for r in range(4)]  # -> [0, 1, 3, 7]


class LTRRanker:
    """A thin, well-documented wrapper around a LightGBM LambdaMART ranker."""

    def __init__(self, params=None):
        self.params = dict(DEFAULT_PARAMS)
        # Apply the domain-prior monotone constraints (in FEATURE_NAMES order).
        self.params["monotone_constraints"] = list(MONOTONE_CONSTRAINTS)
        if params:
            self.params.update(params)
        self.model = None
        self.feature_names = list(FEATURE_NAMES)

    # ----------------------------------------------------------------------
    # Train
    # ----------------------------------------------------------------------
    def fit(self, X, y, group, eval_set=None, eval_group=None):
        """
        Train the ranker.

        X          : (N x NUM_FEATURES) feature matrix (rows grouped by query).
        y          : (N,) integer relevance labels 0..3.
        group      : list of per-query row counts, summing to N (query blocks).
        eval_set   : optional (X_val, y_val) for early-stopping visibility.
        eval_group : per-query counts for the eval_set.
        """
        X = np.asarray(X, dtype="float32")
        y = np.asarray(y, dtype="int32")

        if X.shape[1] != NUM_FEATURES:
            raise ValueError(
                f"Expected {NUM_FEATURES} features, got {X.shape[1]}. "
                "ltr_features.FEATURE_NAMES and the matrix are out of sync."
            )

        self.model = lgb.LGBMRanker(**self.params, label_gain=LABEL_GAIN)

        fit_kwargs = {"group": group, "feature_name": self.feature_names}
        if eval_set is not None and eval_group is not None:
            # Early stopping on a query-grouped validation set: train up to
            # n_estimators rounds but keep the round with the best NDCG@10, so
            # the model does not overfit past its optimum.
            fit_kwargs["eval_set"] = [eval_set]
            fit_kwargs["eval_group"] = [eval_group]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ]

        self.model.fit(X, y, **fit_kwargs)
        return self

    # ----------------------------------------------------------------------
    # Predict raw scores
    # ----------------------------------------------------------------------
    def predict(self, X):
        """Return a raw LTR relevance score for each row (higher = better)."""
        if self.model is None:
            raise RuntimeError("LTRRanker.predict called before fit/load.")
        X = np.asarray(X, dtype="float32")
        if X.shape[0] == 0:
            return np.zeros(0, dtype="float64")
        return np.asarray(self.model.predict(X), dtype="float64")

    # ----------------------------------------------------------------------
    # Rank a single query's candidates
    # ----------------------------------------------------------------------
    def rank_candidates(self, candidate_ids, feature_matrix, top_k=10):
        """
        Score one query's candidates and return the Top-K, best first.

        candidate_ids  : list of product_id strings (aligned with rows).
        feature_matrix : (len(candidate_ids) x NUM_FEATURES) features.
        Returns [(product_id, ltr_score, rank), ...] -- same shape as the
        Week 1-4 retrievers, so the SAME metric functions score it.
        """
        if len(candidate_ids) == 0:
            return []
        scores = self.predict(feature_matrix)
        order = np.argsort(-scores)[:top_k]
        return [
            (str(candidate_ids[pos]), float(scores[pos]), rank)
            for rank, pos in enumerate(order, start=1)
        ]

    # ----------------------------------------------------------------------
    # Feature importance (why the model ranks the way it does)
    # ----------------------------------------------------------------------
    def feature_importance(self, importance_type="gain"):
        """
        Return [(feature_name, importance), ...] sorted high-to-low.
        'gain' = total reduction in loss contributed by splits on that feature
        (the most meaningful importance for interpreting the model).
        """
        if self.model is None:
            raise RuntimeError("feature_importance called before fit/load.")
        booster = self.model.booster_
        importances = booster.feature_importance(importance_type=importance_type)
        pairs = list(zip(self.feature_names, [float(v) for v in importances]))
        pairs.sort(key=lambda kv: -kv[1])
        return pairs

    # ----------------------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------------------
    def save(self, path):
        """Save the trained booster to a text file (LightGBM's native format)."""
        if self.model is None:
            raise RuntimeError("save called before fit.")
        self.model.booster_.save_model(path)

    def load(self, path):
        """Load a booster previously written by save(); enables predict()."""
        booster = lgb.Booster(model_file=path)

        # Wrap the raw booster so predict()/feature_importance() still work.
        class _LoadedModel:
            def __init__(self, b):
                self.booster_ = b

            def predict(self, X):
                return b.predict(X)

        b = booster
        self.model = _LoadedModel(b)
        return self


# ---------------------------------------------------------------------------
# Tiny self-test on synthetic data (runs in well under a second).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # 3 queries, 6 candidates each. Feature 0 (ce_score) correlates with label.
    n_q, n_c = 3, 6
    X, y, group = [], [], []
    for _ in range(n_q):
        for _ in range(n_c):
            label = int(rng.integers(0, 4))
            row = [label + rng.normal(0, 0.3)] + rng.normal(0, 1, NUM_FEATURES - 1).tolist()
            X.append(row)
            y.append(label)
        group.append(n_c)
    X = np.asarray(X, dtype="float32")

    ranker = LTRRanker(params={"n_estimators": 50})
    ranker.fit(X, y, group)
    print("Top features by gain:")
    for name, imp in ranker.feature_importance()[:5]:
        print(f"   {name:<22} {imp:.1f}")
    print("Self-test OK.")
