"""
evaluation.py
─────────────
Technical evaluation of the two intelligent features described in the
project contract, addressing the objective:

    "To evaluate the technical performance of the categorisation and
    anomaly-detection mechanisms, including classifier accuracy/precision/
    recall and the reliability of anomaly flags."

Run directly:  python evaluation.py
Outputs are printed to the console and also written to evaluation_report.txt
so they can be pasted / screenshotted straight into the dissertation.

Two independent evaluations:

1. CLASSIFIER EVALUATION (suggest_category)
   A hand-labelled test set of realistic student transaction descriptions
   (NOT the same phrases as SEED_EXAMPLES, so the model is being tested on
   unseen wording) is split into train/test. The model is trained on the
   seed set + the train split, then scored on the held-out test split using
   accuracy, precision, recall, F1 (sklearn.metrics.classification_report)
   and a confusion matrix.

2. ANOMALY-DETECTION EVALUATION (get_insights)
   Synthetic per-category spending histories are constructed with a known
   percentage change injected in the "current" month, at several points
   around the 25% threshold and around the 3-month minimum-history rule.
   Expected vs. actual flagging behaviour is compared to check the
   detector behaves as designed at its boundaries.
"""

import random
import sqlite3
from collections import defaultdict

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from ai_insights import SEED_EXAMPLES, get_insights


# ════════════════════════════════════════════════════════════════════════
# 1. CLASSIFIER EVALUATION
# ════════════════════════════════════════════════════════════════════════

# Hand-labelled held-out examples: realistic student transaction text that
# does NOT appear in SEED_EXAMPLES, so this measures generalisation rather
# than memorisation. Extend this list freely for a bigger/stronger eval.
TEST_EXAMPLES_EXPENSE = [
    ("weekly shop at netto", "Food & Drinks"),
    ("cafe latte with friends", "Food & Drinks"),
    ("dominos pizza delivery", "Food & Drinks"),
    ("lunch at canteen", "Food & Drinks"),
    ("sushi dinner", "Food & Drinks"),
    ("metro card top up", "Transport"),
    ("uber home from town", "Transport"),
    ("dsb ticket to odense", "Transport"),
    ("fuel for car", "Transport"),
    ("flat rent payment", "Rent"),
    ("rent to landlord", "Rent"),
    ("broadband bill", "Utilities"),
    ("electric bill payment", "Utilities"),
    ("h&m new jumper", "Shopping"),
    ("online shopping order", "Shopping"),
    ("nike trainers", "Shopping"),
    ("cinema with friends", "Entertainment"),
    ("football match ticket", "Entertainment"),
    ("switch game purchase", "Entertainment"),
    ("doctor appointment fee", "Health"),
    ("protein supplements", "Health"),
    ("gym monthly fee", "Health"),
    ("new textbook for module", "Education"),
    ("printing costs", "Education"),
    ("disney plus subscription", "Subscriptions"),
    ("spotify family plan", "Subscriptions"),
    ("put money into savings", "Savings"),
    ("atm withdrawal", "Other"),
    ("misc costs", "Other"),
]

TEST_EXAMPLES_INCOME = [
    ("monthly pay from work", "Salary"),
    ("wage payment", "Salary"),
    ("shift at the bar", "Part-time Job"),
    ("weekend retail shift pay", "Part-time Job"),
    ("freelance design work", "Freelance"),
    ("client invoice paid", "Freelance"),
    ("su loan payment", "Student Loan"),
    ("loan instalment", "Student Loan"),
    ("parents transferred money", "Family Support"),
    ("mum sent cash", "Family Support"),
    ("uni scholarship payment", "Scholarship"),
    ("grant award", "Scholarship"),
    ("amazon refund", "Other Income"),
    ("birthday money", "Other Income"),
]


def _make_inmemory_db():
    """A blank in-memory transactions table so suggest_category-style
    training can run without touching the real data.db."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE transactions (id INTEGER PRIMARY KEY, user_id INTEGER, "
        "type TEXT, description TEXT, category TEXT, amount REAL, date TEXT)"
    )
    return conn


def evaluate_classifier(txn_type, test_examples, seed_split=0.7, random_state=42):
    """Train on SEED_EXAMPLES + a train-split of hand-labelled examples,
    test on the held-out split, and report accuracy/precision/recall/F1 +
    confusion matrix. Mirrors _train_model()'s combination logic in
    ai_insights.py, but keeps a clean train/test separation."""
    texts = [t for t, _ in test_examples]
    labels = [l for _, l in test_examples]

    # Note: with a small hand-labelled set spread across many categories,
    # stratified splitting isn't reliable (some categories have only 1-2
    # examples), so this uses a plain random split. Grow TEST_EXAMPLES_*
    # over time for a more robust per-category breakdown.
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=1 - seed_split, random_state=random_state,
    )

    # Training data = seed examples + train split (same combination the
    # real app uses, just with a proper held-out test split on top)
    train_all_texts = [t for t, _ in SEED_EXAMPLES[txn_type]] + list(train_texts)
    train_all_labels = [l for _, l in SEED_EXAMPLES[txn_type]] + list(train_labels)

    vec = CountVectorizer(lowercase=True, ngram_range=(1, 3))  # must match ai_insights.py's _train_model
    X_train = vec.fit_transform(train_all_texts)
    clf = MultinomialNB()
    clf.fit(X_train, train_all_labels)

    X_test = vec.transform(test_texts)
    predictions = clf.predict(X_test)

    acc = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions, zero_division=0)
    labels_sorted = sorted(set(test_labels) | set(predictions))
    cm = confusion_matrix(test_labels, predictions, labels=labels_sorted)

    return {
        "txn_type": txn_type,
        "n_train": len(train_all_texts),
        "n_test": len(test_texts),
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm,
        "confusion_labels": labels_sorted,
        "test_texts": test_texts,
        "test_labels": test_labels,
        "predictions": predictions,
    }


def format_confusion_matrix(cm, labels):
    col_w = max(len(l) for l in labels) + 2
    header = " " * col_w + "".join(f"{l[:8]:>10}" for l in labels)
    lines = [header]
    for i, row_label in enumerate(labels):
        row = f"{row_label:<{col_w}}" + "".join(f"{v:>10}" for v in cm[i])
        lines.append(row)
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# 2. ANOMALY-DETECTION EVALUATION
# ════════════════════════════════════════════════════════════════════════

# Each scenario: a 3-month baseline for one category, a current-month spend,
# and whether an anomaly SHOULD be flagged given the 25% threshold and the
# 3-month minimum-history rule now enforced in get_insights().
ANOMALY_SCENARIOS = [
    # (label, history_3_months, current_month_spend, expect_flag, note)
    ("baseline (no change)",        [150, 160, 155], 155,  False, "current ~ average"),
    ("small increase (+10%)",       [150, 160, 155], 172,  False, "below 25% threshold"),
    ("just under threshold (+24%)", [150, 160, 155], 192,  False, "just below 25%"),
    ("just over threshold (+25%)",  [150, 160, 155], 194,  True,  "just above 25%"),
    ("large increase (+50%)",       [150, 160, 155], 233,  True,  "well above threshold"),
    ("extreme increase (+100%)",    [150, 160, 155], 310,  True,  "double average"),
    ("small decrease (-10%)",       [150, 160, 155], 140,  False, "below threshold (magnitude)"),
    ("large decrease (-30%)",       [150, 160, 155], 109,  True,  "'nice, you spent less' flag"),
    ("only 2 months history",       [150, 160],      300,  False, "below 3-month minimum -> must be skipped"),
    ("only 1 month history",        [150],           300,  False, "below 3-month minimum -> must be skipped"),
    ("zero history (new category)", [],              200,  False, "no baseline possible"),
]


def _build_history_rows(user_id, category, history_values, current_value, current_month="2026-08"):
    """Builds fake per-month rows for a single category matching the shape
    get_insights() expects, then hands them back via a stub db object."""
    rows = []
    # place history in the months preceding current_month
    months = []
    y, m = 2026, 8
    for _ in range(len(history_values)):
        m -= 1
        if m == 0:
            m, y = 12, y - 1
        months.append(f"{y:04d}-{m:02d}")
    months = list(reversed(months))
    for month, value in zip(months, history_values):
        rows.append({"month": month, "category": category, "total": value})
    if current_value is not None:
        rows.append({"month": current_month, "category": category, "total": current_value})
    return rows


class _StubDB:
    """Minimal stand-in for the sqlite connection get_insights() expects,
    so a single synthetic scenario can be fed straight to it."""
    def __init__(self, rows):
        self._rows = rows

    def execute(self, query, params):
        return self

    def fetchall(self):
        return self._rows


def evaluate_anomaly_detection():
    results = []
    for label, history, current, expect_flag, note in ANOMALY_SCENARIOS:
        rows = _build_history_rows(1, "TestCategory", history, current)
        db = _StubDB(rows)
        insights = get_insights(db, user_id=1, current_month="2026-08")
        actual_flag = any(i["category"] == "TestCategory" for i in insights)
        correct = actual_flag == expect_flag
        results.append({
            "label": label,
            "history": history,
            "current": current,
            "expected": expect_flag,
            "actual": actual_flag,
            "correct": correct,
            "note": note,
        })
    return results


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    out_lines = []

    def log(line=""):
        print(line)
        out_lines.append(line)

    log("=" * 72)
    log("1. CLASSIFIER EVALUATION (suggest_category / Naive Bayes)")
    log("=" * 72)

    for txn_type, test_examples in [("expense", TEST_EXAMPLES_EXPENSE),
                                     ("income", TEST_EXAMPLES_INCOME)]:
        res = evaluate_classifier(txn_type, test_examples)
        log(f"\n--- {txn_type.upper()} classifier ---")
        log(f"Trained on {res['n_train']} examples (seed + train split), "
            f"tested on {res['n_test']} held-out examples")
        log(f"Accuracy: {res['accuracy']:.3f}")
        log("\nPrecision / Recall / F1 per category:")
        log(res["report"])
        log("Confusion matrix (rows = true category, columns = predicted):")
        log(format_confusion_matrix(res["confusion_matrix"], res["confusion_labels"]))
        log("\nMisclassified examples:")
        n_wrong = 0
        for text, true_l, pred_l in zip(res["test_texts"], res["test_labels"], res["predictions"]):
            if true_l != pred_l:
                n_wrong += 1
                log(f"  '{text}'  true={true_l}  predicted={pred_l}")
        if n_wrong == 0:
            log("  (none)")

    log("\n" + "=" * 72)
    log("2. ANOMALY-DETECTION EVALUATION (get_insights / statistical baselining)")
    log("=" * 72)
    anomaly_results = evaluate_anomaly_detection()
    n_correct = sum(r["correct"] for r in anomaly_results)
    log(f"\n{n_correct}/{len(anomaly_results)} scenarios behaved as expected\n")
    header = f"{'Scenario':<32}{'History':<18}{'Current':>8}{'Expect':>9}{'Actual':>9}{'OK?':>6}"
    log(header)
    log("-" * len(header))
    for r in anomaly_results:
        log(f"{r['label']:<32}{str(r['history']):<18}{r['current']:>8}"
            f"{str(r['expected']):>9}{str(r['actual']):>9}{'✓' if r['correct'] else '✗':>6}")
        if not r["correct"]:
            log(f"    -> UNEXPECTED: {r['note']}")

    log("\n" + "=" * 72)
    log("Summary for dissertation")
    log("=" * 72)
    log("- Report the accuracy/precision/recall/F1 and confusion matrix figures "
        "above per transaction type (expense/income) as the classifier evaluation.")
    log("- Report the anomaly scenario table above as evidence the 25% threshold "
        "and 3-month minimum-history rule behave correctly at their boundaries.")
    log("- If any classifier categories show weak recall/precision, discuss likely "
        "causes (e.g. overlapping vocabulary between categories, small seed set) "
        "in the critical analysis chapter.")

    with open("evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    print("\n(Full report written to evaluation_report.txt)")


if __name__ == "__main__":
    main()
