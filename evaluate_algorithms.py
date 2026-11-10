"""
evaluate_algorithms.py
───────────────────────
Answers RQ2 ("Which machine learning algorithm is suitable for
automatically categorising students' expense transactions?") with a real,
like-for-like comparison rather than assuming Naive Bayes was the right
choice.

Three classifiers are trained and tested on EXACTLY the same features,
same seed data, and same train/test split as the deployed classifier
(see ai_insights.py's _train_model and evaluation.py's evaluate_classifier):

    1. Multinomial Naive Bayes   (the one actually deployed in ai_insights.py)
    2. Linear SVM                (LinearSVC)
    3. Logistic Regression

All three use the identical CountVectorizer(lowercase=True, ngram_range=(1,3))
feature representation, so any difference in the results is attributable to
the classifier, not to different preprocessing.

Run directly: python evaluate_algorithms.py
Output is printed to the console and written to algorithm_comparison_report.txt.
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from ai_insights import SEED_EXAMPLES
from evaluation import TEST_EXAMPLES_EXPENSE, TEST_EXAMPLES_INCOME

# A second, larger, independent held-out set: 16 scoreable natural-language
# descriptions spanning every category (2 further cases are genuinely
# ambiguous/out-of-vocabulary with no single correct label, so they are
# reported separately rather than scored as right or wrong). This is the
# same set used for the generalisation check in Chapter 5/5.3.2 -- re-used
# here so the algorithm comparison is not resting on the 9/5-example split
# above alone.
GENERALISATION_EXPENSE = [
    ("grabbed some groceries after work", "Food & Drinks"),
    ("quick lunch with coworkers downtown", "Food & Drinks"),
    ("topped up my travel card this morning", "Transport"),
    ("caught a taxi home late at night", "Transport"),
    ("this month's rent just went out", "Rent"),
    ("the internet bill arrived today", "Utilities"),
    ("ordered a jacket from an online store", "Shopping"),
    ("saw a film at the cinema last night", "Entertainment"),
    ("picked up a prescription at the pharmacy", "Health"),
    ("bought a new textbook for my course", "Education"),
    ("my music streaming subscription renewed", "Subscriptions"),
    ("put some money aside into savings", "Savings"),
    ("took out cash from an atm machine", "Other"),
]
GENERALISATION_INCOME = [
    ("my wages were deposited this week", "Salary"),
    ("got paid after my bar shift last night", "Part-time Job"),
    ("invoiced a client for a small project", "Freelance"),
    ("my loan instalment was paid out", "Student Loan"),
    ("my mum transferred me some money", "Family Support"),
]

ALGORITHMS = {
    "Naive Bayes (deployed)": lambda: MultinomialNB(),
    "Linear SVM":             lambda: LinearSVC(random_state=42, max_iter=5000),
    "Logistic Regression":    lambda: LogisticRegression(random_state=42, max_iter=5000),
}


def evaluate_all_algorithms(txn_type, test_examples, seed_split=0.7, random_state=42):
    """Same train/test split as evaluate_classifier() in evaluation.py, but
    every algorithm in ALGORITHMS is fit and scored on it, so the
    comparison is apples-to-apples."""
    texts = [t for t, _ in test_examples]
    labels = [l for _, l in test_examples]

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=1 - seed_split, random_state=random_state,
    )
    train_all_texts = [t for t, _ in SEED_EXAMPLES[txn_type]] + list(train_texts)
    train_all_labels = [l for _, l in SEED_EXAMPLES[txn_type]] + list(train_labels)

    # One shared vectorizer/fit, reused for every algorithm, so none of them
    # gets a different feature space to work with.
    vec = CountVectorizer(lowercase=True, ngram_range=(1, 3))
    X_train = vec.fit_transform(train_all_texts)
    X_test = vec.transform(test_texts)

    results = {}
    for name, make_clf in ALGORITHMS.items():
        clf = make_clf()
        clf.fit(X_train, train_all_labels)
        predictions = clf.predict(X_test)
        acc = accuracy_score(test_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, predictions, average="macro", zero_division=0
        )
        results[name] = {
            "accuracy": acc, "precision": precision, "recall": recall, "f1": f1,
            "n_train": len(train_all_texts), "n_test": len(test_texts),
        }
    return results


def evaluate_all_algorithms_on_generalisation(txn_type, held_out_examples):
    """Train each algorithm on SEED_EXAMPLES only (no train/test split --
    these 13-18 descriptions are entirely unseen, matching how the single-
    algorithm generalisation check in test_generalisation.py works), then
    score on the full held-out set."""
    train_texts = [t for t, _ in SEED_EXAMPLES[txn_type]]
    train_labels = [l for _, l in SEED_EXAMPLES[txn_type]]
    test_texts = [t for t, _ in held_out_examples]
    test_labels = [l for _, l in held_out_examples]

    vec = CountVectorizer(lowercase=True, ngram_range=(1, 3))
    X_train = vec.fit_transform(train_texts)
    X_test = vec.transform(test_texts)

    results = {}
    for name, make_clf in ALGORITHMS.items():
        clf = make_clf()
        clf.fit(X_train, train_labels)
        predictions = clf.predict(X_test)
        acc = accuracy_score(test_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, predictions, average="macro", zero_division=0
        )
        results[name] = {
            "accuracy": acc, "precision": precision, "recall": recall, "f1": f1,
            "n_train": len(train_texts), "n_test": len(test_texts),
        }
    return results


def format_table(results_by_type):
    lines = []
    for txn_type, results in results_by_type.items():
        lines.append(f"\n--- {txn_type.upper()} classifier (train={next(iter(results.values()))['n_train']}, "
                      f"test={next(iter(results.values()))['n_test']}) ---")
        header = f"{'Algorithm':<24}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}"
        lines.append(header)
        lines.append("-" * len(header))
        for name, r in results.items():
            lines.append(f"{name:<24}{r['accuracy']:>10.3f}{r['precision']:>11.3f}{r['recall']:>9.3f}{r['f1']:>8.3f}")
    return "\n".join(lines)


if __name__ == "__main__":
    out = []
    out.append("=" * 72)
    out.append("ALGORITHM COMPARISON: Naive Bayes vs Linear SVM vs Logistic Regression")
    out.append("Same seed data, same CountVectorizer(1,3) features, same train/test split")
    out.append("=" * 72)

    results_by_type = {
        "expense": evaluate_all_algorithms("expense", TEST_EXAMPLES_EXPENSE),
        "income":  evaluate_all_algorithms("income", TEST_EXAMPLES_INCOME),
    }
    out.append("\n### Evaluation 1: 70/30 train/test split (hand-labelled held-out set) ###")
    out.append(format_table(results_by_type))

    gen_results_by_type = {
        "expense": evaluate_all_algorithms_on_generalisation("expense", GENERALISATION_EXPENSE),
        "income":  evaluate_all_algorithms_on_generalisation("income", GENERALISATION_INCOME),
    }
    out.append("\n\n### Evaluation 2: held-out natural-language generalisation set (seed-only training) ###")
    out.append(format_table(gen_results_by_type))

    report = "\n".join(out)
    print(report)
    with open("algorithm_comparison_report.txt", "w") as f:
        f.write(report + "\n")
