# Expense Tracking App — Personal Income & Expense Tracker

A simple, free, student-focused web application built with **Python Flask + SQLite**.

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
This installs Flask and **scikit-learn** — scikit-learn is required for the real Naive Bayes category-suggestion model; without it the app still runs but silently falls back to simple keyword matching instead.

### 2. Run the app
```bash
cd expense_tracker
python app.py
```

### 3. Open in browser
```
http://localhost:5000
```

Register a new account and start tracking!

---

## Tech Stack

| Layer       | Technology                       |
|-------------|----------------------------------|
| Backend     | Python 3 + Flask                 |
| Database    | SQLite (file: `data.db`)         |
| Frontend    | HTML5, CSS3, Vanilla JavaScript  |
| Charts      | Chart.js (CDN)                   |
| Fonts       | Google Fonts — DM Serif Display + DM Sans |

## Features

- **Authentication** — Register / login with hashed passwords
- **Dashboard** — Monthly overview, quick-add form, recent transactions, spending insights
- **Transactions** — Full CRUD (add, edit, delete), filter by type/category/month, free-text search, custom categories
- **AI category suggestion** — Naive Bayes classifier (scikit-learn) suggests a category as you type a description, seeded with example phrases and refined on your own transaction history; you confirm or change it, it's never applied automatically
- **Spending insights** — flags categories where this month's spend deviates 25%+ from your own recent average
- **Budgets** — Set monthly limits per category, visual progress bars with warnings
- **Analytics** — Doughnut charts, 6-month trend, savings rate, spending insights
- **Account** — export all your data as a JSON file, or permanently delete your account and data
- **Mobile responsive** — Bottom navigation bar on small screens

## Project Structure

```
expense_tracker/
├── app.py                  ← Flask app + API routes
├── ai_insights.py          ← Category-suggestion (Naive Bayes) and spending-insight logic
├── evaluation.py           ← Technical evaluation: classifier accuracy/precision/recall/F1 + anomaly-flag tests
├── evaluation_report.txt   ← Output of the last evaluation.py run
├── requirements.txt
├── data.db                 ← Created automatically on first run
└── templates/
    ├── base.html           ← Shared layout + sidebar nav
    ├── auth.html           ← Register / Login
    ├── dashboard.html      ← Home screen
    ├── transactions.html
    ├── budgets.html
    ├── analytics.html
    └── account.html        ← Data export / account deletion
```

## API Endpoints

| Method | Endpoint                | Description                                   |
|--------|--------------------------|------------------------------------------------|
| POST   | /login                  | Authenticate user                              |
| POST   | /register                | Create account                                 |
| GET/POST/PUT/DELETE | /api/transactions | List (filterable by type/category/`q` search)/add/update/delete transactions |
| GET    | /api/summary             | Monthly totals + chart data                    |
| GET/POST/DELETE | /api/budgets       | List/create-or-update/remove a budget          |
| GET    | /api/recent               | Last 8 transactions (dashboard)                |
| GET/POST/DELETE | /api/categories    | List (defaults + your own)/add/remove a custom category |
| POST   | /api/suggest-category     | Naive Bayes category suggestion for a description |
| GET    | /api/insights              | This month's spending-anomaly alerts           |
| GET    | /api/export                | Download all your data as JSON                 |
| DELETE | /api/account                | Permanently delete your account + data (requires password) |

## Categories

**Income:** Salary, Part-time Job, Freelance, Student Loan, Family Support, Scholarship, Other Income

**Expense:** Food & Drinks, Transport, Rent, Utilities, Shopping, Entertainment, Health, Education, Subscriptions, Savings, Other

You can also add your own categories from the Transactions page ("+ New category") — these are per-account and appear alongside the defaults everywhere in the app.

## Running the technical evaluation

```bash
python evaluation.py
```
Trains and tests the Naive Bayes classifier on held-out phrases (accuracy, precision, recall, F1, confusion matrix) and checks the anomaly-detection thresholds against synthetic spending histories. Results print to the console and are written to `evaluation_report.txt`.
