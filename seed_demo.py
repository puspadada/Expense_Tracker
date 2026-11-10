"""
seed_demo.py
────────────
Run this ONCE before showing the app to your supervisor. It creates a demo
account with a few months of realistic transaction history, so:

  - the dashboard/analytics charts have something to show (not empty)
  - the "spending insight" alert actually fires (it needs 3 months of
    prior history before it will flag anything — with a brand-new empty
    account you would not see this feature live)
  - a budget is already set up so the progress bars aren't empty either

Usage:
    python seed_demo.py

Login afterwards with:
    username: demo
    password: demo1234
"""
import sqlite3, hashlib, random
from datetime import date

DB = 'data.db'

def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def month_str(year, month):
    return f"{year:04d}-{month:02d}"

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

# Make sure tables exist (same schema as app.py's init_db)
conn.executescript('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        currency TEXT DEFAULT 'GBP',
        created_at TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        type TEXT NOT NULL CHECK(type IN ('income','expense')),
        amount REAL NOT NULL,
        category TEXT NOT NULL,
        description TEXT,
        date TEXT NOT NULL,
        created_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY(user_id) REFERENCES users(id)
    );
    CREATE TABLE IF NOT EXISTS budgets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        category TEXT NOT NULL,
        limit_amount REAL NOT NULL,
        month TEXT NOT NULL,
        UNIQUE(user_id, category, month),
        FOREIGN KEY(user_id) REFERENCES users(id)
    );
    CREATE TABLE IF NOT EXISTS custom_categories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        type TEXT NOT NULL CHECK(type IN ('income','expense')),
        name TEXT NOT NULL,
        UNIQUE(user_id, type, name),
        FOREIGN KEY(user_id) REFERENCES users(id)
    );
''')

# Remove any previous demo account so this script is safe to re-run
old = conn.execute('SELECT id FROM users WHERE username=?', ('demo',)).fetchone()
if old:
    uid = old['id']
    conn.execute('DELETE FROM transactions WHERE user_id=?', (uid,))
    conn.execute('DELETE FROM budgets WHERE user_id=?', (uid,))
    conn.execute('DELETE FROM custom_categories WHERE user_id=?', (uid,))
    conn.execute('DELETE FROM users WHERE id=?', (uid,))

cur = conn.execute(
    'INSERT INTO users (username,email,password) VALUES (?,?,?)',
    ('demo', 'demo@example.com', hash_pw('demo1234'))
)
uid = cur.lastrowid

random.seed(42)  # reproducible demo data

# 3 prior months (June, July, August 2026) + current month (September 2026)
months = [(2026, 6), (2026, 7), (2026, 8), (2026, 9)]

# category -> (typical monthly total, per-transaction description pool)
EXPENSE_PLAN = {
    'Food & Drinks': (220, ['netto weekly shop', 'coffee at starbucks', 'takeaway friday', 'irma groceries']),
    'Transport':     (60,  ['rejsekort top up', 'dsb train ticket to campus', 'uber ride home']),
    'Rent':          (450, ['monthly rent payment']),
    'Utilities':     (55,  ['electricity bill', 'internet bill']),
    'Entertainment': (40,  ['cinema ticket', 'spotify premium', 'netflix subscription']),
    'Education':     (25,  ['textbook purchase', 'stationery']),
}
INCOME_PLAN = {
    'Salary':        (0, []),  # not used
    'Part-time Job': (650, ['bar shift pay', 'retail shift pay']),
    'Student Loan':  (450, ['su payment']),
}

def spread_amount(total, n):
    """Split a monthly total into n roughly-even-but-varied transaction amounts."""
    parts = [random.uniform(0.6, 1.4) for _ in range(n)]
    s = sum(parts)
    return [round(total * p / s, 2) for p in parts]

for (y, m) in months:
    is_current = (y, m) == (2026, 9)
    for category, (base_total, descs) in EXPENSE_PLAN.items():
        total = base_total
        if is_current and category == 'Food & Drinks':
            total = base_total * 1.45  # deliberate ~45% spike this month -> triggers an alert
        if is_current and category == 'Entertainment':
            total = base_total * 0.55  # deliberate underspend -> triggers a "spent less" alert
        n_txns = max(1, len(descs))
        amounts = spread_amount(total, n_txns)
        for i, amt in enumerate(amounts):
            day = random.randint(1, 27)
            desc = descs[i % len(descs)] if descs else category.lower()
            conn.execute(
                'INSERT INTO transactions (user_id,type,amount,category,description,date) VALUES (?,?,?,?,?,?)',
                (uid, 'expense', amt, category, desc, f"{y:04d}-{m:02d}-{day:02d}")
            )
    for category, (base_total, descs) in INCOME_PLAN.items():
        if base_total == 0:
            continue
        day = random.randint(1, 5)
        conn.execute(
            'INSERT INTO transactions (user_id,type,amount,category,description,date) VALUES (?,?,?,?,?,?)',
            (uid, 'income', base_total, category, descs[0], f"{y:04d}-{m:02d}-{day:02d}")
        )

# A budget for the category we're about to overspend on, so the Budgets
# page also shows a live "over budget" warning
conn.execute(
    'INSERT OR REPLACE INTO budgets (user_id,category,limit_amount,month) VALUES (?,?,?,?)',
    (uid, 'Food & Drinks', 250, '2026-09')
)
conn.execute(
    'INSERT OR REPLACE INTO budgets (user_id,category,limit_amount,month) VALUES (?,?,?,?)',
    (uid, 'Transport', 80, '2026-09')
)

conn.commit()
conn.close()

print("Demo account ready.")
print("  Login:    demo")
print("  Password: demo1234")
print(f"  {len(months)} months of transaction history seeded (June-Sept 2026)")
print("  'Food & Drinks' spikes ~45% this month -> should trigger an overspend insight")
print("  'Entertainment' drops ~45% this month -> should trigger a 'spent less' insight")
