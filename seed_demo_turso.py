"""
seed_demo_turso.py
───────────────────
Same idea as seed_demo.py, but writes directly to your LIVE Turso database
instead of the local data.db file — so the demo account is ready to show
your supervisor/examiner on the real deployed app, not just on your laptop.

It only ever touches a user called "demo": it deletes and recreates that
one account every time you run it, so it's safe to re-run as many times as
you like (e.g. the morning of your viva, so "this month" is always right).

WHY THIS SCRIPT EXISTS
-----------------------
The spending-insight alert (FR7) only fires once a category has at least
3 *prior* months of history, and it compares "this month" against that
history — where "this month" is whatever month it actually is when
someone opens the app (see app.py's api_insights(), which defaults to
date.today()). A fresh account has no history, so there is nothing to
demo. This script backfills 3 months of realistic history *ending in the
real current month*, calculated from today's date rather than a fixed
date, so it stays correct no matter when your viva actually happens.

SETUP (one-off)
----------------
1. pip install turso-serverless
2. Get your database URL and an auth token from the Turso dashboard:
   https://app.turso.tech -> your database -> "Connect" tab.
   (These are the exact same two values Render already has set as
   TURSO_DATABASE_URL / TURSO_AUTH_TOKEN for your deployed app.)
3. Run, e.g.:
   TURSO_DATABASE_URL="libsql://your-db-xxxx.turso.io" \
   TURSO_AUTH_TOKEN="ey..." \
   python seed_demo_turso.py

Then log in to your LIVE app at:
    username: demo
    password: demo1234
"""
import os
import hashlib
import random
from datetime import date

import turso_serverless

TURSO_DATABASE_URL = os.environ.get('TURSO_DATABASE_URL')
TURSO_AUTH_TOKEN = os.environ.get('TURSO_AUTH_TOKEN', '')

if not TURSO_DATABASE_URL:
    raise SystemExit(
        "TURSO_DATABASE_URL is not set.\n"
        "Set it (and TURSO_AUTH_TOKEN) to the same values your deployed "
        "app uses — see the Turso dashboard's 'Connect' tab, or your "
        "Render environment variables. Then run this script again."
    )


def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()


def add_months(year, month, delta):
    """Shift (year, month) by `delta` months, wrapping year correctly."""
    total = year * 12 + (month - 1) + delta
    return total // 12, total % 12 + 1


def month_str(year, month):
    return f"{year:04d}-{month:02d}"


conn = turso_serverless.connect(TURSO_DATABASE_URL, auth_token=TURSO_AUTH_TOKEN)
conn.row_factory = turso_serverless.Row

# Make sure tables exist (same schema as app.py's init_db) — harmless no-op
# against a database that already has them.
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

# 3 prior months + the REAL current month, worked out from today's date
# rather than hard-coded, so this stays correct whenever you run it.
today = date.today()
months = [add_months(today.year, today.month, delta) for delta in (-3, -2, -1, 0)]
current_month_tuple = months[-1]

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
    'Part-time Job': (650, ['bar shift pay', 'retail shift pay']),
    'Student Loan':  (450, ['su payment']),
}


def spread_amount(total, n):
    """Split a monthly total into n roughly-even-but-varied transaction amounts."""
    parts = [random.uniform(0.6, 1.4) for _ in range(n)]
    s = sum(parts)
    return [round(total * p / s, 2) for p in parts]


for (y, m) in months:
    is_current = (y, m) == current_month_tuple
    # Don't backdate into days that haven't happened yet this month
    max_day = today.day if is_current else 27
    max_day = max(max_day, 2)
    for category, (base_total, descs) in EXPENSE_PLAN.items():
        total = base_total
        if is_current and category == 'Food & Drinks':
            total = base_total * 1.45  # deliberate ~45% spike -> triggers an alert
        if is_current and category == 'Entertainment':
            total = base_total * 0.55  # deliberate underspend -> triggers a "spent less" alert
        n_txns = max(1, len(descs))
        amounts = spread_amount(total, n_txns)
        for i, amt in enumerate(amounts):
            day = random.randint(1, max_day)
            desc = descs[i % len(descs)] if descs else category.lower()
            conn.execute(
                'INSERT INTO transactions (user_id,type,amount,category,description,date) VALUES (?,?,?,?,?,?)',
                (uid, 'expense', amt, category, desc, f"{y:04d}-{m:02d}-{day:02d}")
            )
    for category, (base_total, descs) in INCOME_PLAN.items():
        day = random.randint(1, min(5, max_day))
        conn.execute(
            'INSERT INTO transactions (user_id,type,amount,category,description,date) VALUES (?,?,?,?,?,?)',
            (uid, 'income', base_total, category, descs[0], f"{y:04d}-{m:02d}-{day:02d}")
        )

# A budget for the current month, so the Budgets page also shows a live
# "over budget" warning alongside the dashboard insight.
cm_str = month_str(*current_month_tuple)
conn.execute(
    'INSERT OR REPLACE INTO budgets (user_id,category,limit_amount,month) VALUES (?,?,?,?)',
    (uid, 'Food & Drinks', 250, cm_str)
)
conn.execute(
    'INSERT OR REPLACE INTO budgets (user_id,category,limit_amount,month) VALUES (?,?,?,?)',
    (uid, 'Transport', 80, cm_str)
)

print("Demo account ready on your LIVE Turso database.")
print("  Login:    demo")
print("  Password: demo1234")
print(f"  Seeded months: {', '.join(month_str(y, m) for y, m in months)}")
print(f"  Current month used by the app's insights check: {cm_str}")
print("  'Food & Drinks' spikes ~45% this month -> should trigger an overspend insight")
print("  'Entertainment' drops ~45% this month -> should trigger a 'spent less' insight")
print()
print("Log in to your deployed app now and open the Dashboard / Budgets / Analytics")
print("pages to show the examiner the live ML features working on real data.")
