from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import sqlite3, os, hashlib, re, socket
from datetime import datetime, date, timedelta
from functools import wraps
from ai_insights import suggest_category, get_insights

# Stricter than a bare "has an @ and a dot" check: rejects addresses with
# no real top-level domain, leading/trailing dots or hyphens, or
# consecutive dots (all of which the previous, looser pattern accepted as
# "valid" even though they're not deliverable addresses).
EMAIL_RE = re.compile(
    r'^[A-Za-z0-9](?:[A-Za-z0-9._%+-]*[A-Za-z0-9])?'
    r'@[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?'
    r'(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)*'
    r'\.[A-Za-z]{2,}$'
)

SECURITY_QUESTIONS = [
    "What was the name of your first school?",
    "What is your mother's maiden name?",
    "What was the name of your first pet?",
    "What city were you born in?",
    "What is your favourite teacher's name?",
]

def validate_registration(data):
    username = (data.get('username') or '').strip()
    email = (data.get('email') or '').strip()
    password = data.get('password') or ''
    security_question = (data.get('security_question') or '').strip()
    security_answer = (data.get('security_answer') or '').strip()

    if not username:
        return 'Username is required'
    if len(email) > 254 or not EMAIL_RE.match(email):
        return 'Enter a valid email address'
    if len(password) < 6:
        return 'Password must be at least 6 characters'
    if not re.search(r'[A-Za-z]', password) or not re.search(r'[0-9]', password):
        return 'Password must contain both letters and numbers'
    if security_question not in SECURITY_QUESTIONS:
        return 'Choose a security question'
    if not security_answer:
        return 'Enter an answer to your security question'
    return None

app = Flask(__name__)
# In production, set a SECRET_KEY environment variable (e.g. in Render's
# dashboard). The hardcoded fallback is only for local development, so
# the real key is never committed to a public GitHub repo.
app.secret_key = os.environ.get('SECRET_KEY', 'expense_tracker_secret_2026_dev_only')
# Log a user out after 60 minutes of inactivity. Flask refreshes the
# session's expiry on every request by default (SESSION_REFRESH_EACH_REQUEST),
# so this is a sliding/inactivity timeout, not a fixed time-since-login one:
# as long as the user keeps browsing, the clock keeps resetting.
app.permanent_session_lifetime = timedelta(minutes=60)
DB = os.path.join(os.path.dirname(__file__), 'data.db')

# ─── Database ────────────────────────────────────────────────────────────────
# Render's free tier has no persistent disk: a local SQLite file like `data.db`
# gets silently wiped on every restart, redeploy, or idle-spindown, which used
# to delete every registered user within about a day. Turso (hosted libSQL) is
# used instead whenever TURSO_DATABASE_URL is set (i.e. in production on
# Render); this falls back to the same local-file sqlite3 behaviour as before
# when it isn't set (i.e. local development), so nothing about the SQL itself
# had to change.
#
# This uses the `turso-serverless` package (Turso's own currently-recommended
# client for a remote database), not `libsql-experimental`: that package
# talks a custom binary protocol that, in testing, hung indefinitely when
# deployed on Render (crashing the whole worker rather than failing fast).
# turso-serverless instead speaks plain HTTPS, which is both more reliable
# through hosting-platform networking and already shaped like a sqlite3
# connection (row['column'], dict(row), IntegrityError, `with conn:`), so no
# compatibility wrapper is needed here.
TURSO_DATABASE_URL = os.environ.get('TURSO_DATABASE_URL')
TURSO_AUTH_TOKEN = os.environ.get('TURSO_AUTH_TOKEN', '')

if TURSO_DATABASE_URL:
    import turso_serverless
    # The underlying HTTP calls don't pass an explicit timeout, which would
    # otherwise let a stuck request hang the whole worker indefinitely; this
    # caps every socket operation app-wide instead.
    socket.setdefaulttimeout(20)
    IntegrityError = turso_serverless.IntegrityError
else:
    IntegrityError = sqlite3.IntegrityError

def get_db():
    if TURSO_DATABASE_URL:
        conn = turso_serverless.connect(TURSO_DATABASE_URL, auth_token=TURSO_AUTH_TOKEN)
        conn.row_factory = turso_serverless.Row
        return conn
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn

def _add_column_if_missing(db, alter_sql):
    """ALTER TABLE ... ADD COLUMN, ignoring the error raised when the
    column already exists. Used to migrate databases created before the
    security-question columns existed, without touching existing data."""
    try:
        db.execute(alter_sql)
    except Exception:
        pass

def init_db():
    with get_db() as db:
        db.executescript('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                currency TEXT DEFAULT 'DKK',
                security_question TEXT,
                security_answer TEXT,
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
    with get_db() as db:
        # Safety net for databases created before security questions existed
        # (CREATE TABLE IF NOT EXISTS above won't add columns to a table that
        # already exists).
        _add_column_if_missing(db, 'ALTER TABLE users ADD COLUMN security_question TEXT')
        _add_column_if_missing(db, 'ALTER TABLE users ADD COLUMN security_answer TEXT')
        # The app only ever displays kr. (DKK) -- there's no currency-picker
        # UI, so 'GBP' on existing accounts is just the old hardcoded
        # default from before the switch to DKK, never an actual user choice.
        db.execute("UPDATE users SET currency='DKK' WHERE currency='GBP' OR currency IS NULL")

init_db()

INCOME_CATEGORIES  = ['Salary','Part-time Job','Freelance','Student Loan','Family Support','Scholarship','Other Income']
EXPENSE_CATEGORIES = ['Food & Drinks','Transport','Rent','Utilities','Shopping','Entertainment','Health','Education','Subscriptions','Savings','Other']

def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

# ─── Auth ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        data = request.get_json()
        error = validate_registration(data)
        if error:
            return jsonify({'ok': False, 'error': error})
        try:
            with get_db() as db:
                cur = db.execute(
                    'INSERT INTO users (username,email,password,security_question,security_answer) '
                    'VALUES (?,?,?,?,?)',
                    (data['username'], data['email'].strip(), hash_pw(data['password']),
                     data['security_question'].strip(),
                     hash_pw(data['security_answer'].strip().lower()))
                )
            # Log the new user straight in, same as /login does, so the
            # frontend's redirect to /dashboard right after registering
            # actually works instead of bouncing back to the login page.
            session.permanent = True
            session['user_id'] = cur.lastrowid
            session['username'] = data['username']
            return jsonify({'ok': True})
        except IntegrityError:
            return jsonify({'ok': False, 'error': 'Username or email already exists'})
    return render_template('auth.html', security_questions=SECURITY_QUESTIONS)

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        with get_db() as db:
            user = db.execute('SELECT * FROM users WHERE (email=? OR username=?) AND password=?',
                              (data['login'], data['login'], hash_pw(data['password']))).fetchone()
        if user:
            session.permanent = True
            session['user_id'] = user['id']
            session['username'] = user['username']
            return jsonify({'ok': True})
        return jsonify({'ok': False, 'error': 'Invalid credentials'})
    return render_template('auth.html', security_questions=SECURITY_QUESTIONS)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/forgot-password')
def forgot_password_page():
    return render_template('forgot_password.html')

@app.route('/api/forgot-password/question', methods=['POST'])
def api_forgot_password_question():
    """Step 1 of the security-question reset: look up the account's
    security question by username or email. Deliberately returns the same
    generic error whether the account doesn't exist or has no security
    question set, rather than revealing which, to avoid confirming
    whether a given username/email is registered."""
    d = request.get_json() or {}
    login = (d.get('login') or '').strip()
    with get_db() as db:
        user = db.execute(
            'SELECT security_question FROM users WHERE email=? OR username=?',
            (login, login)
        ).fetchone()
    if not user or not user['security_question']:
        return jsonify({'ok': False, 'error': 'No account found with that username or email'})
    return jsonify({'ok': True, 'question': user['security_question']})

@app.route('/api/forgot-password/reset', methods=['POST'])
def api_forgot_password_reset():
    """Step 2: verify the security-question answer and set a new password."""
    d = request.get_json() or {}
    login = (d.get('login') or '').strip()
    answer = (d.get('answer') or '').strip().lower()
    new_password = d.get('password') or ''

    if len(new_password) < 6:
        return jsonify({'ok': False, 'error': 'Password must be at least 6 characters'})
    if not re.search(r'[A-Za-z]', new_password) or not re.search(r'[0-9]', new_password):
        return jsonify({'ok': False, 'error': 'Password must contain both letters and numbers'})

    with get_db() as db:
        user = db.execute(
            'SELECT id, security_answer FROM users WHERE email=? OR username=?',
            (login, login)
        ).fetchone()
        if not user or not user['security_answer'] or user['security_answer'] != hash_pw(answer):
            return jsonify({'ok': False, 'error': 'That answer doesn’t match our records'})
        db.execute('UPDATE users SET password=? WHERE id=?', (hash_pw(new_password), user['id']))
    return jsonify({'ok': True})

# ─── Pages ───────────────────────────────────────────────────────────────────

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', username=session['username'])

@app.route('/transactions')
@login_required
def transactions():
    return render_template('transactions.html', username=session['username'])

@app.route('/budgets')
@login_required
def budgets_page():
    return render_template('budgets.html', username=session['username'])

@app.route('/analytics')
@login_required
def analytics():
    return render_template('analytics.html', username=session['username'])

@app.route('/profile')
@login_required
def profile_page():
    uid = session['user_id']
    with get_db() as db:
        user = db.execute(
            'SELECT username, email, currency, security_question, created_at '
            'FROM users WHERE id=?', (uid,)
        ).fetchone()
    return render_template('profile.html', username=session['username'], user=user)

@app.route('/account')
@login_required
def account_page():
    return render_template('account.html', username=session['username'])

# ─── API ─────────────────────────────────────────────────────────────────────

@app.route('/api/categories', methods=['GET', 'POST', 'DELETE'])
@login_required
def api_categories():
    """Category list for the current user: the built-in defaults plus any
    categories this user has added themselves. Custom categories are
    per-user (not shared/global), matching the "allow custom categories"
    requirement (FR2) inspired by Monefy and To (2026)."""
    uid = session['user_id']

    if request.method == 'GET':
        with get_db() as db:
            custom = db.execute(
                'SELECT type, name FROM custom_categories WHERE user_id=? ORDER BY name', (uid,)
            ).fetchall()
        income = list(INCOME_CATEGORIES) + [r['name'] for r in custom if r['type'] == 'income']
        expense = list(EXPENSE_CATEGORIES) + [r['name'] for r in custom if r['type'] == 'expense']
        return jsonify({'income': income, 'expense': expense})

    if request.method == 'POST':
        d = request.get_json() or {}
        ttype = d.get('type')
        name = (d.get('name') or '').strip()
        if ttype not in ('income', 'expense'):
            return jsonify({'ok': False, 'error': 'Invalid category type'}), 400
        if not name or len(name) > 40:
            return jsonify({'ok': False, 'error': 'Enter a category name (up to 40 characters)'}), 400
        defaults = (INCOME_CATEGORIES if ttype == 'income' else EXPENSE_CATEGORIES)
        with get_db() as db:
            existing_custom = [r['name'] for r in db.execute(
                'SELECT name FROM custom_categories WHERE user_id=? AND type=?', (uid, ttype)
            ).fetchall()]
        all_existing_lower = [c.lower() for c in defaults] + [c.lower() for c in existing_custom]
        if name.lower() in all_existing_lower:
            return jsonify({'ok': False, 'error': 'That category already exists'}), 400
        try:
            with get_db() as db:
                db.execute(
                    'INSERT INTO custom_categories (user_id, type, name) VALUES (?,?,?)',
                    (uid, ttype, name)
                )
            return jsonify({'ok': True, 'name': name})
        except IntegrityError:
            return jsonify({'ok': False, 'error': 'That category already exists'}), 400

    if request.method == 'DELETE':
        d = request.get_json() or {}
        ttype = d.get('type')
        name = (d.get('name') or '').strip()
        with get_db() as db:
            db.execute(
                'DELETE FROM custom_categories WHERE user_id=? AND type=? AND name=?',
                (uid, ttype, name)
            )
        return jsonify({'ok': True})

@app.route('/api/transactions', methods=['GET','POST','PUT','DELETE'])
@login_required
def api_transactions():
    uid = session['user_id']
    if request.method == 'GET':
        month  = request.args.get('month', date.today().strftime('%Y-%m'))
        ttype  = request.args.get('type', '')
        cat    = request.args.get('category', '')
        q      = request.args.get('q', '').strip()
        limit  = int(request.args.get('limit', 200))
        query  = "SELECT * FROM transactions WHERE user_id=? AND strftime('%Y-%m',date)=?"
        params = [uid, month]
        if ttype: query += ' AND type=?'; params.append(ttype)
        if cat:   query += ' AND category=?'; params.append(cat)
        if q:     query += ' AND (description LIKE ? OR category LIKE ?)'; params.extend(['%' + q + '%', '%' + q + '%'])
        query += ' ORDER BY date DESC, created_at DESC LIMIT ?'
        params.append(limit)
        with get_db() as db:
            rows = [dict(r) for r in db.execute(query, params)]
        return jsonify(rows)

    if request.method == 'POST':
        d = request.get_json()
        with get_db() as db:
            cur = db.execute(
                'INSERT INTO transactions (user_id,type,amount,category,description,date) VALUES (?,?,?,?,?,?)',
                (uid, d['type'], float(d['amount']), d['category'], d.get('description',''), d['date'])
            )
        return jsonify({'ok': True, 'id': cur.lastrowid})

    if request.method == 'PUT':
        d = request.get_json()
        with get_db() as db:
            db.execute(
                'UPDATE transactions SET type=?,amount=?,category=?,description=?,date=? WHERE id=? AND user_id=?',
                (d['type'], float(d['amount']), d['category'], d.get('description',''), d['date'], d['id'], uid)
            )
        return jsonify({'ok': True})

    if request.method == 'DELETE':
        tid = request.get_json()['id']
        with get_db() as db:
            db.execute('DELETE FROM transactions WHERE id=? AND user_id=?', (tid, uid))
        return jsonify({'ok': True})

@app.route('/api/summary')
@login_required
def api_summary():
    uid   = session['user_id']
    month = request.args.get('month', date.today().strftime('%Y-%m'))
    with get_db() as db:
        rows = db.execute(
            "SELECT type, SUM(amount) as total FROM transactions WHERE user_id=? AND strftime('%Y-%m',date)=? GROUP BY type",
            (uid, month)).fetchall()
        cat_rows = db.execute(
            "SELECT category, SUM(amount) as total FROM transactions WHERE user_id=? AND type='expense' AND strftime('%Y-%m',date)=? GROUP BY category ORDER BY total DESC",
            (uid, month)).fetchall()
        income_cats = db.execute(
            "SELECT category, SUM(amount) as total FROM transactions WHERE user_id=? AND type='income' AND strftime('%Y-%m',date)=? GROUP BY category ORDER BY total DESC",
            (uid, month)).fetchall()
        # last 6 months trend
        trend = db.execute(
            "SELECT strftime('%Y-%m',date) as month, type, SUM(amount) as total FROM transactions WHERE user_id=? GROUP BY month,type ORDER BY month DESC LIMIT 12",
            (uid,)).fetchall()
        balance = db.execute("SELECT SUM(CASE WHEN type='income' THEN amount ELSE -amount END) as bal FROM transactions WHERE user_id=?", (uid,)).fetchone()
    totals = {r['type']: r['total'] for r in rows}
    return jsonify({
        'income':       totals.get('income', 0),
        'expenses':     totals.get('expense', 0),
        'balance_all':  balance['bal'] or 0,
        'expense_cats': [dict(r) for r in cat_rows],
        'income_cats':  [dict(r) for r in income_cats],
        'trend':        [dict(r) for r in trend]
    })

@app.route('/api/budgets', methods=['GET','POST','DELETE'])
@login_required
def api_budgets():
    uid   = session['user_id']
    month = request.args.get('month', date.today().strftime('%Y-%m'))
    if request.method == 'GET':
        with get_db() as db:
            budgets = db.execute('SELECT * FROM budgets WHERE user_id=? AND month=?', (uid, month)).fetchall()
            spent   = db.execute(
                "SELECT category, SUM(amount) as total FROM transactions WHERE user_id=? AND type='expense' AND strftime('%Y-%m',date)=? GROUP BY category",
                (uid, month)).fetchall()
        spent_map = {r['category']: r['total'] for r in spent}
        result = []
        for b in budgets:
            b = dict(b)
            b['spent'] = spent_map.get(b['category'], 0)
            result.append(b)
        return jsonify(result)

    if request.method == 'POST':
        d = request.get_json()
        with get_db() as db:
            db.execute(
                'INSERT OR REPLACE INTO budgets (user_id,category,limit_amount,month) VALUES (?,?,?,?)',
                (uid, d['category'], float(d['limit_amount']), d['month'])
            )
        return jsonify({'ok': True})

    if request.method == 'DELETE':
        d = request.get_json()
        with get_db() as db:
            db.execute('DELETE FROM budgets WHERE id=? AND user_id=?', (d['id'], uid))
        return jsonify({'ok': True})

@app.route('/api/suggest-category', methods=['POST'])
@login_required
def api_suggest_category():
    """AI-assisted category suggestion: predicts a likely category for a
    transaction from its free-text description (e.g. 'Netto' -> Food &
    Drinks), trained on common phrases plus the user's own past entries."""
    uid = session['user_id']
    d = request.get_json()
    with get_db() as db:
        result = suggest_category(db, uid, d.get('type', 'expense'), d.get('description', ''))
    return jsonify(result or {})

@app.route('/api/insights')
@login_required
def api_insights():
    """Statistical overspending insights: flags categories where this
    month's spend deviates significantly from the user's own recent
    average."""
    uid = session['user_id']
    month = request.args.get('month', date.today().strftime('%Y-%m'))
    with get_db() as db:
        insights = get_insights(db, uid, month)
    return jsonify(insights)

@app.route('/api/recent')
@login_required
def api_recent():
    uid = session['user_id']
    with get_db() as db:
        rows = db.execute(
            'SELECT * FROM transactions WHERE user_id=? ORDER BY date DESC, created_at DESC LIMIT 8',
            (uid,)).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route('/api/export')
@login_required
def api_export():
    """Let the user download everything the app holds about them, as one
    JSON file. Addresses the "user control of data" requirement (FR9,
    NFR4) — students should be able to see and take away their own data
    without relying on a third party. Password hashes are never included."""
    uid = session['user_id']
    with get_db() as db:
        user = db.execute(
            'SELECT username, email, currency, created_at FROM users WHERE id=?', (uid,)
        ).fetchone()
        transactions = db.execute(
            'SELECT type, amount, category, description, date, created_at FROM transactions WHERE user_id=? ORDER BY date', (uid,)
        ).fetchall()
        budgets = db.execute(
            'SELECT category, limit_amount, month FROM budgets WHERE user_id=? ORDER BY month', (uid,)
        ).fetchall()
        custom_cats = db.execute(
            'SELECT type, name FROM custom_categories WHERE user_id=?', (uid,)
        ).fetchall()

    payload = {
        'exported_at': datetime.now().isoformat(timespec='seconds'),
        'account': dict(user),
        'transactions': [dict(r) for r in transactions],
        'budgets': [dict(r) for r in budgets],
        'custom_categories': [dict(r) for r in custom_cats],
    }
    resp = jsonify(payload)
    resp.headers['Content-Disposition'] = (
        f'attachment; filename=expense-tracker-data-{date.today().isoformat()}.json'
    )
    return resp

@app.route('/api/account', methods=['DELETE'])
@login_required
def api_delete_account():
    """Permanently delete this user's account and every record tied to
    it (transactions, budgets, custom categories). Requires the user to
    re-enter their password, so the delete link alone can't be used by
    someone who has merely left a session open on a shared computer."""
    uid = session['user_id']
    d = request.get_json() or {}
    password = d.get('password') or ''
    with get_db() as db:
        user = db.execute('SELECT password FROM users WHERE id=?', (uid,)).fetchone()
        if not user or user['password'] != hash_pw(password):
            return jsonify({'ok': False, 'error': 'Incorrect password'}), 403
        db.execute('DELETE FROM transactions WHERE user_id=?', (uid,))
        db.execute('DELETE FROM budgets WHERE user_id=?', (uid,))
        db.execute('DELETE FROM custom_categories WHERE user_id=?', (uid,))
        db.execute('DELETE FROM users WHERE id=?', (uid,))
    session.clear()
    return jsonify({'ok': True})

if __name__ == '__main__':
    # Debug mode is OFF by default; set FLASK_DEBUG=1 locally if you want it.
    # Render runs the app via the Procfile (gunicorn), not this block, but
    # it's kept safe here too in case the platform runs `python app.py` directly.
    debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
