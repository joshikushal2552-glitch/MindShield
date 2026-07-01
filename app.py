import os
import csv
import sys
import pickle
import sqlite3
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from train_model import train_autonomous_model

load_dotenv()
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

MODELS_DIR = 'models'
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'vectorizer.pkl')
MODEL_PATH = os.path.join(MODELS_DIR, 'dark_pattern_model.pkl')
DATASET_PATH = os.path.join('dataset', 'dark-patterns-v2.csv')
DATABASE_PATH = 'mindshield.db'

GLOBAL_VECTORIZER = None
GLOBAL_MODEL = None

CONFIDENCE_THRESHOLD = 0.45  # Only flag patterns above this confidence


def init_db():
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                google_id TEXT UNIQUE,
                email TEXT,
                name TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS site_time (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                domain TEXT,
                seconds_spent INTEGER DEFAULT 0,
                UNIQUE(user_id, domain),
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dark_pattern_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                domain TEXT,
                category TEXT,
                count INTEGER DEFAULT 0,
                UNIQUE(user_id, domain, category),
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        ''')
        conn.commit()
        conn.close()
        print("[+] SQLite analytical database layers initialized successfully.")
    except Exception as e:
        print(f"[-] Database structural mapping error: {e}", file=sys.stderr)


@app.route('/api/auth/google', methods=['POST'])
def auth_google():
    try:
        data = request.get_json()
        google_id = data.get('googleId')
        email = data.get('email')
        name = data.get('name')
        
        if not google_id or not email:
            return jsonify({"success": False, "error": "Missing essential profile parameters."}), 400
            
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Check if this user profile already exists in mindshield.db
        cursor.execute("SELECT id, name, email FROM users WHERE google_id = ?", (google_id,))
        user = cursor.fetchone()
        
        if user:
            user_id, db_name, db_email = user
        else:
            # Drop a new user entry record into the database sequence context
            cursor.execute(
                "INSERT INTO users (google_id, email, name) VALUES (?, ?, ?)",
                (google_id, email, name)
            )
            conn.commit()
            user_id = cursor.lastrowid
            db_name = name
            db_email = email
            
        conn.close()
        
        return jsonify({
            "success": True,
            "userId": user_id,
            "name": db_name,
            "email": db_email
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

def log_dark_pattern_encounter(user_id, domain, category, increment=1):
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO dark_pattern_stats (user_id, domain, category, count)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, domain, category) DO UPDATE SET count = count + ?
        ''', (user_id, domain, category, increment, increment))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[-] Error writing telemetry statistics: {e}", file=sys.stderr)


def log_site_time(user_id, domain, seconds):
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO site_time (user_id, domain, seconds_spent)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id, domain) DO UPDATE SET seconds_spent = seconds_spent + ?
        ''', (user_id, domain, seconds, seconds))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[-] Error recording session time log: {e}", file=sys.stderr)


def load_pipeline():
    global GLOBAL_VECTORIZER, GLOBAL_MODEL
    if os.path.exists(VECTORIZER_PATH) and os.path.exists(MODEL_PATH):
        try:
            with open(VECTORIZER_PATH, 'rb') as f:
                GLOBAL_VECTORIZER = pickle.load(f)
            with open(MODEL_PATH, 'rb') as f:
                GLOBAL_MODEL = pickle.load(f)
            print("[+] ML model loaded successfully.")
            return True
        except Exception as e:
            print(f"[-] Error loading model: {e}", file=sys.stderr)
            return False
    else:
        print("[!] Model files not found. Training from dataset...")
        if train_autonomous_model():
            return load_pipeline()
        return False


@app.route('/', methods=['GET'])
def index():
    ready = GLOBAL_VECTORIZER is not None and GLOBAL_MODEL is not None
    return jsonify({
        "app": "MindShield ML Server",
        "status": "online",
        "model_state": "READY" if ready else "UNINITIALIZED",
        "endpoints": ["/api/health", "/api/classify", "/api/report-pattern", "/api/auth/google", "/api/stats/time", "/api/stats/interaction", "/api/stats/cumulative"]
    })


@app.route('/api/health', methods=['GET'])
def health():
    ready = GLOBAL_VECTORIZER is not None and GLOBAL_MODEL is not None
    return jsonify({"status": "online", "model_state": "READY" if ready else "UNINITIALIZED"})


@app.route('/api/classify', methods=['POST'])
def classify():
    if not GLOBAL_VECTORIZER or not GLOBAL_MODEL:
        return jsonify({"error": "Model not loaded."}), 503

    data = request.get_json()
    if not data or 'elements' not in data:
        return jsonify({"error": "Missing 'elements' in request body."}), 400

    try:
        elements = data['elements']
        user_id = data.get('userId')
        domain = data.get('domain', 'unknown')
        results = []

        valid_items = []
        valid_texts = []
        for el in elements:
            text = el.get('text', '').strip()
            if text and len(text) >= 5:
                valid_items.append(el)
                valid_texts.append(text)

        if not valid_texts:
            return jsonify({"results": []})

        features = GLOBAL_VECTORIZER.transform(valid_texts)
        predictions = GLOBAL_MODEL.predict(features)
        probabilities = GLOBAL_MODEL.predict_proba(features)
        class_labels = GLOBAL_MODEL.classes_

        for el, pred, probs in zip(valid_items, predictions, probabilities):
            if pred == "Not Dark Pattern":
                continue
            pred_idx = list(class_labels).index(pred)
            confidence = float(probs[pred_idx])
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            results.append({
                "id": el.get('id', 'unknown'),
                "text": el.get('text', '').strip(),
                "category": pred,
                "confidence": confidence,
                "selector": el.get('selector', '')
            })

            if user_id and domain and domain != 'unknown':
                log_dark_pattern_encounter(user_id, domain, pred, increment=1)

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/report-pattern', methods=['POST'])
def report_pattern():
    data = request.get_json()
    if not data or 'text' not in data or 'category' not in data:
        return jsonify({"error": "Missing text or category parameters."}), 400

    text = data['text'].strip()
    category = data['category'].strip()

    try:
        if os.path.exists(DATASET_PATH):
            with open(DATASET_PATH, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([text, category])
        else:
            os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
            with open(DATASET_PATH, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Pattern String', 'Pattern Category'])
                writer.writerow([text, category])

        if train_autonomous_model():
            load_pipeline()
            return jsonify({"message": "Pattern reported and model retrained successfully."})
        else:
            return jsonify({"error": "Model retraining execution failure."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/analyze-terms', methods=['POST'])
def analyze_terms():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' in request body."}), 400

    filtered_text = data['text']
    try:
        from groq import Groq
        client = Groq()

        system_instruction = (
            "You are an objective data privacy auditor. Analyze the filtered legal clauses provided. "
            "Generate an exceptionally accessible, neutral summary focusing exclusively on content implications. "
            "Ignore all structural cookie dialog statements or click-to-wrap boilerplate agreement filler text. "
            "Break down exactly: 1) How data is tracked/compiled, 2) Third-party sharing policies, "
            "and 3) Any direct consumer risks or hidden subscription liabilities. "
            "Format your response using clear Markdown headers (###) and clean bullet points (*). "
            "Keep definitions extremely concise and scannable."
        )

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Analyze these policy clauses:\n\n{filtered_text}"}
            ],
            temperature=0.15,
            max_tokens=800
        )

        model_response = completion.choices[0].message.content
        return jsonify({"summary": model_response})

    except Exception as err:
        print(f"[!] Policy analysis failure: {err}", file=sys.stderr)
        return jsonify({"error": f"Groq engine integration failure: {str(err)}"}), 500


@app.route('/api/auth/google', methods=['POST'])
def google_auth():
    data = request.get_json()
    if not data or 'googleId' not in data or 'email' not in data:
        return jsonify({"error": "Missing googleId or email mapping inputs."}), 400

    google_id = data['googleId']
    email = data['email']
    name = data.get('name', email.split('@')[0])

    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT id, name, email FROM users WHERE google_id = ?', (google_id,))
        row = cursor.fetchone()
        if row:
            user_id = row[0]
            name = row[1]
            email = row[2]
        else:
            cursor.execute('INSERT INTO users (google_id, email, name) VALUES (?, ?, ?)', (google_id, email, name))
            user_id = cursor.lastrowid
            conn.commit()
        conn.close()
        return jsonify({"success": True, "userId": user_id, "email": email, "name": name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/stats/time', methods=['POST'])
def add_site_time():
    data = request.get_json()
    user_id = data.get('userId')
    domain = data.get('domain')
    seconds = data.get('seconds', 0)
    if not user_id or not domain:
        return jsonify({"error": "Missing required user mapping profiles."}), 400
    log_site_time(user_id, domain, seconds)
    return jsonify({"success": True})


@app.route('/api/stats/interaction', methods=['POST'])
def add_interaction():
    data = request.get_json()
    user_id = data.get('userId')
    domain = data.get('domain')
    category = data.get('category')
    if not user_id or not domain or not category:
        return jsonify({"error": "Missing functional interaction parameters."}), 400
    log_dark_pattern_encounter(user_id, domain, category, increment=1)
    return jsonify({"success": True})


@app.route('/api/stats/cumulative', methods=['GET'])
def get_cumulative_stats():
    user_id = request.args.get('userId')
    if not user_id:
        return jsonify({"error": "Missing required tracking userId argument."}), 400

    try:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
            SELECT category, SUM(count) as total_count 
            FROM dark_pattern_stats 
            WHERE user_id = ? 
            GROUP BY category
            ORDER BY total_count DESC
        ''', (user_id,))
        pattern_counts = [dict(row) for row in cursor.fetchall()]

        cursor.execute('''
            SELECT domain, seconds_spent 
            FROM site_time 
            WHERE user_id = ? 
            ORDER BY seconds_spent DESC
        ''', (user_id,))
        site_times = [dict(row) for row in cursor.fetchall()]

        cursor.execute('''
            SELECT d.domain, d.category, d.count as max_count
            FROM dark_pattern_stats d
            INNER JOIN (
                SELECT domain, MAX(count) as max_c
                FROM dark_pattern_stats
                WHERE user_id = ?
                GROUP BY domain
            ) m ON d.domain = m.domain AND d.count = m.max_c
            WHERE d.user_id = ?
            GROUP BY d.domain
        ''', (user_id, user_id))
        prevalent_patterns = [dict(row) for row in cursor.fetchall()]

        cursor.execute('''
            SELECT domain, seconds_spent
            FROM site_time
            WHERE user_id = ?
            ORDER BY seconds_spent DESC
            LIMIT 5
        ''', (user_id,))
        top_websites = [dict(row) for row in cursor.fetchall()]

        conn.close()

        return jsonify({
            "pattern_counts": pattern_counts,
            "site_times": site_times,
            "prevalent_patterns": prevalent_patterns,
            "top_websites": top_websites
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    init_db()
    load_pipeline()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)