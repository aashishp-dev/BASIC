from flask import Flask, send_from_directory, request, jsonify, render_template
import os
import base64
import io
from PIL import Image
from groq import Groq
import pymupdf
from functools import wraps

from agents import (
    run_doctoraai,
    parse_research,
    parse_risk_level,
    get_specialist,
    ask_agent,
    parse_lab_analysis
)

# Initialize Firebase Admin SDK
try:
    import firebase_admin
    from firebase_admin import credentials, firestore, auth as firebase_auth
    import json

    # Use environment variables for Firebase credentials (more secure)
    firebase_creds = os.environ.get("FIREBASE_SERVICE_ACCOUNT")

    if firebase_creds:
        # Parse JSON from environment variable
        cred_dict = json.loads(firebase_creds)
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        FIREBASE_ENABLED = True
        print("✓ Firebase initialized successfully")
    else:
        FIREBASE_ENABLED = False
        db = None
        print("WARNING: FIREBASE_SERVICE_ACCOUNT not found in environment. Auth features disabled.")
except ImportError:
    FIREBASE_ENABLED = False
    db = None
    print("WARNING: firebase-admin not installed. Auth features disabled.")
except Exception as e:
    FIREBASE_ENABLED = False
    db = None
    print(f"WARNING: Firebase initialization failed: {str(e)}")

app = Flask(__name__)
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# Auth decorator
def require_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not FIREBASE_ENABLED:
            return jsonify({'error': 'Firebase not configured'}), 503

        token = request.json.get('token')
        if not token:
            return jsonify({'error': 'No token provided'}), 401

        try:
            decoded = firebase_auth.verify_id_token(token)
            request.userId = decoded['uid']
            return f(*args, **kwargs)
        except Exception as e:
            return jsonify({'error': 'Invalid token', 'detail': str(e)}), 401

    return decorated_function


def ocr_image_with_groq(file):
    image = Image.open(file)
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    image.thumbnail((1200, 1200))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    image_bytes = buffer.getvalue()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    completion = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract all text from this medical document image exactly as it appears. Just output the raw extracted text, nothing else."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }
        ],
        temperature=0,
        max_completion_tokens=1024,
    )
    return completion.choices[0].message.content


def extract_text(file):
    filename = file.filename.lower()
    if filename.endswith(".pdf"):
        pdf = pymupdf.open(stream=file.read(), filetype="pdf")
        return "".join(page.get_text() for page in pdf)
    elif filename.endswith((".png", ".jpg", ".jpeg", ".webp", ".jfif", ".bmp", ".tiff")):
        return ocr_image_with_groq(file)
    return None


@app.route("/")
def index():
    return render_template(
        "index.html",
        FIREBASE_API_KEY=os.environ.get("FIREBASE_API_KEY", ""),
        FIREBASE_AUTH_DOMAIN=os.environ.get("FIREBASE_AUTH_DOMAIN", ""),
        FIREBASE_PROJECT_ID=os.environ.get("FIREBASE_PROJECT_ID", ""),
        FIREBASE_STORAGE_BUCKET=os.environ.get("FIREBASE_STORAGE_BUCKET", ""),
        FIREBASE_MESSAGING_SENDER_ID=os.environ.get("FIREBASE_MESSAGING_SENDER_ID", ""),
        FIREBASE_APP_ID=os.environ.get("FIREBASE_APP_ID", "")
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    user_query = data.get("query", "")
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    diagnosis, treatment, research, final = run_doctoraai(user_query)
    papers = parse_research(research)
    risk, clean_final = parse_risk_level(final)
    specialist = get_specialist(user_query)

    return jsonify({
        "final": clean_final,
        "risk": risk,
        "papers": papers,
        "specialist": specialist
    })


@app.route("/upload-lab", methods=["POST"])
def upload_lab():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    text = extract_text(file)

    if text is None:
        return jsonify({"error": "Unsupported file type"}), 400

    analysis = ask_agent(
        """
You are a medical lab report analyzer.

Analyze the report and respond ONLY in this exact format:

SUMMARY: [2-3 sentence plain-language summary of the report]

ABNORMAL_COUNT: [number of abnormal values found]

VALUE1_NAME: [test name]
VALUE1_RESULT: [the patient's value]
VALUE1_RANGE: [normal reference range]
VALUE1_STATUS: [High/Low/Critical]

VALUE2_NAME: [test name]
VALUE2_RESULT: [the patient's value]
VALUE2_RANGE: [normal reference range]
VALUE2_STATUS: [High/Low/Critical]

(repeat VALUE3, VALUE4 etc. for each abnormal value found)

CONCERNS: [2-3 sentence explanation of what these abnormalities might indicate]

SPECIALIST: [the single best specialist type to consult]

URGENCY: [Low/Moderate/High]

Only include abnormal values. If everything is normal, set ABNORMAL_COUNT to 0.
        """,
        text
    )

    return jsonify(parse_lab_analysis(analysis))


@app.route("/upload-prescription", methods=["POST"])
def upload_prescription():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        text = extract_text(file)

        if text is None:
            return jsonify({"error": "Unsupported file type"}), 400

        print("OCR TEXT:", text[:300])

        analysis = ask_agent(
            """
You are a prescription analyzer.

Extract all medicines from the prescription.

Return ONLY in this exact repeating format for each medicine found:

MEDICINE: [medicine name]
PURPOSE: [what it treats]
DOSE: [dosage amount]
TIMING: [e.g. 1-1-1 or 1-0-1 or once daily]
DURATION: [how many days]
WARNING: [key side effects or warnings]

Repeat the block above for each medicine. Nothing else.
            """,
            text
        )

        return jsonify({"analysis": analysis})

    except Exception as e:
        print("ERROR in upload_prescription:", str(e))
        return jsonify({"error": str(e)}), 500


# ========================================
# FIREBASE AUTHENTICATION & PROFILE ROUTES
# ========================================

@app.route("/verify-token", methods=["POST"])
def verify_token():
    """Verify Firebase ID token"""
    if not FIREBASE_ENABLED:
        return jsonify({'error': 'Firebase not configured'}), 503

    try:
        token = request.json.get('token')
        if not token:
            return jsonify({'error': 'No token provided'}), 401

        decoded = firebase_auth.verify_id_token(token)
        return jsonify({
            'success': True,
            'userId': decoded['uid'],
            'email': decoded.get('email'),
            'name': decoded.get('name')
        })
    except Exception as e:
        return jsonify({'error': 'Token verification failed', 'detail': str(e)}), 401


@app.route("/save-timeline-entry", methods=["POST"])
@require_auth
def save_timeline_entry():
    """Save a timeline entry to Firestore"""
    try:
        data = request.json
        user_id = request.userId

        entry = {
            'type': data.get('type'),
            'title': data.get('title'),
            'date': data.get('date'),
            'data': data.get('data', {})
        }

        # Save to Firestore
        entry_ref = db.collection('users').document(user_id).collection('timeline').document()
        entry_ref.set(entry)

        return jsonify({'success': True, 'entryId': entry_ref.id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/get-timeline", methods=["POST"])
@require_auth
def get_timeline():
    """Get all timeline entries for a user"""
    try:
        user_id = request.userId

        # Fetch timeline entries
        timeline_ref = db.collection('users').document(user_id).collection('timeline')
        entries = timeline_ref.order_by('date', direction=firestore.Query.DESCENDING).stream()

        timeline = []
        for entry in entries:
            entry_data = entry.to_dict()
            entry_data['id'] = entry.id
            timeline.append(entry_data)

        return jsonify(timeline)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/save-metrics", methods=["POST"])
@require_auth
def save_metrics():
    """Save health metrics (weight, BP, blood sugar)"""
    try:
        data = request.json
        user_id = request.userId

        metrics_data = {}

        if data.get('weight'):
            metrics_data['weight'] = {
                'value': data['weight'],
                'date': firestore.SERVER_TIMESTAMP
            }

        if data.get('bloodPressure'):
            metrics_data['bloodPressure'] = {
                'value': data['bloodPressure'],
                'date': firestore.SERVER_TIMESTAMP
            }

        if data.get('bloodSugar'):
            metrics_data['bloodSugar'] = {
                'value': data['bloodSugar'],
                'date': firestore.SERVER_TIMESTAMP
            }

        # Save to Firestore
        metrics_ref = db.collection('users').document(user_id).collection('metrics').document('latest')
        metrics_ref.set(metrics_data, merge=True)

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/get-metrics", methods=["POST"])
@require_auth
def get_metrics():
    """Get latest health metrics for a user"""
    try:
        user_id = request.userId

        # Fetch metrics
        metrics_ref = db.collection('users').document(user_id).collection('metrics').document('latest')
        metrics_doc = metrics_ref.get()

        if metrics_doc.exists:
            return jsonify(metrics_doc.to_dict())
        else:
            return jsonify({})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
