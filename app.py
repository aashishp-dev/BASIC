from flask import Flask, send_from_directory, request, jsonify
import os
import base64
import io
from PIL import Image
from groq import Groq
import fitz

from agents import (
    run_doctoraai,
    parse_research,
    parse_risk_level,
    get_specialist,
    ask_agent,
    parse_lab_analysis
)

app = Flask(__name__)
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


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
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        return "".join(page.get_text() for page in pdf)
    elif filename.endswith((".png", ".jpg", ".jpeg", ".webp", ".jfif", ".bmp", ".tiff")):
        return ocr_image_with_groq(file)
    return None


@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


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


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
