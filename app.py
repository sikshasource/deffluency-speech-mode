import os
import json
from types import SimpleNamespace

from flask import Flask, render_template, request, flash
from werkzeug.utils import secure_filename

from aijob_core import analyze_application, extract_resume_text

app = Flask(__name__)
app.secret_key = "change-this-secret-key"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    ui_result = None
    result_json = None

    if request.method == "POST":
        job_description = request.form.get("job_description", "").strip()
        extra_context = request.form.get("extra_context", "").strip()
        resume_file = request.files.get("resume_file")

        if not job_description:
            flash("Please paste a job description.")
            return render_template("index.html", result=None, result_json=None)

        resume_text = ""
        if resume_file and resume_file.filename:
            filename = secure_filename(resume_file.filename)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            resume_file.save(save_path)
            try:
                resume_text = extract_resume_text(save_path)
            except Exception as e:
                flash(f"Could not read resume file: {e}")
                resume_text = ""
        else:
            flash("No resume file selected. Analysis will only use the job description.")

        raw_result = analyze_application(job_description, resume_text, extra_context or None)

        # For UI: convert dict to SimpleNamespace so we can use dot notation in Jinja
        ui_result = json.loads(
            json.dumps(raw_result, ensure_ascii=False),
            object_hook=lambda d: SimpleNamespace(**d)
        )
        # For JSON view: pretty-printed string
        result_json = json.dumps(raw_result, indent=2, ensure_ascii=False)

    return render_template("index.html", result=ui_result, result_json=result_json)

if __name__ == "__main__":
    app.run(debug=True)