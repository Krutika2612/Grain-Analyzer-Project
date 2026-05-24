from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from grain_processor import process_file

app = Flask(__name__)

# -------------------------------
# FOLDER CONFIGURATION
# -------------------------------
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


# -------------------------------
# HOME PAGE
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -------------------------------
# PREDICT / PROCESS
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]

    if file.filename == "":
        return "No file selected"

    # Save uploaded file
    filename = secure_filename(file.filename)
    upload_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(upload_path)

    # Process file
    result = process_file(upload_path, app.config["RESULT_FOLDER"])

    # -------------------------------
    # SINGLE IMAGE RESULT
    # -------------------------------
    if isinstance(result, dict):
        return render_template(
            "index.html",
            original_img=filename,
            result_img=result["image"],
            total_count=result["count"],
            avg_area=result["avg_area"],
            csv_file=result["csv"],
            hist_img=result["hist"]
        )

    # -------------------------------
    # ZIP FILE RESULT (MULTIPLE)
    # -------------------------------
    else:
        return render_template(
            "index.html",
            zip_results=result
        )


# -------------------------------
# RUN SERVER
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)