import os
from flask import Flask, request, jsonify, send_from_directory, render_template
from werkzeug.utils import secure_filename

from interface_v2 import generate_song, VALID_STYLES, OUTPUT_DIR


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR_ABS = os.path.join(BASE_DIR, OUTPUT_DIR)

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/input")
def input_page():

    return render_template("input.html")


@app.route("/api/styles")
def list_styles():
    return jsonify({"styles": VALID_STYLES})


@app.route("/api/generate", methods=["POST"])
def api_generate():

    if "file" not in request.files:
        return jsonify({"success": False, "message": "No file part named 'file' in request."}), 400

    file = request.files["file"]
    style = request.form.get("style", VALID_STYLES[0])

    if file.filename == "":
        return jsonify({"success": False, "message": "Please choose a MIDI file."}), 400

    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in {".mid", ".midi"}:
        return jsonify({"success": False, "message": "Only .mid / .midi files are supported."}), 400

    save_path = os.path.join(UPLOAD_DIR, filename)
    file.save(save_path)


    result = generate_song(save_path, style)

    if not result.get("success"):

        return jsonify(result), 500

    midi_path = result.get("midi_path")
    if midi_path:
        midi_name = os.path.basename(midi_path)
        download_url = f"/download/{midi_name}"
    else:
        download_url = None

    response = {
        "success": True,
        "message": result.get("message", "Generation completed."),
        "chord_preview": result.get("chord_preview", []),
        "file_url": download_url,
    }
    return jsonify(response)


@app.route("/download/<path:filename>")
def download_file(filename: str):

    return send_from_directory(OUTPUT_DIR_ABS, filename, as_attachment=True)


if __name__ == "__main__":

    app.run(debug=True)
