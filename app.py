import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
from flask import Flask, request, render_template, redirect, url_for, session
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.secret_key = "secret123"

# Upload Folder
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("🔹 Loading pretrained MobileNetV2 model (ImageNet)...")
model = MobileNetV2(weights='imagenet')
print("✅ Model loaded successfully!")

# --------------------------
# LOGIN PAGE
# --------------------------
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username == "Ganesh" and password == "1234":
            session["username"] = username
            return redirect(url_for("dashboard"))
        else:
            return render_template("login.html", error=True)

    return render_template("login.html")


# --------------------------
# DASHBOARD PAGE
# --------------------------
@app.route("/dashboard")
def dashboard():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html", username=session["username"])


# --------------------------
# UPLOAD + IMAGE ANALYSIS
# --------------------------
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if "username" not in session:
        return redirect(url_for("login"))

    label = None
    prob = None
    filename = None

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            img = image.load_img(filepath, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            preds = model.predict(x)
            decoded = decode_predictions(preds, top=1)[0][0]
            label, prob = decoded[1], round(decoded[2] * 100, 2)

            return render_template("upload.html", filename=file.filename, label=label, prob=prob)

    return render_template("upload.html", filename=None)


# --------------------------
# ABOUT PAGE
# --------------------------
@app.route("/about")
def about():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("about.html")


# --------------------------
# LOGOUT PAGE
# --------------------------
@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("login"))


# --------------------------
# RUN
# --------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
