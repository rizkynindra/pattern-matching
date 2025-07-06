from flask import Flask, render_template, request, jsonify, redirect, url_for
import requests
import os

app = Flask(__name__)

# Configure the backend API endpoint
API_URL = "http://172.28.208.6:8000/match"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        dataset = request.files.getlist("dataset")
        templates = request.files.getlist("templates")

        # Create temporary directories
        dataset_path = "./uploaded/dataset"
        template_path = "./uploaded/templates"
        os.makedirs(dataset_path, exist_ok=True)
        os.makedirs(template_path, exist_ok=True)

        ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
        app.config['UPLOAD_FOLDER'] = dataset_path

        # Save uploaded files
        for file in dataset:
            file.save(os.path.join(dataset_path, file.filename))
        for file in templates:
            file.save(os.path.join(template_path, file.filename))

        # Send the paths to the backend API
        payload = {"dataset_path": dataset_path, "template_path": template_path}
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                results = response.json()
                return render_template("results.html", results=results)
            else:
                return f"Error: {response.text}", response.status_code
        except requests.exceptions.RequestException as e:
            return f"Error connecting to backend API: {e}", 500

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, port=5000, host='0.0.0.0')
