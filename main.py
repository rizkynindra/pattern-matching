from flask import Flask, request, jsonify
import os
import cv2
# import pickle
# import pandas as pd
# import numpy as np
import tempfile
# from tqdm import tqdm
from datetime import datetime
from io import BytesIO
from email_validator import validate_email, EmailNotValidError
from difflib import SequenceMatcher
import phonenumbers
from phonenumbers import carrier, geocoder
import re
import logging 
import json
import uuid
import time
import base64
import socket
import requests
from dotenv import load_dotenv
load_dotenv()  # Load from .env file

# Fungsi untuk menghasilkan UUID
def generate_transaction_id():
    return f"TR-{uuid.uuid4()}"

def generate_trace_id():
    return f"TC-{uuid.uuid4()}"

# Simple logging function
def encrypt(data: str) -> str:
    # Encrypt by encoding the data to bytes and then using base64 encoding
    encrypted_data = base64.b64encode(data.encode('utf-8'))
    return encrypted_data.decode('utf-8')  # Convert bytes back to string for display

start_time = time.time()
def save_log(
    level="INFO",
    transaction_id=None,
    service_name="pattern-matching",
    endpoint="endpoint-notset",
    protocol="REST",
    method="POST",
    execution_type="SYNC",
    function_name="/PostPatternMatching",
    caller_info="CallerInfoDefault",
    execution_time="0",
    server_ip="0.0.0.0",
    client_ip="0.0.0.0",
    event_name="Default Event",
    trace_id=None,
    prev_transaction_id="N/A",
    body="{}",
    result="Success",
    error="Error-Null",
    flag="START",
    message="Default log message"
):
    # Generate unique IDs if not provided
    transaction_id = transaction_id or generate_transaction_id()
    trace_id = trace_id or generate_trace_id()

    # client_ip = request.headers.get("X-Forwarded-For", request.remote_addr)
    # print("🚀 ~ client_ip:", client_ip)
    # server_ip = socket.gethostbyname(socket.gethostname())
    # print("🚀 ~ server_ip:", server_ip)

    # Timestamp with milliseconds
    # timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    # content_type="application/json"
    content_type=request.content_type
    endpoint=request.endpoint or endpoint
    endpoint = "/"+endpoint
    
    # if flag == "STOP":
        # execution_time = time.time() - start_time 
        # execution_time = round((time.time() - start_time) * 1000, 2) 
    
    if body != "{}":
        data = request.get_json()
        json_data = json.dumps(data)
        body = encrypt(json_data)

    client_ip = request.remote_addr
    server_ip = request.host

    # Log format
    log_entry = (
        f"{timestamp} [{level}]  {transaction_id}  {service_name}  {endpoint}  {protocol}  {method}  {execution_type}  "
        f"{content_type}  {function_name}  '{caller_info}'  {execution_time} s  {server_ip}  {client_ip}  {event_name}  "
        f"{trace_id}  {prev_transaction_id}  {body}  {result}  {error}  [{flag}]  '{message}'"
    )

    print(log_entry) 
    with open(f"./logs/pattern-matching-{datetime.now().strftime('%Y-%m-%d')}.log", "a") as log_file:
       log_file.write(log_entry + "\n")
    

def error_log(error_message):
    print("🚀 ~ error_message:", error_message)
    save_log(level="ERROR", error=error_message, message="Message-Null")

def warn_log(warning_message):
    print("🚀 ~ warning_message:", warning_message)
    save_log(level="WARN", error=warning_message, message="Message-Null")

def debug_log(debug_message):
    print("🚀 ~ debug_message:", debug_message)
    save_log(level="DEBUG", error=debug_message, message="Message-Null")

def fatal_log(fatal_message):
    print("🚀 ~ fatal_message:", fatal_message)
    save_log(level="FATAL", error=fatal_message, message="Message-Null")

def info_log(info_message):
    save_log(message=info_message)

def start_log():
    save_log(message="Process started")

def finish_log(execution_time):
    save_log(flag="STOP", execution_time=execution_time, message="Process finished")


app = Flask(__name__)


#PHONE VALIDATOR
#PHONE VALIDATOR
#PHONE VALIDATOR

def validate_phone_number(number, region="ID"):  # Default to Indonesia ("ID")
    # Remove '+' and check if it's all digits
    # logging.info(f"Validating phone number: {number}, Region: {region}")
    sanitized_number = number.lstrip("+")
    if not sanitized_number.isdigit():
        # logging.warning("Invalid phone number format")
        return {"error": "Phone number must contain only numbers"}

    try:
        parsed_number = phonenumbers.parse(number, region)
        is_valid = phonenumbers.is_valid_number(parsed_number)
        is_possible = phonenumbers.is_possible_number(parsed_number)
        
        if is_valid:
            country = geocoder.description_for_number(parsed_number, "en")
            provider = carrier.name_for_number(parsed_number, "en")
            formatted = phonenumbers.format_number(parsed_number, phonenumbers.PhoneNumberFormat.INTERNATIONAL)
            # logging.info("Phone number is valid")
            return {
                "valid": True,
                "possible": is_possible,
                "formatted": formatted,
                "country": country,
                "carrier": provider
            }
        else:
            # logging.info("Phone number is not valid")
            return {"valid": False, "possible": is_possible}
    
    except phonenumbers.NumberParseException:
        # logging.error("Failed to parse phone number", exc_info=True)
        return {"valid": False, "possible": False}

@app.route("/number", methods=["POST"])
def number():
    start_log()
    start_time = time.time()
    data = request.json
    number = data.get('number')
    region = data.get('region', "ID")

    if not number:
        return jsonify({"error": "Phone number is required"}), 400

    result = validate_phone_number(number, region)
    execution_time = round(time.time() - start_time, 2) 
    finish_log(execution_time)
    return jsonify(result)


#EMAIL MATCHING
#EMAIL MATCHING
#EMAIL MATCHING

def calculate_relevance(name, email):
    """Calculate how much the name appears in the email local part (before @)."""
    email_local = email.split("@")[0].lower()
    name_lower = name.replace(" ", "").lower()
    score = SequenceMatcher(None, name_lower, email_local).ratio() * 100
    return round(score)

def validate_email_full(email, name):
    # logging.info(f"Validating email: {email} for name: {name}")
    """Perform multiple email validation checks and relevance scoring."""
    result = {"email": email, "name": name, "valid_syntax": False, "valid_domain": False, "relevance_score": 0, "final_status": "Invalid"}
    
    # 1️⃣ **Syntax & Domain Check using `email-validator`**
    try:
        v = validate_email(email, check_deliverability=True)
        #true jika syntax penulisan email sesuai aturan
        result["valid_syntax"] = True
        #true jika domain penulisan email ada 
        result["valid_domain"] = True
    except EmailNotValidError:
        # logging.warning("Invalid email syntax or domain")
        return result  # Early return if syntax is invalid

    # 2️⃣ **Calculate Name-Email Relevance**
    result["relevance_score"] = calculate_relevance(name, email)

    # 3️⃣ **Final Decision**
    if result["valid_syntax"] and result["valid_domain"] and result["relevance_score"] >= 55:
        result["final_status"] = "Valid"
    
    return result

@app.route('/email', methods=['POST'])
def email():
    start_log()
    start_time = time.time()
    """API endpoint to validate an email address and name relevance."""
    data = request.json
    email = data.get("email")
    name = data.get("name")

    if not email or not name:
        # logging.warning("Email or name not provided in request")
        return jsonify({"error": "Email and name are required"}), 400

    validation_result = validate_email_full(email, name)
    execution_time = round(time.time() - start_time, 2) 
    finish_log(execution_time)
    return jsonify(validation_result), 200

# IMAGE PATTERN MATCHING
# IMAGE PATTERN MATCHING
# IMAGE PATTERN MATCHING

# Resize images
def imageResize(image, maxD=500):
    height, width = image.shape[:2]
    aspectRatio = width / height
    if aspectRatio < 1:
        newSize = (int(maxD * aspectRatio), maxD)
    else:
        newSize = (maxD, int(maxD / aspectRatio))
    return cv2.resize(image, newSize)

# Compute SIFT descriptors
def computeSIFT(image):
    sift = cv2.SIFT_create()
    return sift.detectAndCompute(image, None)

# Match descriptors
def calculateMatches(des1, des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    topResults = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            topResults.append(m)
    return topResults

# Calculate match score
def calculateScore(matches, keypoint1, keypoint2):
    return 100 * (len(matches) / min(len(keypoint1), len(keypoint2)))

TEMPLATE_PATH = "./uploaded/template"

def download_image(url, save_dir):
    """Download an image from a given URL and save it locally."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Referer": url,  # Some websites require a referer
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
    }
    
    try:
        response = requests.get(url, headers=headers ,stream=True)
        # print("🚀 ~ response:", response)
        if response.status_code == 200:
            # print("🚀 ~ response.status_code:", response.status_code)
            ext = os.path.splitext(url)[1].lower()
            # print("🚀 ~ ext:", ext)
            if ext not in {".jpg", ".jpeg", ".png"}:
                warn_log(f"Invalid file extension: {ext}")
                return None  # Invalid file extension
            
            file_path = os.path.join(save_dir, os.path.basename(url))
            # print("🚀 ~ file_path:", file_path)
            with open(file_path, "wb") as file:
                for chunk in response.iter_content(1024):
                    # print("🚀 ~ file:", file)
                    file.write(chunk)
            return file_path
    except Exception as e:
        print(f"Error downloading image {url}: {e}")
    return None

domain_url = os.getenv("DOMAIN_URL")
# print("🚀 ~ domain_url:", domain_url)
@app.route('/image', methods=['POST'])
def image():
    start_time = time.time()
    start_log()

    # Read JSON body
    data = request.get_json()
    if not data or "klaim" not in data or "pengkinian" not in data:
        return jsonify({"error": "Both klaim and pengkinian image URLs must be provided."}), 400

    klaim_urls = data.get("klaim", [])
    pengkinian_urls = data.get("pengkinian", [])

    if not isinstance(klaim_urls, list) or not isinstance(pengkinian_urls, list):
        return jsonify({"error": "Klaim and Pengkinian must be lists of URLs."}), 400

    # Create temporary directories
    klaim_temp_dir = tempfile.mkdtemp()
    pengkinian_temp_dir = tempfile.mkdtemp()

    # Download images
    klaim_paths = [download_image(domain_url+url, klaim_temp_dir) for url in klaim_urls if url]
    # print("🚀 ~ klaim_paths:", klaim_paths)
    pengkinian_paths = [download_image(domain_url+url, pengkinian_temp_dir) for url in pengkinian_urls if url]
    # print("🚀 ~ pengkinian_paths:", pengkinian_paths)

    klaim_paths = [p for p in klaim_paths if p]  # Remove None values
    pengkinian_paths = [p for p in pengkinian_paths if p]  # Remove None values

    if not klaim_paths or not pengkinian_paths:
        error_log("Failed to download valid images.")
        return jsonify({"error": "Failed to download valid images."}), 400

    # Validate template path
    if not os.path.exists(TEMPLATE_PATH):
        error_log("Template path does not exist on the server.")
        return jsonify({"error": "Template path does not exist on the server."}), 400

    # Load template images
    valid_extensions = {".jpg", ".jpeg", ".png"}
    template_files = [f for f in os.listdir(TEMPLATE_PATH) if os.path.splitext(f)[1].lower() in valid_extensions]
    template_paths = [os.path.join(TEMPLATE_PATH, f) for f in template_files]
    templatesBW = [imageResize(cv2.imread(t, 0)) for t in template_paths]

    results = []

    # Process klaim and pengkinian images
    for klaimPath in klaim_paths:
        klaimBW = imageResize(cv2.imread(klaimPath, 0))
        if klaimBW is None:
            continue

        for pengkinianPath in pengkinian_paths:
            pengkinianBW = imageResize(cv2.imread(pengkinianPath, 0))
            if pengkinianBW is None:
                continue

            kp1, des1 = computeSIFT(pengkinianBW)
            kp2, des2 = computeSIFT(klaimBW)
            matches = calculateMatches(des1, des2)

            if len(matches) > 0:
                score = calculateScore(matches, kp1, kp2)
                result = "match" if score >= 17 else "not match"
                results.append({"pengkinian": os.path.basename(pengkinianPath),
                                "klaim": os.path.basename(klaimPath),
                                "result": result,
                                "score": score})

        for template, templatePath in zip(templatesBW, template_paths):
            kp1, des1 = computeSIFT(template)
            kp2, des2 = computeSIFT(klaimBW)
            matches = calculateMatches(des1, des2)

            if len(matches) > 0:
                score = calculateScore(matches, kp1, kp2)
                result = "match" if score >= 17 else "not match"
                results.append({"template": os.path.basename(templatePath),
                                "klaim": os.path.basename(klaimPath),
                                "result": result,
                                "score": score})

    # Filter results to include only those with "match"
    filtered_results = [r for r in results if r["result"] == "match"]

    execution_time = round(time.time() - start_time, 2)
    finish_log(execution_time)
    return jsonify(filtered_results), 200

@app.route('/klaim-to-pengkinian', methods=['POST'])
def klaimToPengkinian():
    start_time = time.time()
    start_log()

    # Read JSON body
    data = request.get_json()
    if not data or "klaim" not in data or "pengkinian" not in data:
        return jsonify({"error": "Both klaim and pengkinian image URLs must be provided."}), 400

    klaim_urls = data.get("klaim", [])
    pengkinian_urls = data.get("pengkinian", [])

    if not isinstance(klaim_urls, list) or not isinstance(pengkinian_urls, list):
        return jsonify({"error": "Klaim and Pengkinian must be lists of URLs."}), 400

    # Create temporary directories
    klaim_temp_dir = tempfile.mkdtemp()
    pengkinian_temp_dir = tempfile.mkdtemp()

    # Download images
    klaim_paths = [download_image(domain_url+url, klaim_temp_dir) for url in klaim_urls if url]
    # print("🚀 ~ klaim_paths:", klaim_paths)
    pengkinian_paths = [download_image(domain_url+url, pengkinian_temp_dir) for url in pengkinian_urls if url]
    # print("🚀 ~ pengkinian_paths:", pengkinian_paths)

    klaim_paths = [p for p in klaim_paths if p]  # Remove None values
    pengkinian_paths = [p for p in pengkinian_paths if p]  # Remove None values

    if not klaim_paths or not pengkinian_paths:
        error_log("Failed to download valid images.")
        return jsonify({"error": "Failed to download valid images."}), 400

    results = []

    # Process klaim and pengkinian images
    for klaimPath in klaim_paths:
        klaimBW = imageResize(cv2.imread(klaimPath, 0))
        if klaimBW is None:
            continue

        for pengkinianPath in pengkinian_paths:
            pengkinianBW = imageResize(cv2.imread(pengkinianPath, 0))
            if pengkinianBW is None:
                continue

            kp1, des1 = computeSIFT(pengkinianBW)
            kp2, des2 = computeSIFT(klaimBW)
            matches = calculateMatches(des1, des2)

            if len(matches) > 0:
                score = calculateScore(matches, kp1, kp2)
                result = "match" if score >= 17 else "not match"
                results.append({"pengkinian": os.path.basename(pengkinianPath),
                                "klaim": os.path.basename(klaimPath),
                                "result": result,
                                "score": score})

    # Filter results to include only those with "match"
    filtered_results = [r for r in results if r["result"] == "match"]

    execution_time = round(time.time() - start_time, 2)
    finish_log(execution_time)
    return jsonify(filtered_results), 200

@app.route('/klaim-to-template', methods=['POST'])
def klaimToTemplate():
    start_time = time.time()
    start_log()

    # Read JSON body
    data = request.get_json()
    if not data or "klaim" not in data:
        return jsonify({"error": "Klaim image URLs must be provided."}), 400

    klaim_urls = data.get("klaim", [])

    if not isinstance(klaim_urls, list) :
        return jsonify({"error": "Klaim must be lists of URLs."}), 400

    # Create temporary directories
    klaim_temp_dir = tempfile.mkdtemp()

    # Download images
    klaim_paths = [download_image(domain_url+url, klaim_temp_dir) for url in klaim_urls if url]
    # print("🚀 ~ klaim_paths:", klaim_paths)

    klaim_paths = [p for p in klaim_paths if p]  # Remove None values

    if not klaim_paths:
        error_log("Failed to download valid images.")
        return jsonify({"error": "Failed to download valid images."}), 400

    # Validate template path
    if not os.path.exists(TEMPLATE_PATH):
        error_log("Template path does not exist on the server.")
        return jsonify({"error": "Template path does not exist on the server."}), 400

    # Load template images
    valid_extensions = {".jpg", ".jpeg", ".png"}
    template_files = [f for f in os.listdir(TEMPLATE_PATH) if os.path.splitext(f)[1].lower() in valid_extensions]
    template_paths = [os.path.join(TEMPLATE_PATH, f) for f in template_files]
    templatesBW = [imageResize(cv2.imread(t, 0)) for t in template_paths]

    results = []

    # Process klaim and pengkinian images
    for klaimPath in klaim_paths:
        klaimBW = imageResize(cv2.imread(klaimPath, 0))
        if klaimBW is None:
            continue

        for template, templatePath in zip(templatesBW, template_paths):
            kp1, des1 = computeSIFT(template)
            kp2, des2 = computeSIFT(klaimBW)
            matches = calculateMatches(des1, des2)

            if len(matches) > 0:
                score = calculateScore(matches, kp1, kp2)
                result = "match" if score >= 17 else "not match"
                results.append({"template": os.path.basename(templatePath),
                                "klaim": os.path.basename(klaimPath),
                                "result": result,
                                "score": score})

    # Filter results to include only those with "match"
    filtered_results = [r for r in results if r["result"] == "match"]

    execution_time = round(time.time() - start_time, 2)
    finish_log(execution_time)
    return jsonify(filtered_results), 200

if __name__ == '__main__':
    # logging.info("Starting Flask server on port 8000")
    app.run(host="0.0.0.0", port=8000, debug=True)
    # app.run(host="0.0.0.0", port=8000)