from flask import Flask, request, jsonify
from email_validator import validate_email, EmailNotValidError
from difflib import SequenceMatcher

app = Flask(__name__)

def calculate_relevance(name, email):
    """Calculate how much the name appears in the email local part (before @)."""
    email_local = email.split("@")[0].lower()
    name_lower = name.replace(" ", "").lower()
    score = SequenceMatcher(None, name_lower, email_local).ratio() * 100
    return round(score)

def validate_email_full(email, name):
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
        return result  # Early return if syntax is invalid

    # 2️⃣ **Calculate Name-Email Relevance**
    result["relevance_score"] = calculate_relevance(name, email)

    # 3️⃣ **Final Decision**
    if result["valid_syntax"] and result["valid_domain"] and result["relevance_score"] > 60:
        result["final_status"] = "Valid"
    
    return result

@app.route('/check-email', methods=['POST'])
def check_email():
    """API endpoint to validate an email address and name relevance."""
    data = request.json
    email = data.get("email")
    name = data.get("name")

    if not email or not name:
        return jsonify({"error": "Email and name are required"}), 400

    validation_result = validate_email_full(email, name)
    return jsonify(validation_result), 200

if __name__ == '__main__':
    app.run(debug=True, port=8100, host='0.0.0.0')
