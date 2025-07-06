from flask import Flask, request, jsonify
import phonenumbers
from phonenumbers import carrier, geocoder
import re

app = Flask(__name__)

def validate_phone_number(number, region="ID"):  # Default to Indonesia ("ID")
    # Remove '+' and check if it's all digits
    sanitized_number = number.lstrip("+")
    if not sanitized_number.isdigit():
        return {"error": "Phone number must contain only numbers"}

    try:
        parsed_number = phonenumbers.parse(number, region)
        is_valid = phonenumbers.is_valid_number(parsed_number)
        is_possible = phonenumbers.is_possible_number(parsed_number)
        
        if is_valid:
            country = geocoder.description_for_number(parsed_number, "en")
            provider = carrier.name_for_number(parsed_number, "en")
            formatted = phonenumbers.format_number(parsed_number, phonenumbers.PhoneNumberFormat.INTERNATIONAL)
            return {
                "valid": True,
                "possible": is_possible,
                "formatted": formatted,
                "country": country,
                "carrier": provider
            }
        else:
            return {"valid": False, "possible": is_possible}
    
    except phonenumbers.NumberParseException:
        return {"valid": False, "possible": False}

@app.route("/phone-validate", methods=["POST"])
def validate():
    data = request.json
    number = data.get('number')
    region = data.get('region', "ID")

    if not number:
        return jsonify({"error": "Phone number is required"}), 400

    result = validate_phone_number(number, region)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8200, debug=True)
