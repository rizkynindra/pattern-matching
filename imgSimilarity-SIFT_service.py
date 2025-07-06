from flask import Flask, request, jsonify
import os
import cv2
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from io import BytesIO

app = Flask(__name__)

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

@app.route('/match', methods=['POST'])
def match():
    data = request.json
    dataset_path = data.get('dataset_path')
    template_path = data.get('template_path')

    if not os.path.exists(dataset_path):
        return jsonify({"error": "Dataset path does not exist."}), 400

    if not os.path.exists(template_path):
        return jsonify({"error": "Template path does not exist."}), 400

    valid_extensions = ['.jpg', '.jpeg', '.png']
    imageList = [f for f in os.listdir(dataset_path) if os.path.splitext(f)[1].lower() in valid_extensions]
    templateList = [f for f in os.listdir(template_path) if os.path.splitext(f)[1].lower() in valid_extensions]

    templatesBW = [imageResize(cv2.imread(os.path.join(template_path, t), 0)) for t in tqdm(templateList)]
    results = []

    for imageName in tqdm(imageList):
        imagePath = os.path.join(dataset_path, imageName)
        imageBW = imageResize(cv2.imread(imagePath, 0))

        for template, templateName in zip(templatesBW, templateList):
            kp1, des1 = computeSIFT(template)
            kp2, des2 = computeSIFT(imageBW)
            matches = calculateMatches(des1, des2)

            if len(matches) > 0:
                score = calculateScore(matches, kp1, kp2)
                # if score >= 10:
                #     result = "match"
                result = "match" if score >= 17 else "not match"
                results.append({"template": templateName,
                                "image": imageName,
                                "result": result,
                                "score": score})

    # Filter results to include only those with "match"
    filtered_results = [r for r in results if r["result"] == "match"]

    return jsonify(filtered_results), 200

if __name__ == '__main__':
    app.run(debug=True, port=8000, host='0.0.0.0')
