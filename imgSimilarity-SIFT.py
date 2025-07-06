import cv2
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import warnings
import itertools
import xlsxwriter
import sys
from tqdm import tqdm
from datetime import datetime
from string import ascii_uppercase as i
from itertools import combinations_with_replacement
from io import BytesIO

# Resize images to a similar dimension
# This helps improve accuracy and decreases unnecessarily high number of keypoints

def imageResizeTrain(image):
    maxD = 500
    height,width = image.shape
    aspectRatio = width/height
    if aspectRatio < 1:
        newSize = (int(maxD*aspectRatio),maxD)
    else:
        newSize = (maxD,int(maxD/aspectRatio))
    image = cv2.resize(image,newSize)
    return image

def imageResizeTest(image):
    maxD = 500
    height,width,channel = image.shape
    aspectRatio = width/height
    if aspectRatio < 1:
        newSize = (int(maxD*aspectRatio),maxD)
    else:
        newSize = (maxD,int(maxD/aspectRatio))
    image = cv2.resize(image,newSize)
    return image

# Using opencv's sift implementation here
def computeSIFT(image):
    sift = cv2.SIFT_create()
    return sift.detectAndCompute(image, None)

#ORB
# def computeSIFT(image):
#     sift = cv2.ORB_create()
#     return sift.detectAndCompute(image, None)

def fetchKeypointFromFile(i):
    filepath = keypoint_img_path + "/" + str(imageList[i].split('.')[0]) + ".txt"
    keypoint = []
    file = open(filepath,'rb')
    deserializedKeypoints = pickle.load(file)
    file.close()
    for point in deserializedKeypoints:
        temp = cv2.KeyPoint(
            x=point[0][0],
            y=point[0][1],
            size=point[1],
            angle=point[2],
            response=point[3],
            octave=point[4],
            class_id=point[5]
        )
        keypoint.append(temp)
    return keypoint

#template
def fetchKeypointFromFile_template(i):
    filepath = keypoint_temp_path + "/" + str(templateList[i].split('.')[0]) + ".txt"
    keypoint = []
    file = open(filepath,'rb')
    deserializedKeypoints = pickle.load(file)
    file.close()
    for point in deserializedKeypoints:
        temp = cv2.KeyPoint(
            x=point[0][0],
            y=point[0][1],
            size=point[1],
            angle=point[2],
            response=point[3],
            octave=point[4],
            class_id=point[5]
        )
        keypoint.append(temp)
    return keypoint


def fetchDescriptorFromFile(i):
    filepath = descriptor_img_path + "/" + str(imageList[i].split('.')[0]) + ".txt"
    file = open(filepath,'rb')
    descriptor = pickle.load(file)
    file.close()
    return descriptor


#template
def fetchDescriptorFromFile_template(i):
    filepath = descriptor_temp_path + "/" + str(templateList[i].split('.')[0]) + ".txt"
    file = open(filepath,'rb')
    descriptor = pickle.load(file)
    file.close()
    return descriptor

def calculateResultsFor(i,j):
    keypoint1 = fetchKeypointFromFile_template(i)
    # keypoint1_template = fetchKeypointFromFile_template(i)
    descriptor1 = fetchDescriptorFromFile_template(i)
    # descriptor1_template = fetchDescriptorFromFile_template(i)
    keypoint2 = fetchKeypointFromFile(j)
    # keypoint2_template = fetchKeypointFromFile_template(j)
    descriptor2 = fetchDescriptorFromFile(j)
    # descriptor2_template = fetchDescriptorFromFile_template(j)
    matches = calculateMatches(descriptor1, descriptor2)
    score = calculateScore(len(matches),len(keypoint1),len(keypoint2))
    # plot = getPlotFor(i,j,keypoint1,keypoint2,matches)
    # print(len(matches),len(keypoint1),len(keypoint2),len(descriptor1),len(descriptor2))
    return score
    # if score > 10:
    #     print("Image already used")

def getPlotFor(i,j,keypoint1,keypoint2,matches):
    image1 = imageResizeTest(cv2.imread(template_path + "/" + templateList[i]))
    image2 = imageResizeTest(cv2.imread(folder_path + "/" + imageList[j]))
    return getPlot(image1,image2,keypoint1,keypoint2,matches)

def calculateScore(matches,keypoint1,keypoint2):
    return 100 * (matches/min(keypoint1,keypoint2))

def calculateMatches(des1,des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    topResults1 = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            topResults1.append([m])

    matches = bf.knnMatch(des2,des1,k=2)
    topResults2 = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            topResults2.append([m])

    topResults = []
    for match1 in topResults1:
        match1QueryIndex = match1[0].queryIdx
        match1TrainIndex = match1[0].trainIdx

        for match2 in topResults2:
            match2QueryIndex = match2[0].queryIdx
            match2TrainIndex = match2[0].trainIdx

            if (match1QueryIndex == match2TrainIndex) and (match1TrainIndex == match2QueryIndex):
                topResults.append(match1)
    return topResults

def getPlot(image1,image2,keypoint1,keypoint2,matches):
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    matchPlot = cv2.drawMatchesKnn(
        image1,
        keypoint1,
        image2,
        keypoint2,
        matches,
        None,
        [255,255,255],
        flags=2
    )
    return matchPlot

def getCombination(imgDir):
    imgList = []
    idxList = []
    idx = 0
    # folder_template = '/Users/aditya.permana/Downloads/Deepfake/01OCT2024'
    for filename_temp in os.listdir(imgDir):
        filepath = os.path.join(imgDir,filename_temp)
        imgList.append(filepath)
        idxList.append(idx)
        idx += 1
    print(len(idxList))
    combinations = list(itertools.combinations(idxList, 2))
    # print(len(combinations))
    print(combinations)

if __name__ =="__main__":
    # Define the folder path
    pathDataset             = sys.argv[1]
    pathReportResult        = sys.argv[2]
    lengthDataset           = len(pathDataset.split('/'))
    dataset_name            = pathDataset.split('/')[lengthDataset-1]
    folder_path             = pathDataset
    template_path           = "/apps/data/DeepFake/template"
    result_path             = folder_path + "/result"
    keypoint_img_path       = folder_path + "/keypoint_image"
    keypoint_temp_path      = folder_path + "/keypoint_template"
    descriptor_img_path     = folder_path + "/descriptor_image"
    descriptor_temp_path    = folder_path + "/descriptor_template"
    feather_path            = folder_path + "/feather"

    if os.path.exists(result_path):
        pass
    else:
        os.mkdir(result_path)

    if os.path.exists(keypoint_img_path):
        pass
    else:
        os.mkdir(keypoint_img_path)

    if os.path.exists(keypoint_temp_path):
        pass
    else:
        os.mkdir(keypoint_temp_path)

    if os.path.exists(descriptor_img_path):
        pass
    else:
        os.mkdir(descriptor_img_path)

    if os.path.exists(descriptor_temp_path):
        pass
    else:
        os.mkdir(descriptor_temp_path)

    if os.path.exists(feather_path):
        pass
    else:
        os.mkdir(feather_path)
    
    if os.path.exists(pathReportResult):
        pass
    else:
        os.mkdir(pathReportResult)

    # List of valid image extensions
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']

    # Get a list of all files in the folder
    files_in_folder    = os.listdir(folder_path)
    template_in_folder = os.listdir(template_path)

    # Filter files by image extensions
    image_files    = [file for file in files_in_folder if os.path.splitext(file)[1].lower() in valid_extensions]
    # print("🚀 ~ image_files:", image_files)
    template_files = [file for file in template_in_folder if os.path.splitext(file)[1].lower() in valid_extensions]
    # print("🚀 ~ template_files:", template_files)

    # Define a list of images the way you like
    imageList    = image_files
    templateList = template_files

#Batch
batch_size = 500  # Adjust the batch size based on your system's capacity

#TEMPLATE
templatesBW = []
for templateName in tqdm(templateList):
    templatePath = os.path.join(template_path,str(templateName))
    try:
        templatesBW.append(imageResizeTrain(cv2.imread(templatePath,0))) # flag 0 means grayscale
    except:
        continue

# Process Images in Batches
for batch_start in range(0, len(imageList), batch_size):
    # Get a batch of images
    batch_images = imageList[batch_start:batch_start + batch_size]

    imagesBW = []
    for templateName in tqdm(batch_images):
        imagePath = os.path.join(folder_path, str(templateName))
        try:
            # Read and resize the image (grayscale)
            img = cv2.imread(imagePath, 0)  # 0 = grayscale
            imagesBW.append(imageResizeTrain(img))
        except Exception as e:
            print(f"Error loading image {templateName}: {e}")
            continue

    keypoints_template = []
    descriptors_template = []
    for i,image in tqdm(enumerate(templatesBW)):
        keypointTemp, descriptorTemp = computeSIFT(image)
        keypoints_template.append(keypointTemp)
        descriptors_template.append(descriptorTemp)

# Extract keypoints and descriptors for each batch
    keypoints = []
    descriptors = []
    for i, image in tqdm(enumerate(imagesBW)):
        try:
            keypointTemp, descriptorTemp = computeSIFT(image)
            keypoints.append(keypointTemp)
            descriptors.append(descriptorTemp)
        except Exception as e:
            print(f"Error processing image {batch_images[i]}: {e}")
            continue

    # keypoints = []
    # descriptors = []
    # for i,image in tqdm(enumerate(imagesBW),disable=True):
    #     keypointTemp, descriptorTemp = computeSIFT(image)
    #     keypoints.append(keypointTemp)
    #     descriptors.append(descriptorTemp)

    # for i,keypoint in enumerate(keypoints):
    #     deserializedKeypoints = []
    #     filepath = keypoint_img_path + "/" + str(imageList[i].split('.')[0]) + ".txt"
    #     for point in keypoint:
    #         temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
    #         deserializedKeypoints.append(temp)
    #     with open(filepath, 'wb') as fp:
    #         pickle.dump(deserializedKeypoints, fp)

    # Save keypoints and descriptors to files
    for i, keypoint in enumerate(keypoints):
        deserializedKeypoints = [
            (point.pt, point.size, point.angle, point.response, point.octave, point.class_id) 
            for point in keypoint
        ]
        filepath = keypoint_img_path + "/" + str(batch_images[i].split('.')[0]) + ".txt"
        with open(filepath, 'wb') as fp:
            pickle.dump(deserializedKeypoints, fp)

    #template
    for i,keypoint in enumerate(keypoints_template):
        deserializedKeypoints = []
        filepath = keypoint_temp_path + "/" + str(templateList[i].split('.')[0]) + ".txt"
        for point in keypoint:
            temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
            deserializedKeypoints.append(temp)
        with open(filepath, 'wb') as fp:
            pickle.dump(deserializedKeypoints, fp)

    for i, descriptor in enumerate(descriptors):
        filepath = descriptor_img_path + "/" + str(batch_images[i].split('.')[0]) + ".txt"
        with open(filepath, 'wb') as fp:
            pickle.dump(descriptor, fp)
    
    # for i,descriptor in enumerate(descriptors):
    #     filepath = descriptor_img_path + "/" + str(imageList[i].split('.')[0]) + ".txt"
    #     with open(filepath, 'wb') as fp:
    #         pickle.dump(descriptor, fp)

    #template
    for i,descriptor in enumerate(descriptors_template):
        filepath = descriptor_temp_path + "/" + str(templateList[i].split('.')[0]) + ".txt"
        with open(filepath, 'wb') as fp:
            pickle.dump(descriptor, fp)

# Print the obtained combinations
colname = ['image1','image2','score']
all_result = pd.DataFrame(columns=colname)

for i in range(len(templateList)): #loop template
    # print("🚀 ~ i:", i)
    for j in tqdm(range(len(imageList))): #loop dataset
        # print("🚀 ~ j:", j)
        try:
            score = calculateResultsFor(i,j)
            # print("🚀 ~ score:", score)
        except Exception as e:
            print(f"Error processing {templateList[i]} and {imageList[j]}: {str(e)}")
            continue
        row = [templateList[i], imageList[j], score]
        result = pd.DataFrame([row],columns = colname)
        all_result = pd.concat([all_result,result])
        high_score = all_result[all_result['score'] >= 17]
# print(feather_path + "/result_"+ dataset_name +".feather")
high_score.to_feather(feather_path + "/result_"+ dataset_name +".feather")
# print("🚀 ~ high_score:", high_score)

name_split = high_score['image2']
# print("🚀 ~ name_split:", name_split)

# Extract the part before the first underscore for each filename
extracted_ids = [filename.split('_')[1] if len(filename.split('_')) > 1 else None for filename in name_split]
# extracted_ids                   = [filename.split('_')[1] for filename in name_split]
high_score['kode_pengajuan']    = extracted_ids

today       = datetime.today().strftime('%d%m%Y')
# dir_name    = "./Klaim/"
# tem_name    = "./Klaim/template/"

w = 10
h = 10
fig        = plt.figure(figsize=(8, 8))
columns    = 2
rows       = high_score.shape[0]
filename   = f'{pathReportResult}/results_klaim_{dataset_name}.xlsx'
high_excel = high_score.to_excel(f"{filename}")

# Get today's date in the desired format (e.g., '02102024')
today = datetime.today().strftime('%d%m%Y')

workbook  = xlsxwriter.Workbook(f'{pathReportResult}/compared_image_result_{dataset_name}.xlsx')
worksheet = workbook.add_worksheet()
worksheet.write('A1', "Image")
worksheet.write('B1', "Filename Image 1")
worksheet.write('C1', "Filename Image 2")
worksheet.write('D1', "Score")
worksheet.write('E1', "Kode Pengajuan")
# Iterate through high_score DataFrame
row_num = 2  # Start adding images from the second row (B2, C2, etc.)
for index, row in enumerate(high_score.itertuples(), start=1):
    # Load images
    img1 = mpimg.imread(f'{template_path}/{row.image1}')
    img2 = mpimg.imread(f'{folder_path}/{row.image2}')

    # Create a figure with both images
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].imshow(img1)
    axes[0].axis('off')
    axes[1].imshow(img2)
    axes[1].axis('off')

    # Save the figure as an image
    img_filename = f"{pathReportResult}/compare_{row.image1}_{row.image2}.jpeg"
    plt.savefig(img_filename)
    plt.close(fig)
    
    fileBaseImage = open(img_filename, 'rb')
    dataBaseImage = BytesIO(fileBaseImage.read())
    fileBaseImage.close()
    # worksheet.write('A'+str(row_num), "Base Image")
    worksheet.insert_image('A'+str(row_num), img_filename, {'image_data': dataBaseImage})
    worksheet.write('B'+str(row_num), f'{row.image1}')
    worksheet.write('C'+str(row_num), f'{row.image2}')
    worksheet.write('D'+str(row_num), row.score)
    worksheet.write('E'+str(row_num), row.kode_pengajuan)
    row_num += 1
    
workbook.close()
print(f"Images successfully saved to"+f'{pathReportResult}/compared_image_result_{dataset_name}.xlsx')
