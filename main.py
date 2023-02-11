import cv2
import numpy as np
import copy
from guided_filter.core.filter import GuidedFilter
import os

def k_means(k, img):

    shape = img.shape
    img = img.flatten()
    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # Set flags (Just to avoid line break in the code)
    flags = cv2.KMEANS_RANDOM_CENTERS
    # Apply KMeans
    img = np.float32(img)
    compactness,labels,centers = cv2.kmeans(img,k,None,criteria,10,flags)
    
    center_rank = dict()
    arr = np.zeros(centers.shape[0])

    for i, center in enumerate(centers):
        arr[i] = center[0]
    arr = np.sort(arr)
    for i, center in enumerate(arr):
        center_rank[center] = i
    labels = labels.reshape(shape)
    for x in range(labels.shape[0]):
        for y in range(labels.shape[1]):
            labels[x,y] = center_rank[centers[labels[x,y]][0]] * (255 / (k-1))
    return labels

def process_image(INPUT_IMAGE_PATH, INPUT_IMAGE_NAME, OUTPUT_IMAGE_DIR, USE_VAL_AS_GRAY):

    # Creating output folder
    print(f"Processing the image - {INPUT_IMAGE_PATH}")
    OUTPUT_IMAGE_PATH = f"{OUTPUT_IMAGE_DIR}{INPUT_IMAGE_NAME.replace('.','-')}\\"
    if not os.path.exists(OUTPUT_IMAGE_PATH):
        os.makedirs(OUTPUT_IMAGE_PATH)
    
    # Parameters
    gamma = 2
    alpha = 0.5
    radius = 20
    eps = 0.001
    k_means_num = 20

    # STEP 1 - Get grayscale image
    if USE_VAL_AS_GRAY:
        # Convert image from BGR to HSV color space
        # Use the 'V' (value) matrix as grayscale image
        img = cv2.imread(INPUT_IMAGE_PATH)
        I = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,2]
    else:
        # Get grayscale image by averaging RGB components
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                temp = 0
                for c in range(img.shape[2]):
                    temp = temp + img[y,x,c]
                I[y,x] = temp/3

    cv2.imwrite(OUTPUT_IMAGE_PATH + "\\input_image.jpg", img)
    cv2.imwrite(OUTPUT_IMAGE_PATH + "\\grayed_input_image.jpg", I)

    I_gamma = copy.deepcopy(I)

    # STEP - 2
    # Performing gamma correction, storing gamma corrected image in I_gamma
    for y in range(I.shape[0]):
        for x in range(I.shape[1]):
                I_gamma[y,x] = int(255 * ((I[y,x]/255)**(1/gamma)))

    cv2.imwrite(OUTPUT_IMAGE_PATH + "\\gamma_corrected.jpg", I_gamma)

    # Histogram equalization
    I_he = cv2.equalizeHist(I) 
    O = copy.deepcopy(I_he)

    # Merge histogram equalized image and gamma corrected image
    for y in range(I.shape[0]):
        for x in range(I.shape[1]):
                O[y,x] = (1 - alpha)*I_gamma[y,x] + alpha*I_he[y,x]

    cv2.imwrite(OUTPUT_IMAGE_PATH + "\\hist_gamma_merged_image.jpg", I_he)

    # Recoloring histogram,gamma corrected and both merged image
    # For debugging purposes only
    # Can be skipped
    if USE_VAL_AS_GRAY:
        O_rgb = copy.deepcopy(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        O_rgb[:,:,2] = I_he
        O_rgb = cv2.cvtColor(O_rgb, cv2.COLOR_HSV2BGR)
    else:
        O_rgb = copy.deepcopy(img)
        for y in range(O_rgb.shape[0]):
            for x in range(O_rgb.shape[1]):
                for c in range(O_rgb.shape[2]):
                    O_rgb[y,x,c] = img[y,x,c]*(I_he[y,x]/(I[y,x]+1))

    cv2.imwrite(OUTPUT_IMAGE_PATH + "\\alpha_blended_image.jpg", O)
    cv2.imwrite(OUTPUT_IMAGE_PATH + "\\histogram_eq_colored.jpg", O_rgb)

    if USE_VAL_AS_GRAY:
        O_rgb = copy.deepcopy(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        O_rgb[:,:,2] = I_gamma
        O_rgb = cv2.cvtColor(O_rgb, cv2.COLOR_HSV2BGR)
    else:
        O_rgb = copy.deepcopy(img)
        for y in range(O_rgb.shape[0]):
            for x in range(O_rgb.shape[1]):
                for c in range(O_rgb.shape[2]):
                    O_rgb[y,x,c] = img[y,x,c]*(I_gamma[y,x]/(I[y,x]+1))

    cv2.imwrite(OUTPUT_IMAGE_PATH + "\\gamma_corrected_colored.jpg", O_rgb)

    if USE_VAL_AS_GRAY:
        O_rgb = copy.deepcopy(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        O_rgb[:,:,2] = O
        O_rgb = cv2.cvtColor(O_rgb, cv2.COLOR_HSV2BGR)
    else:
        O_rgb = copy.deepcopy(img)
        for y in range(O_rgb.shape[0]):
            for x in range(O_rgb.shape[1]):
                for c in range(O_rgb.shape[2]):
                    O_rgb[y,x,c] = img[y,x,c]*(O[y,x]/(I[y,x]+1))

    cv2.imwrite(OUTPUT_IMAGE_PATH + "\\gamma_histogram_colored.jpg", O_rgb)
    # Skippable part ends

    # STEP 3 - Otsu thresholding / k-means clustering

    # Otsu Threshold
    # ret, W = cv2.threshold(I, 0, 255, cv2.THRESH_BINARY +  cv2.THRESH_OTSU)
    # technique_name = "otsu's_threshold"

    # k-means clustering
    W = k_means(k_means_num, I)
    technique_name = f"k-means-{k_means_num}"

    cv2.imwrite(OUTPUT_IMAGE_PATH + f"\\{technique_name}.jpg", W)

    # STEP - 4
    # Getting smoothened weight map using guided filter
    GF = GuidedFilter(I, radius, eps)
    W_cap = GF.filter(W)
    cv2.imwrite(OUTPUT_IMAGE_PATH + "\\guided_filter_mask.jpg", (W_cap)*255)

    # STEP - 5
    O_cap = copy.deepcopy(O)
    for x in range(O_cap.shape[0]):
        for y in range(O_cap.shape[1]):
            O_cap[x,y] = (1-W_cap[x,y])*O_cap[x,y] + (W_cap[x,y])*I[x,y]

    cv2.imwrite(OUTPUT_IMAGE_PATH + "\\grayed_output.jpg", O_cap)

    # STEP - 6
    # Recolorize output
    if USE_VAL_AS_GRAY:
        O_rgb = copy.deepcopy(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        O_rgb[:,:,2] = O_cap
        O_rgb = cv2.cvtColor(O_rgb, cv2.COLOR_HSV2BGR)
    else:
        O_rgb = copy.deepcopy(img)
        for y in range(O_rgb.shape[0]):
            for x in range(O_rgb.shape[1]):
                for c in range(O_rgb.shape[2]):
                    O_rgb[y,x,c] = img[y,x,c]*(O_cap[y,x]/(I[y,x]+1))
                    
    cv2.imwrite(OUTPUT_IMAGE_PATH + "\\colored_output.jpg", O_rgb)

    cv2.imwrite(OUTPUT_IMAGE_PATH + "\\final_comparison.jpg", np.hstack((img, O_rgb)))

if __name__ == "__main__":
    
    BULK_PROCESSING_MODE = False
    INPUT_IMAGE_NAME = [
        "25.jpg"
    ]
    INPUT_IMAGE_DIR = f".\input images\\"
    OUTPUT_IMAGE_DIR = f".\output_images4\\"
    USE_VAL_AS_GRAY = True
    if BULK_PROCESSING_MODE:
        for img in os.listdir(INPUT_IMAGE_DIR):
            process_image(f"{INPUT_IMAGE_DIR}{img}", img, OUTPUT_IMAGE_DIR, USE_VAL_AS_GRAY)
    else:
        for img in INPUT_IMAGE_NAME:
            process_image(f"{INPUT_IMAGE_DIR}{img}", img, OUTPUT_IMAGE_DIR, USE_VAL_AS_GRAY)
