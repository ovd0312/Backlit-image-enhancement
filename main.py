import cv2
import numpy as np
import copy
from guided_filter.core.filter import GuidedFilter
import os
import sklearn
import skfuzzy as fuzz
import matplotlib
from skfuzzy import control as ctrl
import math

def nearestPowerOf2(N):
    a = int(math.log2(N))
    if 2**a == N:
        return N
    return int(2**(a + 1))

def mst_clustering(k, img):
    print(k)
    shape = img.shape
    pixels = img.reshape(shape[0]*shape[1])
    pixels = sorted(pixels)
    pixels.reverse()
    label_map = {}
    data_diff = []
    for i in range(len(pixels)-1):
        data_diff.append(pixels[i]-pixels[i+1])
    diff_dict = {}
    for i in range(len(data_diff)):
        diff_dict[data_diff[i]] = i
    data_diff = sorted(data_diff)
    data_diff.reverse()
    boundaries = []
    for i in range(k-1):
        pos = diff_dict[data_diff[i]]
        boundaries.append(pos)
    boundaries = sorted(boundaries)
    print(boundaries)

    print(len(boundaries))
    curr = 0
    for i in range(len(pixels)):
        if curr <k-1 and i>boundaries[curr]:
            curr = curr+1
        label_map[pixels[i]] = curr
    labels = copy.deepcopy(img)
    # for i in range(256):
        # print(i, label_map[i])
    for x in range(labels.shape[0]):
        for y in range(labels.shape[1]):
            labels[x,y] = label_map[img[x,y]]*(255/(k-1))
            # print(label_map[img[x,y]], img[x,y])
    # print(f"k {k}")
    return labels

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

def recolorize_output(O, O_cap, I_he, img, I):
    O_rgb = []
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
    return O_rgb

def pyramid_blending(W, I, O):
    num_levels = 3
    row, col = W.shape
    ex = int(nearestPowerOf2(row)-row)
    ec = int(nearestPowerOf2(col)-col)
    # print(ex,ec)
    W = np.pad(W,((0,ex), (0, ec)), mode = 'constant', constant_values = 0)
    I = np.pad(I,((0,ex), (0, ec)), mode = 'constant', constant_values = 0)
    O = np.pad(O,((0,ex), (0, ec)), mode = 'constant', constant_values = 0)
    # print(W.shape)

    G = copy.deepcopy(I)
    gpA = [G]
    for i in range(num_levels):
        G = cv2.pyrDown(G)
        gpA.append(G)
    
    G = copy.deepcopy(O)
    gpB = [G]
    for i in range(num_levels):
        G = cv2.pyrDown(G)
        gpB.append(G)

    G = copy.deepcopy(W)
    gpW = [G]
    for i in range(num_levels):
        G = cv2.pyrDown(G)
        gpW.append(G)

    lpA = [gpA[num_levels-1]]
    for i in range(num_levels-1,0,-1):
        GE = cv2.pyrUp(gpA[i])
        # print(GE.shape, gpA[i-1].shape)
        L = cv2.subtract(gpA[i-1],GE)
        lpA.append(L)

    lpB = [gpB[num_levels-1]]
    for i in range(num_levels-1,0,-1):
        GE = cv2.pyrUp(gpB[i])
        L = cv2.subtract(gpB[i-1],GE)
        lpB.append(L)

    LS = []
    gpW.reverse()
    gpW = gpW[1:]
    for wa,la,lb in zip(gpW,lpA,lpB):
        ls = copy.deepcopy(la)
        print(ls.shape, wa.shape)
        for y in range(ls.shape[0]):
            for x in range(ls.shape[1]):
                ls[y,x] = (1 - wa[y,x])*lb[y,x] + wa[y,x]*la[y,x]
        LS.append(ls)

    ls_ = LS[0]
    for i in range(1,num_levels):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])

    return ls_[:row, :col]


def get_fuzzy(c, img):
    height = img.shape[0]
    width = img.shape[1]
    cnt = height * width
    pixels = img.reshape(1,cnt)
    # print(pixels.shape)
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(pixels, c, 1.5, error=0.005, maxiter=1000, init=None)
    
    weight_map = []
    if cntr[0] > cntr[1]:
        weight_map = u[0]
    else:
        weight_map = u[1]
    weight_map = weight_map.reshape(cnt,1)
    weight_map = weight_map.reshape(height, width)
    # print(weight_map)
    return weight_map

def process_image(INPUT_IMAGE_PATH, INPUT_IMAGE_NAME, OUTPUT_IMAGE_DIR, USE_VAL_AS_GRAY):

    # Creating output folder
    print(f"Processing the image - {INPUT_IMAGE_PATH}")
    OUTPUT_IMAGE_PATH = f"{OUTPUT_IMAGE_DIR}{INPUT_IMAGE_NAME.replace('.','-')}\\"
    if not os.path.exists(OUTPUT_IMAGE_PATH):
        os.makedirs(OUTPUT_IMAGE_PATH)
    
    # Parameters
    gamma = 2
    alpha = 0.5
    radius = 60
    eps = 0.001
    k_means_num = 5

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

    cv2.imwrite(OUTPUT_IMAGE_PATH + "\\alpha_blended_image.jpg", I_he)
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
    cv2.imwrite(OUTPUT_IMAGE_PATH + "\\mask.jpg", W)

    # cv2.imshow("W",W)
    # cv2.waitKey(0)
    technique_name = f"k-means-{k_means_num}"

    # STEP - 4
    # Getting smoothened weight map using guided filter
    GF = GuidedFilter(I, radius, eps)
    W_cap = GF.filter(W)
    print(W.max())
    cv2.imwrite(OUTPUT_IMAGE_PATH + "\\k_means_mask.jpg", (W_cap)*255)


    
    W_cap_fuzzy = get_fuzzy(2,I)
    cv2.imwrite(OUTPUT_IMAGE_PATH + "\\fuzzy_mask.jpg", (W_cap_fuzzy)*255)

    
    # STEP - 5
    O_cap = copy.deepcopy(O)
    for x in range(O_cap.shape[0]):
        for y in range(O_cap.shape[1]):
            O_cap[x,y] = (1-W_cap[x,y])*O_cap[x,y] + (W_cap[x,y])*I[x,y]
    # O_cap = pyramid_blending(W_cap,I,O_cap)

    O_cap_fuzzy = copy.deepcopy(O)
    for x in range(O_cap_fuzzy.shape[0]):
        for y in range(O_cap_fuzzy.shape[1]):
            O_cap_fuzzy[x,y] = (1-W_cap_fuzzy[x,y])*O_cap_fuzzy[x,y] + (W_cap_fuzzy[x,y])*I[x,y]

    cv2.imwrite(OUTPUT_IMAGE_PATH + "\\grayed_output.jpg", O_cap)

    # STEP - 6
    # Recolorize output
   
    O_rgb = recolorize_output(O,O_cap,I_he,img,I)
    O_rgb_fuzzy = recolorize_output(O,O_cap_fuzzy,I_he,img,I)

    
                    
    cv2.imwrite(OUTPUT_IMAGE_PATH + "\\colored_output.jpg", O_rgb)
    cv2.imwrite(OUTPUT_IMAGE_PATH + "\\colored_output_fuzzy.jpg", O_rgb_fuzzy)
    cv2.imwrite(OUTPUT_IMAGE_PATH + "\\colored_output_comp.jpg", np.hstack((O_rgb, O_rgb_fuzzy)))

    

    cv2.imwrite(OUTPUT_IMAGE_PATH + "\\final_comparison.jpg", np.hstack((img, O_rgb)))

if __name__ == "__main__":
    
    BULK_PROCESSING_MODE = False
    INPUT_IMAGE_NAME = [
        "15.jpg"
    ]
    INPUT_IMAGE_DIR = f".\input images\\"
    OUTPUT_IMAGE_DIR = f".\output_images5\\"
    USE_VAL_AS_GRAY = True
    if BULK_PROCESSING_MODE:
        for img in os.listdir(INPUT_IMAGE_DIR):
            process_image(f"{INPUT_IMAGE_DIR}{img}", img, OUTPUT_IMAGE_DIR, USE_VAL_AS_GRAY)
    else:
        for img in INPUT_IMAGE_NAME:
            process_image(f"{INPUT_IMAGE_DIR}{img}", img, OUTPUT_IMAGE_DIR, USE_VAL_AS_GRAY)


