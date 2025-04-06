import pandas as pd
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

def extract_glcm_features(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    # Convert to grayscale if necessary
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize image to 8-bit (if not already)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Compute GLCM
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)

    # Extract statistical properties
    contrast = graycoprops(glcm, 'contrast').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()

    return [contrast, dissimilarity, homogeneity, energy, correlation]

def extract_lbp_features(image, P=8, R=1, method='uniform'):
    # Convert to grayscale if necessary
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize image to 8-bit (if not already)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Compute LBP
    lbp = local_binary_pattern(image, P=P, R=R, method=method)
    
    # Generate histogram of LBP values
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))

    # Normalize the histogram
    hist = hist.astype("float")
    hist /= hist.sum()

    return hist.tolist()

def extract_color_variance(image):
    # Convert to RGB if image is in BGR format (OpenCV default)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Split channels (R, G, B)
    r, g, b = cv2.split(image)
    
    # Mean values
    mean_r = np.mean(r)
    mean_g = np.mean(g)
    mean_b = np.mean(b)
    
    # Variance values
    var_r = np.var(r)
    var_g = np.var(g)
    var_b = np.var(b)
    
    # Overall variance (across all channels)
    overall_var = np.var(image)
    
    return [mean_r, mean_g, mean_b, var_r, var_g, var_b, overall_var]

def extract_color_hist(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    hist_r = cv2.calcHist([image], [0], None, [32], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [32], [0, 256])
    hist_b = cv2.calcHist([image], [2], None, [32], [0, 256])

    hist_r = hist_r / hist_r.sum()
    hist_g = hist_g / hist_g.sum()
    hist_b = hist_b / hist_b.sum()

    hist_r_flat = hist_r.flatten()
    hist_g_flat = hist_g.flatten()
    hist_b_flat = hist_b.flatten()

    return np.concatenate((hist_r_flat,hist_g_flat,hist_b_flat))


def save_file(rq_file, filepath, filename):
    fn = secure_filename(filename)
    rq_file.save(os.path.join(filepath, fn))



def proc_image(filepath, filename):
    img = cv2.imread(os.path.join(filepath, filename))
    glcm_features = extract_glcm_features(img)
    lbp_features = extract_lbp_features(img)
    color_var_features = extract_color_variance(img)
    color_hist_features = extract_color_hist(img)

    # print(glcm_features)
    # print(lbp_features)
    # print(color_var_features)
    # print(color_hist_features)

    os.remove(os.path.join(filepath, filename))

    return glcm_features, lbp_features, color_var_features, color_hist_features

def get_df(pred_obj):
    with open("model_training/feature_names.txt", "r") as f:
        features = [x.strip() for x in f.readlines()]
    concatenated_values = np.concatenate((pred_obj[2], pred_obj[3], pred_obj[1], pred_obj[0]))
    return pd.DataFrame([concatenated_values], columns=features)
