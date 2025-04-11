import pandas as pd
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score

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

def extract_image_id(file_name):
    # Extract the base name (e.g., 'ISIC_0024306.jpg')
    base_name = file_name.split('\\')[-1]
    # Remove the file extension (e.g., '.jpg')
    image_id = os.path.splitext(base_name)[0]
    return image_id

def load_csvs():
    # Define paths
    base_path = "dataset/data_exploration/"
    metadata_path = "dataset/dataverse_files/"

    # Load feature files
    color_var = pd.read_csv(os.path.join(base_path, "color_variance_features.csv"))
    color_hist = pd.read_csv(os.path.join(base_path, "combined_color_histogram_features.csv"))
    lbp = pd.read_csv(os.path.join(base_path, "combined_lbp_features.csv"))
    glcm = pd.read_csv(os.path.join(base_path, "glcm_features.csv"))
    metadata = pd.read_csv(os.path.join(metadata_path, "HAM10000_metadata"))

    # Apply the function to extract image_id
    color_var['image_id'] = color_var['file_name'].apply(extract_image_id)
    color_hist['image_id'] = color_hist['file_name'].apply(extract_image_id)
    lbp['image_id'] = lbp['file_name'].apply(extract_image_id)
    glcm['image_id'] = glcm['file_name'].apply(extract_image_id)

    # Sort feature DataFrames by image_id
    color_var_sorted = color_var.sort_values(by='image_id').reset_index(drop=True)
    color_hist_sorted = color_hist.sort_values(by='image_id').reset_index(drop=True)
    lbp_sorted = lbp.sort_values(by='image_id').reset_index(drop=True)
    glcm_sorted = glcm.sort_values(by='image_id').reset_index(drop=True)

    # Sort metadata by image_id
    metadata_sorted = metadata.sort_values(by='image_id').reset_index(drop=True)

    return color_var_sorted,color_hist_sorted,lbp_sorted,glcm_sorted,metadata_sorted

def merge_features():
    # Start with the first DataFrame
    dataset = load_csvs()
    merged = dataset[0]
    # Merge the rest
    for df in dataset[1:-1]:
        merged = pd.merge(merged, df, on='image_id', how='inner', suffixes=('', '_dup'))
        # Drop duplicate columns
        merged = merged.loc[:, ~merged.columns.str.endswith('_dup')]
    merged['dx'] = dataset[-1]['dx']
    return merged

def preprocessing():
    # Data Preprocessing
    # Encode labels
    SEED = 42
    np.random.seed(SEED)
    le = LabelEncoder()
    full_data = merge_features()
    full_data['label'] = le.fit_transform(full_data['dx'])

    # Define X and y (features and target)
    X = full_data.drop(columns=['label'])
    y = full_data['label']

    # Drop non-feature columns
    X_numeric = X.select_dtypes(include=['number'])

    # Output removed columns
    removed_cols = list(set(X.columns) - set(X_numeric.columns))
    print("removed_cols:", removed_cols)

    # Update X
    X = X_numeric

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=SEED
    )

    return X_test, y_test

def get_metrics(model, X, y):
    y_pred = model.predict(X)
    accuracy = f"{accuracy_score(y, y_pred):.4f}"
    precision = f"{precision_score(y, y_pred, average='weighted'):.4f}"
    return {"accuracy": accuracy, "precision": precision}