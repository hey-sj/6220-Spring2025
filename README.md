# Classifying Skin Lesions from HAM10000 dataset

### Contributers
Siting Liang, Grace Chong, Jicheng Li, Teng Liu, Qingyang Guo

### Project Structure
```
├── server.py                # Main Flask application
├── requirements.txt         # Python dependencies
├── models_performance.ipynb # Summary notebook of model performance
├── templates/               # 
├── server_utils             # 
│   ├── image_upload         # 
│   └── modules              # 
├── model_training/          # Saved models and notebook for model training
├── dataset/                 # 
│   ├── data_exploration     # 
│   └── dataverse_files      #  
└── README.md                # Project documentation
```

### Overview
This project presents a machine learning approach to classify skin cancer types using the HAM10000 dataset, a large collection of dermatoscopic images containing 10,015 labeled skin lesion samples. The dataset encompasses seven diagnostic categories, including melanoma, basal cell carcinoma, and vascular lesions. Our goal is to evaluate the effectiveness of traditional machine learning models vs. deep learning models in distinguishing between these conditions.

We conducted thorough data exploration, extracting both color-based and texture-based features. Support Vector Machine (SVM), Random Forest (RF), Bagging, and Ensemble models were trained using an 80/20 train-test split. Hyperparameter tuning was performed for the SVM and RF models via RandomizedSearchCV. We employed transfer learning on Convolutional Neural Networks (CNNs) utilizing the VGG16 and MobileNetV2 architectures. Tuning was performed on these models using Keras’ Hyperband Tuner. 
Our findings suggest that classical models, when combined with handcrafted features, can potentially serve as effective baselines for image classification tasks in healthcare. Our traditional algorithms performed better than the CNNs, which highlights the importance and impact of feature extraction and engineering in model performance. We also see that classifying skin lesions is a complex task, with the models consistently classifying some classes better than others. Future work may include additional feature extraction and engineering with the input of medical and imaging professionals; multi-modal learning; adding explainability and interpretability; and deployment of a comprehensive user-interface via a web interface.

### Dataset

We utilized the HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. We additionally utilized a HuggingFace library hosting the raw image data for our neural networks [found here](https://huggingface.co/datasets/marmal88/skin_cancer).

Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset. Scientific Data, 5, 180161. https://doi.org/10.1038/sdata.2018.161

### Installation

1. Clone the repository
2. Navigate to project directory
3. Install dependencies:
    `pip install -r requirements.txt`

### Running the Application

1. Start the Flask server:
    `python3 server.py`
2. Access the application in your browser:
    `http://localhost:8000`
