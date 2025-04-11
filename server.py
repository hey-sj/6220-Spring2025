from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import server_utils.modules as su
import gzip

UPLOAD_DIR = "server_utils/image_upload"
models=None
le=None
disease_map = {
    "nv": "melanocytic nevi",
    "mel": "melanoma",
    "bkl": "seborrheic keratoses/lichen-planus like keratoses",
    "bcc": "basal cell carcinoma",
    "akiec": "actinic keratoses and intraepithelial carcinoma / Bowen's disease",
    "vasc": "vascular legions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage)",
    "df": "dermatofibroma"
}

metrics_obj=None

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR

def load_model():
    global models, le
    models = {
        'SVM': joblib.load('model_training/best_svm_model.pkl'),
        'RandomForest': joblib.load('model_training/best_rf_model.pkl'),
        'XGBoost': joblib.load('model_training/xgboost_model.pkl'),
        'Bagging': joblib.load('model_training/bagging_dtc_model.pkl'),
    }

    # Decompress Ensemble Model
    ensemble_model_path = 'model_training/best_ensemble_model.pkl.gz'
    with gzip.open(ensemble_model_path, 'rb') as f:
        ensemble_model = joblib.load(f)
    models['Ensemble'] = ensemble_model

    le = joblib.load("model_training/label_encoder.pkl");

def set_metric_obj():
    global metrics_obj
    X, y = su.preprocessing()
    metrics_obj = {}
    for model_name, model in models.items():
        metrics_obj[model_name] = su.get_metrics(model, X, y)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/load_models")
def load_models():
    try:
        if(models is None):
            load_model()
        if(metrics_obj is None):
            set_metric_obj()
        return "Success"
    except Exception as e:
        print(e)
        res = jsonify({"msg": "uh-oh"})
        res.status = 500
        return res

@app.route("/test", methods=['GET', 'POST'])
def test():
    if(models is None):
        load_model()
    if request.method == "POST":
        reqBody = request.get_json()
        return jsonify({
            "message": f"Hello there, {reqBody['name']}", 
        })
    else:
        return jsonify({"message": f"Hello there, {reqBody['name']}"})
        

@app.route("/proc_upload", methods=["POST"])
def proc_upload():
    if(models is None):
        load_model()
    request_file = request.files["image"]
    if(request_file):
        su.save_file(request_file, app.config["UPLOAD_FOLDER"], request_file.filename)
        img_features = su.proc_image(app.config["UPLOAD_FOLDER"], request_file.filename)
        df = su.get_df(img_features)
        predictions = {}
        for model_name, model in models.items():
            prediction = le.inverse_transform(model.predict(df)).tolist()[0];
            predictions[model_name] = disease_map[prediction]
        predictions["message"] = "Report below!"
        return jsonify(predictions)
    else:
        res = jsonify({"message": "Uploading error!"})
        res.status_code = 500
        return res
    
@app.route("/metrics", methods=["GET"])
def metrics():
    global metrics_obj
    if(models is None):
        load_model()
    if(metrics_obj is None):
       set_metric_obj()
    return jsonify(metrics_obj)

if(__name__ == "__main__") :
    app.run(port=8000, debug=True)