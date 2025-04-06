from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import server_utils.modules as su
import gzip

UPLOAD_DIR = "server_utils/image_upload"
models=None
le=None

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
    # ensemble_model_path = 'model_training/best_ensemble_model.pkl.gz'
    # with gzip.open(ensemble_model_path, 'rb') as f:
    #     ensemble_model = joblib.load(f)
    # models['Ensemble'] = ensemble_model

    le = joblib.load("model_training/label_encoder.pkl");

@app.route("/")
def home():
    if(models is None):
        load_model()
    return render_template("home.html")

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
            predictions[model_name] = le.inverse_transform(model.predict(df)).tolist();
        predictions["message"] = "Prediction below!"
        return jsonify(predictions)
    else:
        res = jsonify({"message": "Uploading error!"})
        res.status_code = 500
        return res
if(__name__ == "__main__") :
    app.run(port=8000, debug=True)