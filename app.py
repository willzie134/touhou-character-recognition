from flask import Flask, render_template, request, redirect, url_for
import os
import uuid
import cv2
from utils.inference import predict_faces_in_image, annotate_image, load_models

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load models once on startup
svm_model, lda_model, scaler, label_names = load_models()

@app.route('/', methods=['GET', 'POST'])
def index():

    os.makedirs(app.congfig['UPLOAD_FOLDER'], exist_ok=True)

    for fname in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        if os.path.isfile(file_path):
            os.remove(file_path)

    if request.method == 'POST':
        file = request.files['image']
        threshold = float(request.form.get('threshold', 0.5))

        if file:
            filename = f"{uuid.uuid4().hex}.jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img_cv = cv2.imread(filepath)
            results = predict_faces_in_image(img_cv, svm_model, lda_model, scaler, label_names, threshold=threshold)

            # Annotate and save
            annotated = annotate_image(img_cv, results)
            outpath = os.path.join(app.config['UPLOAD_FOLDER'], f"annotated_{filename}")
            cv2.imwrite(outpath, annotated)

            return render_template('index.html', image_file=url_for('static', filename=f"uploads/annotated_{filename}"), predictions=results)
            
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)