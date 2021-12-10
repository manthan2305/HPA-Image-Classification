import numpy as np
import cv2
import io
from datetime import datetime
from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications.densenet import preprocess_input
from flask import Flask, jsonify, request, render_template
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/hpa'

# setup database
db = SQLAlchemy(app)

class Inference(db.Model):
    sno = db.Column(db.Integer, primary_key=True)
    id = db.Column(db.String(255), nullable=False)
    target = db.Column(db.String(255), nullable=False)
    date = db.Column(db.String(12), nullable=True)

# Load checkpoint
model = keras.models.load_model('saved_model/model_4.h5')

class_names = ['Nucleoplasm', 'Cytosol', 'Plasma membrane',
                'Nucleoli', 'Mitochondria', 'Golgi apparatus',
                'Nuclear bodies', 'Nuclear spackles', 'Nucleoli fibrillar center']

def preprocess_image(image_bytes):
    image = np.array(Image.open(io.BytesIO(image_bytes)))
    image = preprocess_input(image.astype(np.float32))
    image = cv2.resize(image, (224, 224), cv2.INTER_AREA)
    image = np.expand_dims(image, axis = 0)
    return image

def get_predictions(image_bytes):
    labels = []
    image = preprocess_image(image_bytes)
    pred = model.predict(image)
    for i in range(9):
        if pred[0][i] > 0.1:
            labels.append(class_names[i])
    return labels

@app.route('/', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return
        image_bytes = file.read()
        predictions = get_predictions(image_bytes)

        # Database entry
        entry = Inference(id = file, target = str(predictions), date = datetime.now())
        db.session.add(entry)
        db.session.commit()
        return render_template('result.html', name = predictions)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
    


