from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import base64

app = Flask(__name__)

_model = tf.keras.models.load_model('model/myModel.h5')
_classes = ['Beetle', 'Butterfly', 'Cat', 'Cow', 'Dog', 'Elephant', 'Gorilla', 'Hippo', 'Lizard', 'Monkey', 'Mouse', 'Panda', 'Spider', 'Tiger', 'Zebra']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/prediction', methods=['GET','POST'])
def klasifikasi():
    img_uploaded = None
    probability = None 
    class_pred = None

    if request.method=='POST':
        file = request.files['file']

        img_bytes = file.read()
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [224, 224])
        img = np.expand_dims(img, axis=0)
        
        animals_pred = _model.predict(img, verbose=0)
        index = np.argmax(animals_pred[0])
        probability= round(animals_pred[0][index]*100, 2)
        class_pred = _classes[index]
        print(class_pred)
        print(probability)

        # Display image
        img_uploaded = base64.b64encode(img_bytes).decode('utf-8')

    return render_template('klasifikasi.html', img_uploaded=img_uploaded, probability=probability, class_pred=class_pred)




if __name__ == '__main__':
    app.run(debug=True)