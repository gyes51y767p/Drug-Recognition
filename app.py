import pickle
from flask import Flask, request, jsonify, render_template ,url_for,flash,redirect
from flask_wtf import FlaskForm
from flask_wtf.file import  FileRequired, FileAllowed

from wtforms import FileField ,SubmitField
from werkzeug.utils import secure_filename
import os

import numpy as np
import tensorflow as tf

# import matplotlib.pyplot as plt
# from tensorflow import keras
# from tensorflow.keras import layers ,regularizers
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras import  models, callbacks
# from tensorflow.keras.models import Sequential

data_dir = './drug_images'
batch_size = 3
img_height = 180
img_width = 180

(train_ds,val_ds) = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  labels='inferred',
  label_mode='int',
  class_names=None,
  color_mode='rgb',
  batch_size=2,
  image_size=(img_height, img_width),
  shuffle=True,
  seed=123,
  validation_split=0.2,
  subset="both",
  interpolation='bilinear',
  follow_links=False,
  crop_to_aspect_ratio=False)

class_names = train_ds.class_names
# loadedmodel=tf.keras.models.load_model('whole_model')

TF_MODEL_FILE_PATH = 'model2.tflite' # The default path to the saved TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path=TF_MODEL_FILE_PATH)
classify_lite = interpreter.get_signature_runner('serving_default')



app = Flask(__name__)
app.config['SECRET_KEY'] = 'werwefwefwef'
app.config['UPLOAD_FOLDER'] = 'static/files'




class UploadFileForm(FlaskForm):
    # file2 = FileField('file', validators=[FileRequired(), FileAllowed(['jpg', 'jpeg', 'pdf'])])
    file = FileField('file', validators=[FileRequired()], render_kw={"placeholder": "And how about here file"})
        #     ('what is this for', validators=[
        # FileRequired(),
        # FileAllowed(['jpg', 'jpeg', 'pdf'])],
        # render_kw={"placeholder": "and how about here ile"})

    submit = SubmitField('lets see')


@app.route('/',methods=['GET','POST'])
@app.route('/home',methods=['GET','POST'])
def home():
    return render_template('home.html')


@app.route('/predict',methods=['POST','GET'])
def predict():
    form = UploadFileForm()
    submission_successful = False
    prediction_message = ""
    prediction = "this is the prediction"
    path = os.path.abspath('')


    if form.validate_on_submit():
        file = form.file.data
        # success_message = "File has been uploaded successfully"
        target_directory = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'])
        os.makedirs(target_directory, exist_ok=True)
        file.save(os.path.join(target_directory, secure_filename(file.filename)))

        # file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],secure_filename(file.filename)))
        print(f"the file save to : {os.path.join(target_directory, secure_filename(file.filename))}", flush=True)
        submission_successful = True


        filepath = os.path.join(os.path.abspath(""), app.config['UPLOAD_FOLDER'], file.filename)
        print(f"now the filepath is : {filepath}", flush=True)
        print("------________----")
        print(filepath)
        print("-----_________-----")


        img = tf.keras.utils.load_img(
            filepath, target_size=(180, 180)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        predictions_lite = classify_lite(input_2=img_array)['dense_1']
        score_lite = tf.nn.softmax(predictions_lite)

        prediction_message = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(
            class_names[np.argmax(score_lite)], 100 * np.max(score_lite))
        print(f"this is the precdioctioin result : {prediction_message}")

    return render_template('predict.html', form=form, submission_successful=submission_successful,
                           prediction_message=prediction_message)

#srv-cn8kjn7109ks739pe7n0
#srv-cn8kjn7109ks739pe7n0
#          rnd_4znDVOg1TI2zckYM4WBuLtg8T6Yo
if __name__ == '__main__':
    app.run(debug=False)
