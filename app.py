import pickle
from flask import Flask, request, jsonify, app, render_template ,url_for,flash,redirect
from flask_wtf import FlaskForm
from flask_wtf.file import  FileRequired, FileAllowed

from wtforms import FileField ,SubmitField
from werkzeug.utils import secure_filename
import os


# import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
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




app = Flask(__name__)
app.config['SECRET_KEY'] = 'werwefwefwef'
app.config['UPLOAD_FOLDER'] = 'static/files'

loadedmodel=tf.keras.models.load_model('whole_model')
print(loadedmodel.summary())
if hasattr(loadedmodel, 'class_names'):
    class_names = loadedmodel.class_names
    print("Class Names:", class_names)
else:
    print("Class names not found. Check how the model was trained.")


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
    form = UploadFileForm()
    submission_successful = False
    prediction_message=""
    prediction = "this is the prediction"
    path= os.path.abspath('')

    print(path)
    if form.validate_on_submit():
        file = form.file.data


        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename)))
        success_message = "File has been uploaded successfully"
        submission_successful = True
        filepath=os.path.join(os.path.abspath(""),app.config['UPLOAD_FOLDER'], file.filename)
        print(filepath)


        img = tf.keras.utils.load_img(
            filepath, target_size=(180, 180)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        predictions = loadedmodel.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        prediction_message= "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))

        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
        )



    return render_template('home.html', form=form, submission_successful=submission_successful, prediction_message=prediction_message)


@app.route('/predict_api',methods=['POST'])
def predict_api():
    pass


if __name__ == '__main__':
    app.run(debug=True)
    print("hello")
