# Drug Recognition

In healthcare, it is important to be able to identify drugs when serving patients,
distribute medication, and to be able to identify drugs in the case of an overdose.
This project used [TensorFlow](https://www.tensorflow.org/) for training a model to recognize pills.
Then, deploy the model to application that can be used everywhere in any device.

# Motivation
In work or daily life, it's common to accidentally drop or mix up pills, 
making it challenging to identify them. In my search online, 
I've explored various medication recognition apps, but none has fully met my needs.

# Table of contents
1. [Constraints](#constraints)
2. [For developers](#developers)
3. [Future work](#future-work)
4. [Contributing](#contributing)

# Constraints<a name="constraints"></a>
There is a limited dataset available online for training the model, 
but I have created the template model for the project. If additional 
datasets become available in the future, I will update the model accordingly.

# Go to the App <a name="predictApp"></a>
[Drug recognition app](https://drug-dockerfile.onrender.com/)

# For Developers<a name="developers"></a>

## Use the model
1. Clone the repository
```bash
git@github.com:gyes51y767p/drug_recognition.git
```
2. Install the required packages
```bash 
pip install -r requirements.txt
```
If you want to deploy the model automatically, you need to config the workflow file `mian.yaml`, replace the api key and id



# Future work<a name="future-work"></a>
Acquire additional data samples to enhance the accuracy of the model.

# 👏 Contributing<a name="contributing"></a>

Would love your help! Contribute or advises by raising issue and opening pull requests. 