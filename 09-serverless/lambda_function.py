import tflite_runtime.interpreter as tflite
import numpy as np
from io import BytesIO
from urllib import request

from PIL import Image

target_size=(150, 150)


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.Resampling.NEAREST)
    return img

def preprocess_input(x):
    return (np.array(x)/255).astype('float32')

def interpret_pred(pred):
    label = int(pred[0][0] >= 0.5)
    classes = ['dino', 'dragon']
    return classes[label]

def preprocessor(url, target_size=(150, 150)):
    img = download_image(url)
    x = prepare_image(img, target_size=target_size)
    X = np.array([preprocess_input(x)])
    return X



interpreter = tflite.Interpreter(model_path='dino-vs-dragon-v2.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


classes = ['dino', 'dragon']

# url = 'http://bit.ly/mlbookcamp-pants'
url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/df/Smaug_par_David_Demaret.jpg/1280px-Smaug_par_David_Demaret.jpg'
def predict(url):
    X = preprocessor(url, target_size=target_size)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    # float_predictions = preds[0].tolist()

    return {interpret_pred(preds): float(preds[0][0])}

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result