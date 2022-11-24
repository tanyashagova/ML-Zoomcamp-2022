import tensorflow.lite as tflite
import numpy as np
from io import BytesIO
from urllib import request

from PIL import Image

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

# print(tflite.__version__)

url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/df/Smaug_par_David_Demaret.jpg/1280px-Smaug_par_David_Demaret.jpg'

img = download_image(url)
x = prepare_image(img, target_size=(150, 150))
X = np.array([preprocess_input(x)])



interpreter = tflite.Interpreter(model_path='dino_dragon.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

interpreter.set_tensor(input_index, X)
interpreter.invoke()

pred = interpreter.get_tensor(output_index)

print(interpret_pred(pred))