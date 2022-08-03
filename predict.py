import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import time
from PIL import Image
import argparse
import json
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Ignore some warnings that are not relevant (you can remove this if you prefer)
import warnings
warnings.filterwarnings('ignore')

# create parser
parser=argparse.ArgumentParser(description='Command Line Application for my first AI application')

# add argument inside the parser
parser.add_argument('input', type= str, action= 'store', help= 'Path to the image', metavar= '')
parser.add_argument('model', type= str,action= 'store',help='Path to the trained model classifier', metavar='')
parser.add_argument('--top_k', default=5, type= int, action= 'store',help= 'define how many top classes is needed',metavar='')
parser.add_argument('--category_names', default= './label_map.json',type= str,action= 'store',help= 'json for for labeling the images',metavar='')

args=parser.parse_args()
top_k=args.top_k

def process_image(image):
    test_image = tf.cast(image, tf.float32)
    test_image = tf.image.resize(test_image, (224,224))
    test_image /= 255
    test_image=test_image.numpy()
    return test_image;

def predict(image_path, model, top_k=5):
#image_path='./test_images/cautleya_spicata.jpg'
    #top_k=5
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_image=process_image(test_image)
    processed_image=np.expand_dims(processed_image, axis=0)
        #ps = model.predict(processed_image)
    ps = model.predict(processed_image)
    result = tf.math.top_k(ps,k=top_k)
    probs=result.values.numpy()[0].flatten()
    classes=result.indices.numpy()[0].flatten()
    classes_list=[]
    probs_list=[]
    for i in range(top_k):
        classes_list.append(classes[i])
        probs_list.append(probs[i])
        

    return probs_list,classes_list

with open(args.category_names,'r') as file:
    mapping=json.load(file)
    
loaded_model= tf.keras.models.load_model(args.model,custom_objects={'KerasLayer': hub.KerasLayer},compile = False)
print('Top {} Classes: '.format(top_k))
probs,labels=predict(args.input, loaded_model, top_k)

for prob, label in zip(probs,labels):
    print('Label: ',label)
    print('Class name: ', mapping[str(label+1)].title())
    print('Probability: ',prob)


