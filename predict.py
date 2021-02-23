# predict.py
# Import library
import argparse
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
import json

def Main():
    # Create the parser
    parser = argparse.ArgumentParser(description='Predict the name of the flower\'s image')
    
    #add the parsers
    parser.add_argument("path", type=str, help="The path where flowers are example ./test_images/orchid.jpg")
    parser.add_argument("model", type=str, help="The name of the model")
    parser.add_argument("--top_k", type=int, help="the last top K probs", required = False, default = 1)
    parser.add_argument("--category_names", type=str, help="the name of the json file with the key pair names", required = False)
    args = parser.parse_args()
    #invoke the predict function
    predict(args.path, args.model, args.top_k, args.category_names)
    
def predict(image_path, model, top_k, json_file):
    path_keras_model = model
    # TODO: Load the Keras model
    reloaded_keras_model = tf.keras.models.load_model('./'+ path_keras_model, custom_objects={'KerasLayer': hub.KerasLayer},compile=False)
        
    im = Image.open(image_path)
    image = np.asarray(im)
    image = process_image(image)
    image = np.expand_dims(image,axis=0)
    prediction = reloaded_keras_model.predict(image)
    probs, classes = tf.math.top_k(prediction, int(top_k))
    classes +=1
    
    probs = probs.numpy().squeeze()
    classes = classes.numpy().squeeze()
    
    class_names = []
    
    if int(top_k) == 1:
        classes = [classes]
        probs = [probs]
    
    print("the image {} match with the flower: ".format(image_path))
    if (json_file != None):
        with open('./' + json_file, 'r') as f:
            class_names = json.load(f)
        flowers_names = [class_names[str(x)] for x in classes]
        for i in range(len(probs)):
            print("{}, probability is {} ".format(flowers_names[i], probs[i]))
    else:
        for i in range(len(probs)):
            print("Class ID: {}, probability: {} ".format(classes[i], probs[i]))
    
        
def process_image(img):
    image = tf.convert_to_tensor(img) #convert to tensor
    image = tf.image.resize(image, (224, 224)) #resize to 224x224
    image = (image/255) #normalize
    image = image.numpy() #convert to numpy
    return image



    
if __name__ == '__main__':
    Main()

#python predict.py ./test_images/cautleya_spicata.jpg My_Keras_Model.h5 --top_k 5 --category_names label_map.json
