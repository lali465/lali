import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image
mobile=tf.keras.applications.mobilenet.MobileNet()
filename="C:/Users/lalit/OneDrive/Pictures/cat/cats_00001.jpg"
image=Image(filename, width=224, height=224)
image
from tensorflow.keras.preprocessing import image
img=image.load_img(filename,target_size=(224,224))
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input, MobileNet
from tensorflow.keras.applications.mobilenet import decode_predictions
import matplotlib.pyplot as plt
from tensorflow.keras.applications import imagenet_utils as iu
from tensorflow.keras.preprocessing import image
resized_img=image.img_to_array(img)
final_image=np.expand_dims(resized_img,axis=0)
final_image=tf.keras.applications.mobilenet.preprocess_input(final_image)
predictions=mobile.predict(final_image)
result=iu.decode_predictions(predictions)
result
mobile=tf.keras.applications.mobilenet_v2.MobileNetV2()   #gives better prediction
predictions=mobile.predict(final_image)
result=iu.decode_predictions(predictions)
result
