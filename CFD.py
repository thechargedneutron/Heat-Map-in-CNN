import keras
import numpy as np
import cv2
from keras.preprocessing import image
from keras import backend as K
from keras.applications.resnet50 import preprocess_input, decode_predictions

img_path = 'dog.jpg'
img = image.load_img(img_path, target_size=(224, 224))
X = image.img_to_array(img)
X = np.expand_dims(X, axis=0)
#X = preprocess_input(X)

a = np.zeros(224*224)
print(a)

clf = keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

inp = clf.input
outputs = [layer.output for layer in clf.layers]
functor = K.function([inp] + [K.learning_phase()], outputs)

test = np.random.random((224, 224, 3))[np.newaxis, :]
layer_outs = functor([test, 1.])
print(type(layer_outs[30]))
print(np.squeeze(layer_outs[30]).shape)
cv2.imshow('Image', np.squeeze(layer_outs[30]))
cv2.waitKey(5000)

preds = clf.predict(X)

print('Predicted:', decode_predictions(preds, top=3)[0])
