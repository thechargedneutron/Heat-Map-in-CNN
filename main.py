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

clf = keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
preds = clf.predict(X)

for box_size in range(3, X.shape[0] + 1):
	for i in range(0, X.shape[0] - box_size + 1):
		for j in range(0, X.shape[1] - box_size + 1):
			print(i, j)
			X_temp = return_padded_array(X, (box_size, box_size), (i, j))
			preds = clf.predict(X_temp)
			prob = 0.1 #TO BE UPDATED
			Prob_matrix = add_probabilities(Prob_matrix, (box_size, box_size), (i, j), prob)

print('Predicted:', decode_predictions(preds, top=3)[0])
