from os import listdir
from os import walk
from os import sep
from os import system
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow import keras


def img_generator(path):
	count = 0;
	# Find all image files in the data directory.
	#y = [] # Class ids will go here.
	#filenames = []
	for root, dirs, files in walk(path):
		for name in files:
			if name.endswith(".jpg"):
				#filenames.append(name)
				# Load the image:
				img = plt.imread(root + sep + name)
				# Resize it to the net input size:
				img = cv2.resize(img, (224,224))
				# Convert the data to float, and remove mean:
				img = img.astype(np.float32)
				img -= 128

				yield [img, root, name]

def training_to_validation(path, target, ratio):
	count = 0;
	# Find all image files in the data directory.
	#y = [] # Class ids will go here.
	#filenames = []
	for root, dirs, files in walk(path):
		for name in files:
			if name.endswith(".jpg"):
				None

				
def training_generator(path):
	class_names = sorted(listdir(path+r'\\train'))
	for root, dirs, files in walk(path):
		for name in files:
			if name.endswith(".jpg"):
				label = root.split(sep)[-1]
				y = keras.utils.to_categorical(class_names.index(label), len(class_names))
				#filenames.append(name)
				# Load the image:
				img = plt.imread(root + sep + name)
				# Resize it to the net input size:
				img = cv2.resize(img, (224,224))
				# Convert the data to float, and remove mean:
				img = img.astype(np.float32)
				img -= 128
				img = img[np.newaxis,...]
				print(np.shape(img))
				yield ({'input_1': img}, {'output': y})


def extract_features_for_training(path, model):
	calculator = 0;
	class_names = sorted(listdir(path+r'\\train'))
	# Find all image files in the data directory.
	X = [] # Feature vectors will go here.
	y = [] # Class ids will go here.
	filenames = []
	for data in img_generator(path):
			img = data[0]
			root = data[1]
			filenames.append(data[2])

			x = model.predict(img[np.newaxis, ...])[0]
			X.append(x)
				
			# Extract class name from the directory name:
			label = root.split(sep)[-1]
			y.append(class_names.index(label))
			calculator = calculator+1
			print("Picturecount: "+str(calculator))
			system('cls')

	return [X, y, class_names, filenames]



def extract_features(path, model):
	calculator = 0;
	# Find all image files in the data directory.
	X = [] # Feature vectors will go here.
	filenames = []
	for root, dirs, files in walk(path):
		for name in files:
			if name.endswith(".jpg"):
				filenames.append(name)
				# Load the image:
				img = plt.imread(root + sep + name)
				# Resize it to the net input size:
				img = cv2.resize(img, (224,224))
				# Convert the data to float, and remove mean:
				img = img.astype(np.float32)
				img -= 128
				# Push the data through the model:
				x = model.predict(img[np.newaxis, ...])[0]
				# And append the feature vector to our list.
				X.append(x)
				
				calculator = calculator+1
				#print("Picturecount: "+str(calculator))
				#system('cls')

	return [X, filenames]