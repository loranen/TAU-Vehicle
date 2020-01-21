from os import listdir
from os import walk
from os import sep
from os import system
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

from datetime import datetime


def train_model(path, model, batch_size, epochs, name):

	checkpoint = ModelCheckpoint(datetime.now().strftime("%m_%d_%Y_%H_%M_%S")+"_"+name+".h5", monitor='loss', verbose=1,
    save_best_only=True, mode='auto', period=1)

#	train_datagen = ImageDataGenerator(
#	    rescale=1./255,
#	    width_shift_range=0.2,
#	    height_shift_range=0.2,
#   	zoom_range=0.2,
#    	horizontal_flip=True)

	train_datagen = ImageDataGenerator(
		rescale=1./255,
	    shear_range=0.2,
	    zoom_range=0.2,
	    horizontal_flip=True, 
	    rotation_range=20,
	    width_shift_range=0.2,
	    height_shift_range=0.2)

	test_datagen = ImageDataGenerator(rescale=1./255)

	train_generator = train_datagen.flow_from_directory(
	    path+'\\training',
	    target_size=(224, 224),
	    batch_size=batch_size)

	validation_generator = test_datagen.flow_from_directory(
	    path+'\\test',
	    target_size=(224, 224),
	    batch_size=batch_size,
	    class_mode='categorical')

	history = model.fit_generator(train_generator, 
		validation_data=validation_generator, 
		steps_per_epoch=np.ceil(22444 / batch_size), 
		epochs=epochs,
		callbacks=[checkpoint])
	
	return model, history



