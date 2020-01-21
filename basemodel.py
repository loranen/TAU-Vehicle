import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D

def init_basemodel(modelname):


	if modelname == "MobileNet":
		base_model = keras.applications.mobilenet.MobileNet(input_shape = (224,224,3), include_top=False)
	elif modelname == "MobileNetV2":
		base_model = keras.applications.mobilenet_v2.MobileNetV2(input_shape = (224,224,3), include_top=False)
	elif modelname == "InceptionV3":
		base_model = keras.applications.inception_v3.InceptionV3(input_shape = (224,224,3), include_top=False)
	elif modelname == "Xception":
		base_model = keras.applications.xception.Xception(include_top=False, weights='imagenet', input_shape=(224,224,3), pooling='max')
	
	return base_model


def construct_and_compile_MobileNetV2(base_model):
	in_tensor = base_model.inputs[0]
	out_tensor = base_model.outputs[0]

	out_tensor = GlobalAveragePooling2D()(out_tensor)
	out_tensor = Dense(128, activation='relu')(out_tensor)
	out_tensor = Dense(17, activation='softmax')(out_tensor)

	# Define the full model by the endpoints.
	model = Model(inputs = [in_tensor], outputs = [out_tensor])

	#tf.keras.utils.plot_model(model)
	
	# Compile the model for execution. Losses and optimizers
	# can be anything here, since we don’t train the model.
	#model.compile(loss = "categorical_crossentropy", optimizer = 'sgd')
	model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adadelta(), metrics=['accuracy'])

	return model

def construct_and_compile_Xception(base_model):
    
	# freeze_layers(base_model, -10, 0)

	in_tensor = base_model.inputs[0]
	out_tensor = base_model.outputs[0]
	
	out_tensor = Flatten()(out_tensor)
	out_tensor = Dense(128, activation='relu')(out_tensor)
	
	out_tensor = Dense(17, activation='softmax')(out_tensor)

	# Define the full model by the endpoints.
	model = Model(inputs = [in_tensor], outputs = [out_tensor])

	#tf.keras.utils.plot_model(model)
	
	# Compile the model for execution. Losses and optimizers
	# can be anything here, since we don’t train the model.
	#model.compile(loss = "categorical_crossentropy", optimizer = 'sgd')
	model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adadelta(), metrics=['accuracy'])
	return model

def construct_and_compile_InceptionV3(base_model):
    
	in_tensor = base_model.inputs[0]
	out_tensor = base_model.outputs[0]

	out_tensor = GlobalAveragePooling2D()(out_tensor)
	out_tensor = Dense(128, activation='relu')(out_tensor)
	out_tensor = Dense(17, activation='softmax')(out_tensor)

	# Define the full model by the endpoints.
	model = Model(inputs = [in_tensor], outputs = [out_tensor])

	#tf.keras.utils.plot_model(model)
	
	# Compile the model for execution. Losses and optimizers
	# can be anything here, since we don’t train the model.
	#model.compile(loss = "categorical_crossentropy", optimizer = 'sgd')
	model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adadelta(), metrics=['accuracy'])

	return model

def freeze_layers(model, from_layer, to_layer, freeze = False):
	# Freeze layers in model
	for layer in model.layers[from_layer:to_layer]:
		layer.trainable = freeze
	return model
