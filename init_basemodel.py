import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

def init_basemodel():
	base_model = keras.applications.mobilenet.MobileNet(input_shape = (224,224,3), include_top=False)

	tf.keras.utils.plot_model(base_model)

	in_tensor = base_model.inputs[0]
	out_tensor = base_model.outputs[0]

	out_tensor = tf.keras.layers.GlobalAveragePooling2D()(out_tensor)

	# Define the full model by the endpoints.
	model = tf.keras.models.Model(inputs = [in_tensor],
	outputs = [out_tensor])

	# Compile the model for execution. Losses and optimizers
	# can be anything here, since we don’t train the model.
	model.compile(loss = "categorical_crossentropy", optimizer = 'sgd')

	return model