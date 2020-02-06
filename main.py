import basemodel
import network_training
import submission_output
import numpy as np
from matplotlib import pyplot as plt


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score as scorer
from sklearn.metrics import accuracy_score as acc_test
from split_data import split_dataset_into_test_and_train_sets
from os import listdir
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def classify_testdata(model, batchsize, test_dir):

	class_names = sorted(listdir(r'.\\training'))

	test_datagen = ImageDataGenerator(rescale=1./255)

	test_generator = test_datagen.flow_from_directory(
	        test_dir,
	        target_size=(224, 224),
	        color_mode="rgb",
	        shuffle = False,
	        class_mode='categorical',
	        batch_size=batchsize)

	filenames = test_generator.filenames
	nb_samples = len(filenames)

	predictions = model.predict_generator(test_generator,steps = 
                                   np.ceil(nb_samples/batchsize))

	predictions = predictions.tolist()
	prediction_table = []

	#print(np.shape(predictions))
	#print(filenames)

	for i, filename in enumerate(filenames):
		filenames[i] = filename.split('\\')[1]
	
	for p in predictions:
		prediction_table.append(p.index(max(p)))

	print(predictions[1])
	print(np.shape(prediction_table))
	print(prediction_table[1])
	print(filenames[1])

	submission_output.output_submission(prediction_table, test_generator.filenames, class_names)

def evaluate_base_models(path):
	models = ["MobileNet", "MobileNetV2", "InceptionV3", "Xception"]
	
	for base in models:
		base_model = basemodel.init_basemodel_param(base)
		model = basemodel.construct_and_compile(base_model)
		trained_model = network_training.train_model(path, model, base)


def draw_accuracy(history):
	# Draw accuracy over epochs
	# History is output from model.fit()

	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	# summarize history for loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()


if __name__ == "__main__":
	

	name = "InceptionV3"

	base_model = basemodel.init_basemodel(name)
	
	model = basemodel.construct_and_compile_InceptionV3(base_model)
	# model = load_model(r'C:\Users\Leevi\Documents\GitHub\TAU-Vehicle\01_20_2020_14_07_43_Xception.h5')
	
	model.summary()
	path = r'C:\Users\Leevi\Documents\TAU_vehicle\TAU_train_data_split'

	trained_model, history = network_training.train_model(path, model, 30, 50, name)
	draw_accuracy(history)

	#model = load_model("12_04_2019_08_42_12_Xception_082.h5")

	# classify_testdata(model, 5, r'.\\test')