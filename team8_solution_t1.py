import init_basemodel
import extract_features
import submission_output
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score as scorer
from sklearn.metrics import accuracy_score as acc_test

def test_classifiers():

	base_model = init_basemodel.init_basemodel()

	data = extract_features.extract_features_for_training(r'.\\train\\', base_model)


	# Cast the python lists to a numpy array.
	X = np.array(data[0])
	y = np.array(data[1])
	class_names = data[2]

	print(np.shape(data))
	print(np.shape(X))
	print(np.shape(y))

	print(y[1])

	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

	print(np.shape(x_train))
	print(np.shape(x_test))
	print(np.shape(y_train))
	print(np.shape(y_test))

	classifiers = [LDA(), SVC(kernel='linear'), SVC(), LR(), RFC()]
	results = []

	for classifier in classifiers:
		classifier.fit(x_train, y_train)
		prediction = classifier.predict(x_test)

		results.append(acc_test(y_test, prediction))

	print(results)

	return [classifiers, class_names]

def classify_testdata(classifier, class_names):
	
	base_model = init_basemodel.init_basemodel()

	data = extract_features.extract_features(r'.\\test\\', base_model)


	# Cast the python lists to a numpy array.
	X = np.array(data[0])
	filenames = data[1]

	results = []

	predictions = classifier.predict(X)

	submission_output.output_submission(predictions, filenames, class_names)


if __name__ == "__main__":

	training = test_classifiers()

	classify_testdata(training[0][2], training[1])