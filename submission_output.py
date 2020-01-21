from PIL import Image
import glob
from tensorflow.keras.models import load_model


def output_submission(predictions, filenames, classnames):
	
	with open("submission.csv", "w+") as fp:
		fp.write("Id,Category\n")
		for i, prediction in enumerate(predictions): 

			label = classnames[prediction]
			fp.write("%d,%s\n" % (int(filenames[i].split(".")[0]), label))

def predict_test_data(model):
	
	mobile = load_model(r'C:\Users\Leevi\Documents\GitHub\TAU-Vehicle\01_08_2020_01_20_11_MobileNetV2.h5')
	inception = load_model(r'C:\Users\Leevi\Documents\GitHub\TAU-Vehicle\01_21_2020_10_50_22_InceptionV3.h5')
	xception = load_model(r'C:\Users\Leevi\Documents\GitHub\TAU-Vehicle\01_20_2020_22_05_51_Xception.h5')

	image_list = []
	for filename in glob.glob(r'C:\Users\Leevi\Documents\TAU_vehicle\test\testset.jpg'):
	    im=Image.open(filename)
	    image_list.append(im)
		
	predictions_mobile = mobile.predict(image_list)
	predictions_inception = inception.predict(image_list)
	predictions_xception = xception.predict(image_list)
	
	
