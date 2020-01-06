def output_submission(predictions, filenames, classnames):
	with open("submission.csv", "w+") as fp:
		fp.write("Id,Category\n")
		for i, prediction in enumerate(predictions): 

			label = classnames[prediction]
			fp.write("%d,%s\n" % (int(filenames[i].split(".")[0]), label))
