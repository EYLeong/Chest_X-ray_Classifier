import matplotlib.pyplot as plt
import numpy as np

def generate_graph(epoch_list, train_loss, validate_loss, model):
	'''
	This function is to generate the learning curves for the training phase
	'''
	fig = plt.figure()
	plt.plot(epoch_list, train_loss, label = "Training Loss")
	plt.plot(epoch_list, validate_loss, label = "Validation Loss")
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title('Train and Validation Loss Plot over training/testing steps')
	plt.legend()

	if model == "fl":
		fig.savefig('./saved_graphs/fl_learning_curve.png')
	elif model == "sl":
		fig.savefig('./saved_graphs/sl_learning_curve.png')

def generate_performance_display(val_loader, pred, labels):
	'''
	This function is to display the performance of our trained model on the 24 images in the validation set.
	'''

	# Initialize figure and other required variables
	plt.figure(figsize = (10, 10))
	num_images = len(val_loader.dataset)
	validation_data = np.zeros([num_images,150,150])
	validation_groundtruth = np.zeros([num_images,3])

	validation_pred = np.zeros([num_images,3])

	for data, target in val_loader:

		for i in range(num_images):
			validation_data[i] = data[i][0].to("cpu").numpy()
			validation_groundtruth[i] = target[i].to("cpu").numpy()

			validation_pred[i] = pred[i].to("cpu").numpy()

		for i in range(num_images):
			plt.subplot(5,5, i+1)
			plt.imshow(validation_data[i], cmap='gray', interpolation='none')
			plt.title("Label: " + labels[tensor_to_label(validation_groundtruth[i])] + "\n" + "Predicted: " + labels[tensor_to_label(validation_pred[i])])
			plt.xticks([])
			plt.yticks([])

	plt.tight_layout()
	plt.show()

def tensor_to_label(tensor):
	'''
	Turns output of form [1. 0. 0.] into an appropriate 0,1,2 value in order to be used by hashmap
	'''
	val = None
	for i in range(3):
		if tensor[i] == 1.0:
			val = i

	return val

