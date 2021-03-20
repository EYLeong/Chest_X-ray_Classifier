import matplotlib.pyplot as plt
import numpy as np
import torch

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
		fig.savefig('./saved_graphs/fl_learning_curve_augmented.png')
	elif model == "sl":
		fig.savefig('./saved_graphs/sl_learning_curve_augmented.png')

def generate_performance_display(val_loader, pred, labels):
    '''
    This function is to display the performance of our trained model on the 24 images in the validation set.
    '''

    # Initialize figure and other required variables
    fig = plt.figure(figsize = (10, 10))
    num_images = len(val_loader.dataset)
    validation_data = np.zeros([num_images,150,150])
    validation_groundtruth = np.zeros([num_images,3])

    validation_pred = np.zeros([num_images,3])

    correct = 0

    data = torch.empty(0, 1, 150, 150)
    target = torch.empty(0, 3)

    for temp_data, temp_target in val_loader:
        data = torch.cat((data, temp_data), 0)
        target = torch.cat((target, temp_target), 0)

    for i in range(num_images):
        validation_data[i] = data[i][0].to("cpu").numpy()
        validation_groundtruth[i] = target[i].data.to("cpu").numpy()
        validation_pred[i] = pred[i].to("cpu").numpy()

        if labels[tensor_to_label(validation_groundtruth[i])] == labels[tensor_to_label(validation_pred[i])]:
            correct += 1

    for i in range(num_images):
        plt.subplot(5,5, i+1)
        plt.imshow(validation_data[i], cmap='gray', interpolation='none')
        plt.title("Label: " + labels[tensor_to_label(validation_groundtruth[i])] + "\n" + "Predicted: " + labels[tensor_to_label(validation_pred[i])])
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.suptitle("Validation Set Pictures, with Predicted and Ground Truth Labels \n Average Performance {}/{} = {:.2f}%".format(correct,num_images, (correct/num_images)*100))
    plt.subplots_adjust(top=0.88)
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

