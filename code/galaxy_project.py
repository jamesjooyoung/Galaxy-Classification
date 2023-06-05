"""This is the main python file for the project.

The project can be run either by running this file directly, or by importing
this file in the project's jupyter notebook.
"""

import numpy as np
import convolution_model
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
import visualkeras

def split_data(images, labels, split_ratio):
    """Splits data into training, validation, and test sets.

    Args:
        images: Numpy array of shape (17736, 256, 256, 3).
        labels: Numpy array of shape (17736,).
        split_ratio: List of length 3, whose elements are each an integer
            in the range [1, 99], representing the percentages of the data to
            use for training, validation, and testing, respectively.

    Returns:
        A 6-tuple of numpy arrays representing the training images, training 
        labels, validation images, validationlabels, testing images, and 
        testing labels, respectively.
    """

    print("Started splitting the data.")
    # Shuffle the data.
    rng = np.random.default_rng()
    permuted_indices = rng.permutation(len(labels))
    images = images[permuted_indices]
    labels = labels[permuted_indices]

    # Split the data in a way that maintains the proportions of each class in each split.
    our_test_valid_size = (split_ratio[1] + split_ratio[2]) * 0.01
    #print("valid test size = ", our_test_valid_size)
    split1 = StratifiedShuffleSplit(n_splits=1, test_size=our_test_valid_size, random_state=42)
    for train_index, test_valid_index in split1.split(images, labels):
        train_img, test_valid_img = images[train_index], images[test_valid_index]
        train_lab, test_valid_lab = labels[train_index], labels[test_valid_index]

    size_of_test_as_frac_of_test_valid = (split_ratio[2] / (split_ratio[1] + split_ratio[2]))
    #print("valid size as frac of test_valid = ", size_of_valid_as_frac_of_test_valid)
    split2 = StratifiedShuffleSplit(n_splits=1, test_size=size_of_test_as_frac_of_test_valid, random_state=42)
    for valid_index, test_index in split2.split(test_valid_img, test_valid_lab):
        valid_img, test_img = test_valid_img[valid_index], test_valid_img[test_index]
        valid_lab, test_lab = test_valid_lab[valid_index], test_valid_lab[test_index]

    orig_class_ids, orig_class_counts = np.unique(labels, return_counts=True)
    train_class_ids, train_class_counts = np.unique(train_lab, return_counts=True)
    test_class_ids, test_class_counts = np.unique(test_lab, return_counts=True)
    valid_class_ids, valid_class_counts = np.unique(valid_lab, return_counts=True)

    """
    num_orig_examples = images.shape[0]
    for orig_class_id, orig_class_count in enumerate(orig_class_counts):
        print(f"Class {orig_class_id} originally has {orig_class_count/num_orig_examples} fraction of examples.")
    num_valid_examples = valid_img.shape[0]
    for valid_class_id, valid_class_count in enumerate(valid_class_counts):
        print(f"Class {valid_class_id} originally has {valid_class_count/num_valid_examples} fraction of examples.")
    """

    print("Finished splitting the data.")
    return train_img, train_lab, valid_img, valid_lab, test_img, test_lab

def show_example_predictions(test_img, test_lab, predictions):
    # Get indices of first image for each class.
    first_img_indices = []
    for i in range(0, 10):
        this_class_starting_index = np.where(test_lab == i)[0][0]
        first_img_indices.append(this_class_starting_index)

    # Plot the corresponding images.
    fig, ax = plt.subplots(1, 10)
    fig.set_size_inches(12, 1.2)
    for i, each_image in enumerate(first_img_indices):
        ax[i].imshow(test_img[each_image].astype('uint8'))
        ax[i].tick_params(left=False)
        ax[i].tick_params(bottom=False)
        ax[i].tick_params(labelleft=False)
        ax[i].tick_params(labelbottom=False)
        ax[i].set_xlabel(f"Class {i}\nPrediction: {predictions[each_image]}")
    fig.suptitle("Example images from each class, along with their predictions.")
    plt.show()

def run_cnn_model(images, labels, multi=True, split_ratio=[70, 15, 15]):
    """Generates and trains a CNN model.

    Args:
        images: Numpy array of shape (17736, 256, 256, 3).
        labels: Numpy array of shape (17736,).
        split_ratio: List of length 3, whose elements are each an integer
            in the range [1, 99], representing the percentages of the data to
            use for training, validation, and testing, respectively. Defaults
            to [70, 15, 15].

    Returns:
        None
    """

    # Raise an error if the split_ratio values don't sum to 100.
    if split_ratio[0] + split_ratio[1] + split_ratio[2] != 100:
        raise AssertionError("Split ratio values don't sum to 100.")

    # Split the data into training, validation, and test sets.
    train_img, train_lab, valid_img, valid_lab, test_img, test_lab = split_data(
        images, labels, split_ratio)

    # Generate the multi-layer CNN model.
    if multi==True:
        cnn_model = convolution_model.get_multi_CNN_model()
    
    # Generate the single-layer CNN model
    else:
        cnn_model = convolution_model.get_single_CNN_model()
    

    # Train the CNN model.
    print("Starting model training.")
    history = cnn_model.model.fit(
        train_img, train_lab,
        epochs = cnn_model.epochs,
        batch_size = cnn_model.batch_size,
        validation_data = (valid_img, valid_lab),
    )

    # Test the CNN model.
    print("Starting model testing.")
    test_results = cnn_model.model.evaluate(test_img, test_lab, cnn_model.batch_size)
    print("Test loss, test accuracy: ", test_results)

    # Print summary of the CNN model.
    cnn_model.model.summary()

    # Print textual confusion matrix.
    predictions = np.argmax(cnn_model.model.predict(test_img), -1)
    confusion_mtx = tf.math.confusion_matrix(predictions, test_lab)
    print(confusion_mtx)

    # Show confusion matrix image.
    matrix = confusion_matrix(test_lab, predictions)
    sns.heatmap(matrix, annot=True, fmt='.0f')
    plt.title('Galaxy Confusion Matrix (Tensorflow model)')
    plt.xlabel('Predicted class')
    plt.ylabel('True class')
    plt.show()

    # Print some example predictions.
    show_example_predictions(test_img, test_lab, predictions)

    visualkeras.layered_view(cnn_model.model).show()