# Deep Neural Networks for Galaxy Morphology Classification

For our final project, we implemented both a single layer and a multi layer Convolutional Neural Network (CNN) model on the Galaxy10: DECaLS dataset to correctly classify images of 10 galaxies based on their shapes. We additionally built a DenseNet201 model as well as a comparison.

## Code Structure
The exploration.ipynb file contains the code for pulling the data, preprocessing the data, and running our single-layer and multi-layer CNN models, as well as creeating visualizations. The convolution_model.py file contains our model architectures for the single layer and multi layer models and contains some preprocessing and data augmentation. The galaxy_project.py file contained the necessary training, splitting, and testing functions to actually run our CNN models. The test.ipynb file contains the DenseNet201 model built by Bitao.

## Dataset

We used the Galaxy10 DECaLS data set, which contains 17,736 galaxy images (256x256 pixels, colored) from DECaLS, labeled (through Galaxy Zoo) according to 10 classes: Class 0 (1081 images): Disturbed Galaxies, Class 1 (1853 images): Merging Galaxies, Class 2 (2645 images): Round Smooth Galaxies, Class 3 (2027 images): In-between Round Smooth Galaxies, Class 4 ( 334 images): Cigar Shaped Smooth Galaxies, Class 5 (2043 images): Barred Spiral Galaxies, Class 6 (1829 images): Unbarred Tight Spiral Galaxies, Class 7 (2628 images): Unbarred Loose Spiral Galaxies, Class 8 (1423 images): Edge-on Galaxies without Bulge, Class 9 (1873 images): Edge-on Galaxies with Bulge.

## Preprocessing

The dataset was preprocessed by converting the images and labels into floats, adding center cropping to the images, resizing the images to 69x69 pixels, and one-hot encoding the labels. When training our model, we decided to add in Random Rotation and Random Flip layers as data augmentation to improve the performance of our model. Finally, we divided the 17,736 images from the data set into 70% in training, 15% in validation, and 15% in test sets.

## Model Building

We implemented a single layer Convolutional Neural Network (CNN), a multi-layer CNN and a DenseNet CNN as the primary architectures of our project. The single-layer CNN consisted of: 1) 2D Convolution, 2) Batch Normalization, 3) Max-Pooling, 4) Dropout, 5) Flatten, 6) Dropout, 7) Dense with LeakyRelu, 8) Dropout, 9) Dense with SoftMax. The multi-layer CNN architecture was very similar except that it used three 2D Convolution layers, with each layer followed by a Batch Normalization layer.DenseNet proposes two changes to the standard CNN model: a dense block that allows gradients to flow through, and a transition layer that improves model efficiency by reducing parameters As we intended to build completely new architectures to classify galaxy images, we tuned our hyperparameters (epochs and batch size) and selected filter sizes and dropout rates based on trial and error, as well as general shared knowledge.

## Evaluation | Accuracy

The evaluation metric used for our model architectures was accuracy. Test accuracy of the single layer convolutional neural network on the dataset was 0.6102. Test accuracy of the multi layer convolutional neural network on the dataset was 0.7662. Test accuracy of the DenseNet201 on the dataset was 0.9483.
