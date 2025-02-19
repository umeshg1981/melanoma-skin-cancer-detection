# Melanoma Skin Cancer Detection
The goal is to create a CNN-based model capable of accurately detecting melanoma, a deadly form of skin cancer responsible for 75% of skin cancer deaths. This automated image analysis tool will aid dermatologists in early diagnosis, potentially saving lives and significantly reducing manual diagnostic effort.


## Table of Contents
* [General Info](#general-information)
* [Model Architecture](#model-architecture)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)

## General Information
The dataset comprises 2357 images depicting malignant and benign oncological conditions, sourced from the International Skin Imaging Collaboration (ISIC). These images were categorized based on the classification provided by ISIC, with each subset containing an equal number of images.
In order to address the challenge of class imbalance, the Augmentor Python package (https://augmentor.readthedocs.io/en/master/) was employed to augment the dataset. This involved generating additional samples for all classes, ensuring that none of the classes had insufficient representation.
The aim of this task is to assign a specific class label to a particular type of skin cancer.

## Model Architecture
The break down of the final provided CNN architecture step by step:

1. Data Augmentation: The augmentation_data variable refers to the augmentation techniques applied to the training data. Data augmentation is used to artificially increase the diversity of the training dataset by applying random transformations such as rotation, scaling, and flipping to the images. This helps in improving the generalization capability of the model.

2. Normalization: The Rescaling(1./255) layer is added to normalize the pixel values of the input images. Normalization typically involves scaling the pixel values to a range between 0 and 1, which helps in stabilizing the training process and speeding up convergence.

3. Convolutional Layers: Three convolutional layers are added sequentially using the Conv2D function. Each convolutional layer is followed by a rectified linear unit (ReLU) activation function, which introduces non-linearity into the model. The padding='same' argument ensures that the spatial dimensions of the feature maps remain the same after convolution. The number within each Conv2D layer (16, 32, 64) represents the number of filters or kernels used in each layer, determining the depth of the feature maps.

4. Pooling Layers: After each convolutional layer, a max-pooling layer (MaxPooling2D) is added to downsample the feature maps, reducing their spatial dimensions while retaining the most important information. Max-pooling helps in reducing computational complexity and controlling overfitting.

5. Dropout Layer: A dropout layer (Dropout) with a dropout rate of 0.2 is added after the last max-pooling layer. Dropout is a regularization technique used to prevent overfitting by randomly dropping a fraction of the neurons during training.

6. Flatten Layer: The Flatten layer is added to flatten the 2D feature maps into a 1D vector, preparing the data for input into the fully connected layers.

7. Fully Connected Layers: Two fully connected (dense) layers (Dense) are added with ReLU activation functions. The first dense layer consists of 128 neurons, and the second dense layer outputs the final classification probabilities for each class label.

8. Output Layer: The number of neurons in the output layer is determined by the target_labels variable, representing the number of classes in the classification task. The output layer does not have an activation function specified, as it is followed by the loss function during training.

9. Model Compilation: The model is compiled using the Adam optimizer (optimizer='adam') and the Sparse Categorical Crossentropy loss function (loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)), which is suitable for multi-class classification problems. Additionally, accuracy is chosen as the evaluation metric (metrics=['accuracy']).

10. Training: The model is trained using the fit method with the specified number of epochs (epochs=50). The ModelCheckpoint and EarlyStopping callbacks are employed to monitor the validation accuracy during training. The ModelCheckpoint callback saves the model with the best validation accuracy, while the EarlyStopping callback stops training if the validation accuracy does not improve for a specified number of epochs (patience=5 in this case). These callbacks help prevent overfitting and ensure that the model converges to the best possible solution.

## Conclusions
- Initial model has memorized the training data, including its noise, but struggles to generalize to unseen data.
- Data augmentation successfully mitigated overfitting but poor performance on both training and validation sets indicates underfitting.
- The training data exhibits a significant class imbalance. Seborrheic keratosis represents only 3.44% of the samples. Pigmented benign keratosis (20.63%) and Melanoma (19.56%) are over represented.
- Class rebalancing significantly improved model performance on both datasets.
- Final model achieved 84% training accuracy and approximately 79% validation accuracy. The close agreement between training and validation accuracies demonstrates the model's strong generalization ability.
- Batch normalization did not improve training or validation accuracy.


## Technologies Used
- Python - version 3.11.4
- Matplotlib - version 3.7.1
- Numpy - version 1.24.3
- Pandas - version 1.5.3
- Seaborn - version 0.12.2
- Tensorflow - version 2.15.0


## Contact
Created by [@umeshg1981] - feel free to contact me!
