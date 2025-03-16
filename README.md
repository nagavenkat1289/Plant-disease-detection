# Plant Disease Detection using Python

This project aims to develop a system for detecting plant diseases using Python programming language. The system uses image processing techniques to analyze images of plants and determine whether they are infected with a disease.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Methodology](#methodology)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository

       git clone https://github.com/nagavenkat1289/Plant-disease-detection.git
       cd Patho_Plant
  
  
2. Create a virtual environment and activate it

       python -m venv venv
       source venv/bin/activate # for Linux/Mac
       venv\Scripts\activate.bat # for Windows 

  
3. Install the required packages

       pip install -r requirements.txt

## Usage

1. Download the dataset and extract it to a directory named `data` at the root of the project.

2. Train the model

       python train.py

3. Test the model on a single image

       python test.py --image path/to/image.jpg

4. Test the model on a directory of images

       python test.py --dir path/to/directory
  
## Data

The dataset used in this project is the [PlantVillage](https://plantvillage.psu.edu/) dataset. It contains images of healthy and diseased plant leaves, and the diseases are labeled. The dataset can be downloaded from the [official website](https://www.kaggle.com/datasets/emmarex/plantdisease).

## Methodology

The system uses a convolutional neural network (CNN) to classify images into healthy and diseased plants. The CNN is trained on the PlantVillage dataset using transfer learning with the VGG16 architecture. The weights of the pre-trained VGG16 model are frozen, and the final layers are replaced with a fully connected layer and a softmax activation function. The model is trained using the categorical cross-entropy loss function and the Adam optimizer.

### Data Preprocessing

1. **Image Resizing**: All images are resized to a fixed dimension (e.g., 224x224) to match the input size required by the VGG16 model.
2. **Normalization**: Pixel values are normalized to the range [0, 1] by dividing by 255.
3. **Data Augmentation**: To improve the model's generalization, data augmentation techniques such as rotation, zoom, horizontal flip, and vertical flip are applied to the training images.

### Model Architecture

1. **Base Model**: VGG16 pre-trained on ImageNet is used as the base model. The weights of this model are frozen to retain the learned features.
2. **Custom Layers**: The top layers of the VGG16 model are replaced with custom layers, including:
   - Flatten layer
   - Fully connected dense layer with 256 units and ReLU activation
   - Dropout layer with a dropout rate of 0.5
   - Output dense layer with a number of units equal to the number of classes and softmax activation

### Training Process

1. **Loss Function**: Categorical cross-entropy loss is used since the task is a multi-class classification problem.
2. **Optimizer**: Adam optimizer is used with an initial learning rate of 0.001.
3. **Batch Size**: A batch size of 32 is used for training.
4. **Epochs**: The model is trained for 50 epochs, with early stopping if the validation loss does not improve for 10 consecutive epochs.

### Evaluation Metrics

1. **Accuracy**: The primary metric for evaluating the model's performance.
2. **Precision, Recall, F1 Score**: These metrics are calculated per class to understand the model's performance on individual diseases.
3. **Confusion Matrix**: A confusion matrix is generated to visualize the performance of the classification model.

## Results

The results of the model will be presented here once the training and evaluation are completed.

## Contributing

Contributions are welcome! Please read the [contributing guidelines](CONTRIBUTING.md) for more information.
