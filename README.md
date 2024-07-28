# DishVision: Computer Vision Multi-Class Image Classifier
This repository houses an advanced project focused on classifying images of three distinct types of food using cutting-edge deep learning and computer vision techniques. The project leverages transfer learning with PyTorch to achieve an impressive accuracy of **96+%** on the test set. Additionally, it includes a detailed replication of the Vision Transformer (ViT) architecture from a prominent machine learning research paper, showcasing both the potential and challenges of this advanced approach. The implementation is executed on Google Colab for efficient computation and easy accessibility. The optimal model has been deployed on Hugging Face as an interactive app and can be accessed [here](https://huggingface.co/spaces/IsraelAzoulay/DishVision).

## Key Features
- **Data Source:** Preprocessed food image datasets, prepared in the 'Structuring_the_Food101_Dataset' notebook.
- **Deep Learning Framework:** PyTorch
- **Model Type:** A diverse range of models including Transfer Learning Feature Extraction, Transfer Learning with Data Augmentation, TinyVGG, Vision Transformer (ViT) with Machine Learning Research Paper replication, and more.
- **Objective:** Classifying images into three distinct food categories.
- **Accuracy:** Achieves an exceptional **96+%** accuracy on the test set.
- **Development Environment:** Google Colab for development and execution.
- **Deployment:** The optimal model is deployed on Hugging Face as an interactive app. 

## Repository Contents
- **Notebooks:**
  - **Structuring_the_Food101_Dataset:** Notebook dedicated to preparing and structuring the Food-101 dataset sourced from 'torchvision.datasets'.
  - **Computer_Vision_Multi_Class_Image_Classifier_Project:** Comprehensive notebook detailing data loading and preprocessing, model training, evaluation, prediction, saving, deployment, and more.
- **Scripts:**
  - **helper_functions.py:** Contains essential utility functions required for the project, located in the 'helpers' folder.
- **Data:** Includes preprocessed datasets and a custom image for prediction purposes, located in the 'data' folder.

## Getting Started
1. **Clone the repository:**
   !git clone https://github.com/IsraelAzoulay/multi-class-image-classifier-computer-vision.git
2. **Open the provided Google Colab notebooks:**
Navigate to the 'notebooks' folder and open the desired notebook in Google Colab.
3. **Run the Notebooks:**
Follow the instructions in the notebooks to download the datasets, preprocess the data, train, evaluate, predict, save and deploy the models.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License.
