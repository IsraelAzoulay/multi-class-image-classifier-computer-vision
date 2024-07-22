"""
This file contains all the helper functions needed for the project.
"""
import os
import zipfile
import requests
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from typing import List
from typing import Tuple, List, Dict


def download_data(source: str, 
                  destination: str,
                  remove_source: bool = True) -> Path:
    """This function downloads a zipped dataset from source and unzips to destination.

    Args:
        source (str): A link to a zipped file containing data.
        destination (str): A target directory to unzip data to.
        remove_source (bool): Whether to remove the source after downloading and extracting.
    
    Returns:
        'pathlib.Path' to downloaded data.
    """
    # Defining the path to the dataset folder
    data_path = Path("data/")
    image_path = data_path / destination

    # Verify the existence of the image folder. If it does not exist, proceed to download it
    if image_path.is_dir():
        print(f"The {image_path} directory exists. Skipping download.")
    else:
        print(f"Did not find {image_path} directory. Creating one...")
        image_path.mkdir(parents=True, exist_ok=True)
        
        # Download the dataset
        target_file = Path(source).name
        with open(data_path / target_file, "wb") as f:
            request = requests.get(source)
            print(f"Downloading {target_file} from {source}...")
            f.write(request.content)

        # Unzip the dataset
        with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
            print(f"Unzipping {target_file} dataset...") 
            zip_ref.extractall(image_path)

        # Remove the '.zip' file
        if remove_source:
            os.remove(data_path / target_file)
    
    return image_path


def walk_through_dir(dir_path):
  """
  This function traverses the 'dir_path' and returns its contents.

  Args:
    dir_path (str or pathlib.Path): target directory

  Returns:
    A print out of:
      The number of subdiretories in the 'dir_path'.
      The number of images (files) in each subdirectory.
      The name of each subdirectory.
  """
  for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


def plot_transformed_images(image_paths, transform, n=3, seed=42):
    """This function plots a series of random images from 'image_paths'. It will
    open 'n' image paths from 'image_paths', transform them using 'transform',
    and plot them side by side.

    Args:
        image_paths (list): List of target image paths.
        transform (PyTorch Transforms): Transforms to apply to images.
        n (int, optional): Number of images to plot. Defaults to 3.
        seed (int, optional): Random seed for the random generator. Defaults to 42.
    """
    # Set the random seed
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # Transform and plot image
            transformed_image = transform(f).permute(1, 2, 0)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)


def display_random_images(dataset: torch.utils.data.dataset.Dataset,
                          classes: List[str] = None,
                          n: int = 10,
                          display_shape: bool = True,
                          seed: int = None):
    # Adjust the display in case that 'n' is too high
    if n > 10:
        n = 10
        display_shape = False
        print(f"For display purposes, 'n' shouldn't be larger than 10. Setting it to 10, and removing shape display.")

    # Set random seed
    if seed:
        random.seed(seed)

    # Retrieve random sample indexes
    random_samples_idx = random.sample(range(len(dataset)), k=n)

    # Define the plot
    plt.figure(figsize=(16, 8))

    # Loop through the samples and display the random samples
    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]

        # Modify the image tensor shape for plotting and plot
        targ_image_adjust = targ_image.permute(1, 2, 0)
        plt.subplot(1, n, i+1)
        plt.imshow(targ_image_adjust)
        plt.axis("off")
        if classes:
            title = f"class: {classes[targ_label]}"
            if display_shape:
                title = title + f"\nshape: {targ_image_adjust.shape}"
        plt.title(title)


def set_seeds(seed: int=42):
    """This function sets the random seeds for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Setting the seed for the general torch operations
    torch.manual_seed(seed)
    # Setting the seed for CUDA torch operations, which are the ones that happen on the GPU
    torch.cuda.manual_seed(seed)


def plot_loss_curves(results):
    """The function plots the training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    # Retrieve the loss values of the results dictionary
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Retrieve the accuracy values of the results dictionary
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Find out the number of epochs there was
    epochs = range(len(results['train_loss']))

    # Define the plot
    plt.figure(figsize=(15, 7))
    # Plot the loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    # Plot the accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();


def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str,
                        class_names: List[str] = None,
                        transform=None,
                        device: torch.device = device):
    """The function makes a prediction on a target image, and plots the image
    with its prediction.

    Args:
      model (torch.nn.Module): Trained model.
      image_path (str): Filepath to the target image.
      class_names (List[str], optional): Different class names for target image. Defaults to None.
      transform (_type_, optional): Transform of target image. Defaults to None.
      device (torch.device, optional): Target device to compute on. 
    
    Returns:
      A matplotlib plot of target image and model prediction as title.""" 
      
    # Read the image and convert the tensor values to float32
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    # Normalise
    target_image = target_image / 255.

    # Apply the transform if necessary
    if transform:
        target_image = transform(target_image)

    # Set the model on the target device
    model.to(device)

    # Set the model to evaluation mode
    model.eval()
    # Activate the inference mode
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)
        target_image_pred = model(target_image.to(device))

    # Convert the logits (the model output) to prediction probabilities
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # Convert the prediction probabilities to prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # Plot the image
    plt.imshow(target_image.squeeze().permute(1, 2, 0))
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else:
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False);


def create_writer(experiment_name: str,
                  model_name: str,
                  extra: str=None) -> torch.utils.tensorboard.writer.SummaryWriter():
    """This function creates a 'SummaryWriter()' instance, saving to a specific 'log_dir'.
    'log_dir' is a combination of 'runs/timestamp/experiment_name/model_name/extra',
    where 'timestamp' is the current date in the YYYY-MM-DD format.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to 'log_dir'.
    """
    from datetime import datetime
    import os

    # Retrieve the timestamp of the current date
    timestamp = datetime.now().strftime("%Y-%m-%d")

    if extra:
        # Define the log directory path
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)

    print(f"Created SummaryWriter. Saving to: {log_dir}...")

    return SummaryWriter(log_dir=log_dir)
