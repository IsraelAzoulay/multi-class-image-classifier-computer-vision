"""
This file trains a image classification model on device agnostic code.
"""
import os
import torch
import data_setup, engine, model_builder, utils
from torchvision import transforms


# Define the hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# Define the directories
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

# Define the target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create the transform pipeline
data_transform = transforms.Compose([
  transforms.Resize((64, 64)),
  transforms.ToTensor()
])

# Create the train and test DataLoaders
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

# Create the model and set it on the device
model = model_builder.TinyVGG(
    input_shape=3,
    hidden_units=HIDDEN_UNITS,
    output_shape=len(class_names)
).to(device)

# Define the loss function
loss_fn = torch.nn.CrossEntropyLoss()
# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Begin the training process
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

# Save the model
utils.save_model(model=model,
                 target_dir="models",
                 model_name="going_modular_script_mode_tinyvgg_model.pth")
