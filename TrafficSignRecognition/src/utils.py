#HELPER FUNCTIONS IN THIS FILE

#It will contain two function.
    #1. Save the model
    #2. Save loss and accuracy graphs to the disk

import torch
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

def save_model(epochs,model,optimizer, criterion):
    """
    Function to save the trained model to the disk
    """
    torch.save(
        {
            'epoch':epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
        }, 
        f"/content/outputs/model.pth"     # save the model to this location

    )

def save_plots(train_acc, valid_acc, train_loss, valid_loss):
    """ 
    Function to save accuracy and loss plots to the disk
    """

    # Accuracy plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    #Accuracy Plots are saved at this location
    plt.savefig(f"/content/outputs/accuracy.png")
    
    # Loss plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    #Loss Plots are saved at this location
    plt.savefig(f"/content/outputs/loss.png")
