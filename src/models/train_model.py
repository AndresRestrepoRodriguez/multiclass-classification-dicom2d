import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np
from typing import Optional



def train_model(
    num_epochs: int,
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    patience: int = 10
) -> None:
    
    """
    Trains a PyTorch model using a given training and validation data loader.

    Args:
        num_epochs (int): Number of epochs to train the model.
        model (torch.nn.Module): PyTorch model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        criterion (torch.nn.Module): Loss function used for training.
        optimizer (torch.optim.Optimizer): Optimizer used to update the model parameters.
        patience (int): Number of epochs to wait for improvement in validation loss before early stopping. Defaults to 10.

    Returns:
        None
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        train_losses, train_acc = [], []

        train_bar = tqdm(train_loader, desc="Training", position=0, leave=True)
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device, dtype=torch.long)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            predictions = torch.argmax(outputs, dim=1)
            train_acc.append(accuracy_score(labels.cpu().detach().numpy(), predictions.cpu().detach().numpy()))
            train_bar.set_postfix(loss=np.mean(train_losses), acc=np.mean(train_acc))

        train_loss_avg = np.mean(train_losses)
        train_acc_avg = np.mean(train_acc)

        model.eval()
        val_losses, val_acc = [], []

        val_bar = tqdm(val_loader, desc="Validating", position=0, leave=True)
        with torch.no_grad():
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device, dtype=torch.long)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())
                predictions = torch.argmax(outputs, dim=1)
                val_acc.append(accuracy_score(labels.cpu(), predictions.cpu()))
                val_bar.set_postfix(loss=np.mean(val_losses), acc=np.mean(val_acc))

        val_loss_avg = np.mean(val_losses)
        val_acc_avg = np.mean(val_acc)

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save(model.state_dict(), "best_model.pth")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs with no improvement")
                break

        print(f"Summary - Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc_avg:.4f}, Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc_avg:.4f}")