import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np


def train_model(num_epochs, model, train_loader, val_loader, criterion, optimizer, writer=None, patience=10):
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
                images, labels = images.to(device), labels.to(device, dtype=torch.float32)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())
                predictions = torch.argmax(outputs, dim=1)
                val_acc.append(accuracy_score(labels.cpu(), predictions.cpu()))
                val_bar.set_postfix(loss=np.mean(val_losses), acc=np.mean(val_acc))

        val_loss_avg = np.mean(val_losses)
        val_acc_avg = np.mean(val_acc)

        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar('Loss/train', train_loss_avg, epoch)
            writer.add_scalar('Accuracy/train', train_acc_avg, epoch)
            writer.add_scalar('Loss/validation', val_loss_avg, epoch)
            writer.add_scalar('Accuracy/validation', val_acc_avg, epoch)

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