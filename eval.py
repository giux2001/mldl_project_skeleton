import train
from models.model import CustomNet
import torch
from torch import nn
import wandb


# Validation loop
def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0

    correct, total = 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            # todo...
            outputs = model(inputs)
            loss = criterion(outputs, targets)


            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct / total

    wandb.log({"val_loss": val_loss, "val_accuracy": val_accuracy})

    print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')
    return val_accuracy


model = CustomNet().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

best_acc = 0

from data.data import tiny_imagenet_dataset_train
from data.data import tiny_imagenet_dataset_val

train_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_train, batch_size=32, shuffle=True, num_workers = 8)
val_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_val, batch_size=32, shuffle=False, num_workers = 8)

# Run the training process for {num_epochs} epochs
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    train(epoch, model, train_loader, criterion, optimizer)

    # At the end of each training iteration, perform a validation step
    val_accuracy = validate(model, val_loader, criterion)

    # Best validation accuracy
    best_acc = max(best_acc, val_accuracy)

    # Salva il miglior modello
    if val_accuracy > best_acc:
        best_acc = val_accuracy
        torch.save(model.state_dict(), "best_model.pth")
        wandb.save("best_model.pth")

wandb.finish()
print(f'Best validation accuracy: {best_acc:.2f}%')