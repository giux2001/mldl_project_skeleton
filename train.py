import torch 
import wandb

def train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        # todo...
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Logga la loss ogni N batch
        if batch_idx % 10 == 0:
            wandb.log({"train_loss": loss.item()})

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    wandb.log({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy})
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')

