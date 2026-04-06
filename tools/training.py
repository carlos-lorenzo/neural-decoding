import os
import torch
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    """
    A versatile, highly modular training loop for PyTorch models.
    Supports dynamic **kwargs for models requiring graph edge indices, attention masks, etc.
    """

    def __init__(self, model, optimizer, criterion, device=torch.device("cpu"), log_dir="tensorboard_logs", experiment_config=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)
        self.best_loss = float('inf')
        self.best_acc = 0.0
        self.experiment_config = experiment_config or {}

    def _move_kwargs_to_device(self, kwargs):
        """Moves any tensor arguments in kwargs to the specified device."""
        device_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                device_kwargs[k] = v.to(self.device)
            else:
                device_kwargs[k] = v
        return device_kwargs

    def train(self, train_loader, test_loader, epochs, patience=50, log_freq=10, save_path="best_model.pth", **kwargs):
        """
        Executes the training loop with validation, early stopping, and checkpointing.
        Any extra **kwargs are passed directly into the model's forward pass.
        """
        device_kwargs = self._move_kwargs_to_device(kwargs)
        os.makedirs(self.log_dir, exist_ok=True)
        epochs_no_improve = 0

        print(f"Starting training on {self.device}...")
        print(f"Logging TensorBoard to: {self.log_dir}")

        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0

            for batch_data, batch_labels in train_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)

                self.optimizer.zero_grad()

                # Pass data and any kwargs (like edge_index) to the model
                outputs = self.model(batch_data, **device_kwargs)

                loss = self.criterion(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()

                # Apply model-specific constraints (e.g. Max-Norm for EEGNet)
                if hasattr(self.model, 'apply_max_norm') and callable(self.model.apply_max_norm):
                    self.model.apply_max_norm()

                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)

            # Validation step (evaluating every epoch to track curves properly)
            avg_test_loss, accuracy = self.evaluate(
                test_loader, **device_kwargs)

            # TensorBoard logging
            self.writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            self.writer.add_scalar('Loss/Test', avg_test_loss, epoch)
            self.writer.add_scalar('Accuracy/Test', accuracy, epoch)

            if epoch % log_freq == 0 or epoch == 1:
                print(
                    f"Epoch [{epoch}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Test Acc: {accuracy:.2f}%")

            # Checkpointing and Early Stopping
            if avg_test_loss < self.best_loss:
                self.best_loss = avg_test_loss
                self.best_acc = accuracy  # Keep track of best accuracy corresponding to best loss
                torch.save(self.model.state_dict(),
                           os.path.join(self.log_dir, save_path))
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

        # Log hyperparameters and ultimate best metrics
        if self.experiment_config:
            self.writer.add_hparams(
                hparam_dict=self.experiment_config,
                metric_dict={
                    'hparam/best_loss': self.best_loss,
                    'hparam/best_accuracy': self.best_acc
                },
                run_name="."  # Keep it in the same directory
            )

        self.writer.close()
        print("Training complete. Best model saved to:",
              os.path.join(self.log_dir, save_path))

    def evaluate(self, test_loader, **kwargs):
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(data, **kwargs)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_test_loss = test_loss / len(test_loader)
        accuracy = 100 * correct / total
        return avg_test_loss, accuracy
