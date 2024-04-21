import datetime
import os

import torch
from torch.utils.data import random_split, DataLoader, ConcatDataset
from .utils.Accuracy import Accuracy
from ..utils.Loss import Loss
from ..utils.TrackBest import TrackBest
from ..utils.TimeEstimator import TimeEstimator


def reduce(func, iterable):
    result = 0
    for i in iterable:
        result = func(result, i)
    return result


class CrossValidation:
    def __init__(self, data, model, num_splits=10):
        size = len(data) // num_splits
        last_size = len(data) - (num_splits - 1) * size
        self.splits = random_split(data, [size] * (num_splits - 1) + [last_size])
        self.data = data
        self.model = model
        self.num_splits = num_splits

    def get_loader(self, index, batch_size=8):
        if index is None:
            return DataLoader(dataset=self.data, batch_size=batch_size, shuffle=True), None
        data = self.splits[index]
        other = self.splits[:index] + self.splits[index + 1:]
        train_loader = DataLoader(dataset=ConcatDataset(other), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader

    def train_split(self, index, device, loss_fn, optimizer, save_path, epochs=100, **kwargs):
        train_loader, validation_loader = self.get_loader(index, **kwargs)
        history = self.train_loop(device, loss_fn, optimizer, save_path, train_loader, validation_loader, epochs, **kwargs)
        return history

    def train(self, device, loss_fn, optimizer, save_path, **kwargs):
        histories = []
        for i in range(self.num_splits):
            print(f'Split {i + 1}/{self.num_splits} start')
            history = self.train_split(i, device, loss_fn, optimizer, os.path.join(save_path, f'split_{i}'), **kwargs)
            histories.append({
                'train_accuracy': history['train_accuracy'].get_best(),
                'val_accuracy': history['val_accuracy'].get_best(),
                'train_loss': history['train_loss'].get_best(),
                'val_loss': history['val_loss'].get_best()
            })
            print(f'Split {i + 1}/{self.num_splits} ended')
            print(f'Best Train Accuracy: {history["train_accuracy"].get_best():.2f}%')
            print(f'Best Validation Accuracy: {history["val_accuracy"].get_best():.2f}%')
            print(f'Best Train Loss: {history["train_loss"].get_best():.4f}%')
            print(f'Best Validation Loss: {history["val_loss"].get_best():.4f}%')
            print('----------------------------------------------------------------')

        acc = reduce(lambda x, y: x + y['train_accuracy'], histories) / self.num_splits
        loss = reduce(lambda x, y: x + y['train_loss'], histories) / self.num_splits
        val_acc = reduce(lambda x, y: x + y['val_accuracy'], histories) / self.num_splits
        val_loss = reduce(lambda x, y: x + y['val_loss'], histories) / self.num_splits
        print('Cross Validation Ended:')
        print(f'Train Accuracy: {acc}')
        print(f'Validation Accuracy: {val_acc}')
        print(f'Train Loss: {loss}')
        print(f'Validation Loss: {val_loss}')

        train_loader, _ = self.get_loader(None)
        self.train_loop(device, loss_fn, optimizer, save_path, train_loader, None)

    def train_loop(self, device, loss_fn, optimizer, save_path, train_loader, validation_loader, epochs=100, **kwargs):
        model = self.model
        train_accuracy = Accuracy()
        validation_accuracy = Accuracy()
        train_loss = Loss()
        validation_loss = Loss()
        train_accuracy_saver = TrackBest(0, lambda x, y: y > x, os.path.join(save_path, 'best_train', 'accuracy'))
        val_accuracy_saver = TrackBest(0, lambda x, y: y > x, os.path.join(save_path, 'best_val', 'accuracy'))
        train_loss_saver = TrackBest(float('inf'), lambda x, y: y < x, os.path.join(save_path, 'best_train', 'loss'))
        val_loss_saver = TrackBest(float('inf'), lambda x, y: y < x, os.path.join(save_path, 'best_val', 'loss'))

        time_estimator = TimeEstimator(epochs)
        for ep in range(epochs):
            start_time = datetime.datetime.now()
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                _, predicted = torch.max(outputs.data, 1)
                train_accuracy.update(labels.size(0), (predicted == labels).sum().item())
                train_loss.update(loss.item())
            if validation_loader is not None:
                model.eval()
                with torch.no_grad():
                    for images, labels in validation_loader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        validation_accuracy.update(labels.size(0), (predicted == labels).sum().item())
                        loss = loss_fn(outputs, labels)
                        validation_loss.update(loss.item())

            train_accuracy_value = train_accuracy.get_accuracy()
            train_loss_value = train_loss.get_loss()
            if validation_loader is not None:
                validation_accuracy_value = validation_accuracy.get_accuracy()
                validation_loss_value = validation_loss.get_loss()

            train_accuracy_saver.update_value(train_accuracy_value, model)
            train_loss_saver.update_value(train_loss_value, model)
            if validation_loader is not None:
                val_accuracy_saver.update_value(validation_accuracy_value, model)
                val_loss_saver.update_value(validation_loss_value, model)

            end_time = datetime.datetime.now()
            time_estimator.update((end_time - start_time).total_seconds())
            remaining_time = time_estimator.get_time()
            print(
                f"Epoch [{ep + 1}/{epochs}] | Estimated Remaining Time: {datetime.timedelta(seconds=remaining_time)}")
            print(f'Train Loss: {train_loss_value:.4f} | Train Accuracy: {train_accuracy_value:.2f}%')
            if validation_loader is not None:
                print(
                    f'Validation Loss: {validation_loss_value:.4f} | Validation Accuracy: {validation_accuracy_value:.2f}%')
            print(
                '=====================================================================================================')

            train_accuracy.reset()
            train_loss.reset()
            if validation_loader is not None:
                validation_accuracy.reset()
                validation_loss.reset()

        return {
            'train_accuracy': train_accuracy,
            'train_loss': train_loss,
            'val_accuracy': validation_accuracy,
            'val_loss': validation_loss
        }
