import datetime
import os
from math import ceil
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from packages.utils.Loss import Loss
from packages.utils.Sensetivity import Sensitivity
from packages.utils.TrackBest import TrackBest
from packages.utils.TimeEstimator import TimeEstimator
from packages.utils.Metrics import ConfusionMatrix
import torch.nn.functional as F

def visualize_images_with_masks(inputs, labels, predicted):
    inputs = inputs.cpu().numpy()
    labels = labels.cpu().numpy()
    predicted = predicted.cpu().numpy()
    batch_size = inputs.shape[0]

    for i in range(batch_size):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(np.transpose(inputs[i], (1, 2, 0)))
        plt.title('Input Image')
        plt.subplot(1, 3, 2)
        plt.imshow(labels[i], cmap='gray')
        plt.title('Ground Truth Mask')
        plt.subplot(1, 3, 3)
        plt.imshow(predicted[i], cmap='gray')
        plt.title('Predicted Segmentation')
        plt.show()

class KFoldSimpleRun:
    def __init__(self, data, model, is_segmentation=False, k_folds=5):
        self.data = data
        self.model = model
        self.is_segmentation = is_segmentation
        self.k_folds = k_folds

    def get_loader(self, indices, batch_size=1, shuffle=False):
        subset = Subset(self.data, indices)
        loader = DataLoader(dataset=subset, batch_size=batch_size, drop_last=False, shuffle=shuffle)
        return loader, ceil(len(subset) / batch_size)

    def train(self, device, loss_fn, optimizer, save_path, **kwargs):
        kfold = KFold(n_splits=self.k_folds, shuffle=True)
        fold_results = []
        en = kfold.split(self.data)
        for fold, (train_ids, val_ids) in enumerate(en):
            # Modify save path to include fold number
            fold_save_path = os.path.join(save_path, f'fold_{fold}')
            # Add fold number to kwargs
            kwargs['fold'] = fold
            print(f'FOLD {fold}')

            print('--------------------------------')

            loader_kwargs = {key: kwargs[key] for key in ['batch_size', 'shuffle'] if key in kwargs}

            print(f"loader_kwargs = {loader_kwargs}")

            train_loader, train_n_batch = self.get_loader(train_ids, **loader_kwargs)
            validation_loader, val_n_batch = self.get_loader(val_ids, shuffle=False, batch_size=kwargs.get('batch_size', 1))

            history = self.train_loop(device, loss_fn, optimizer, fold_save_path, train_loader, validation_loader, train_n_batch, val_n_batch, **kwargs)
            fold_results.append(history)
            print(f'Finished Fold {fold}')
            print('--------------------------------')

            # Save best results of the fold to a text file
            self.save_fold_results(history, fold, fold_save_path)

        return fold_results

    def train_loop(self, device, loss_fn, optimizer, save_path, train_loader, validation_loader, train_n_batch, val_n_batch, epochs=100, **kwargs):
        model = self.model.to(device)
        train_confusion = ConfusionMatrix()
        validation_confusion = ConfusionMatrix()
        train_loss = Loss()
        validation_loss = Loss()

        train_accuracy_saver = TrackBest(0, lambda x, y: y > x, os.path.join(save_path, 'best_train', 'accuracy'))
        train_loss_saver = TrackBest(float('inf'), lambda x, y: y < x, os.path.join(save_path, 'best_train', 'loss'))
        train_dice_saver = TrackBest(0, lambda x, y: y > x, os.path.join(save_path, 'best_train', 'dice'))
        train_sens_saver = TrackBest(0, lambda x, y: y > x, os.path.join(save_path, 'best_train', 'sensitivity'))
        train_specificity_saver =TrackBest(0, lambda x, y: y > x, os.path.join(save_path, 'best_train', 'specificity'))

        val_accuracy_saver = TrackBest(0, lambda x, y: y > x, os.path.join(save_path, 'best_val', 'accuracy'))
        val_loss_saver = TrackBest(float('inf'), lambda x, y: y < x, os.path.join(save_path, 'best_val', 'loss'))
        val_dice_saver = TrackBest(0, lambda x, y: y > x, os.path.join(save_path, 'best_val', 'dice'))
        val_sens_saver = TrackBest(0, lambda x, y: y > x, os.path.join(save_path, 'best_val', 'sensitivity'))
        val_specificity_saver = TrackBest(0, lambda x, y: y > x, os.path.join(save_path, 'best_train', 'specificity'))

        time_estimator = TimeEstimator(epochs, train_n_batch)
        remaining_time = 0

        for epoch_number in range(epochs):
            start_time = datetime.datetime.now()
            model.train()
            for i, data in enumerate(train_loader):
                data = list(map(lambda x: x.to(device), data))
                labels = data[-1]
                if self.is_segmentation:
                    labels = labels.squeeze(1)

                outputs = model(*data[:-1])
                optimizer.zero_grad()
                l1_lambda = kwargs.get('l1_lambda', 0)
                l1_regularization = torch.tensor(0.0).to(device)
                for param in model.parameters():
                    l1_regularization += torch.norm(param, p=1)

                loss = loss_fn(outputs, labels) + l1_lambda * l1_regularization
                loss.backward()
                optimizer.step()

                if self.is_segmentation:
                    predicted = torch.zeros(labels.size()).to(labels.get_device())
                    idx = outputs[:, 1, :, :] > 0.5
                    predicted[idx] = 1
                    predicted[predicted != 1] = 0
                else:
                    _, predicted = torch.max(outputs.data, dim=1)

                train_confusion.update(predicted, labels)
                train_loss.update(loss.item())

                if (i == 0 or i % 2 == 1 or i == train_n_batch - 1) and ('verbose' not in kwargs or kwargs['verbose'] is False):
                    done = epoch_number * 10 // epochs
                    remain = 10 - done
                    est = datetime.timedelta(seconds=time_estimator.get_time()) if remaining_time else "infinity"
                    if self.is_segmentation:
                        metrics = (f'Dice[{train_confusion.get_value("dice"):0.2f}|{train_confusion.get_best("dice"):0.2f}]' +
                                   f'Sens[{train_confusion.get_value("sensitivity"):0.2f}|{train_confusion.get_best("sensitivity"):0.2f}]')
                        val_metrics = (f'Dice[{validation_confusion.get_value("dice"):0.2f}|{validation_confusion.get_best("dice"):0.2f}]' +
                                       f'Sens[{validation_confusion.get_value("sensitivity"):0.2f}|{validation_confusion.get_best("sensitivity"):0.2f}]')
                    else:
                        metrics = (f'Acc[{train_confusion.get_value("accuracy"):0.2f}|{train_confusion.get_best("accuracy"):0.2f}] ' +
                                   f'Sens[{train_confusion.get_value("sensitivity"):0.2f}|{train_confusion.get_best("sensitivity"):0.2f}] ' +
                                   f'Spec[{train_confusion.get_value("specificity"):0.2f}|{train_confusion.get_best("specificity"):0.2f}]'+
                                   f'F1_Score[{train_confusion.get_value("dice"):0.2f}|{train_confusion.get_best("dice"):0.2f}]')
                        val_metrics = (f'Acc[{validation_confusion.get_value("accuracy"):0.2f}|{validation_confusion.get_best("accuracy"):0.2f}] ' +
                                       f'Sens[{validation_confusion.get_value("sensitivity"):0.2f}|{validation_confusion.get_best("sensitivity"):0.2f}] ' +
                                       f'Spec[{validation_confusion.get_value("specificity"):0.2f}|{validation_confusion.get_best("specificity"):0.2f}]'+
                                       f'F1_Score[{validation_confusion.get_value("dice"):0.2f}|{validation_confusion.get_best("dice"):0.2f}]')

                    print(('\r' + f'Epoch [{epoch_number + 1}/{epochs}] | Batch [{i + 1}/{train_n_batch}] | '
                           + '=' * (done - 1)) + '>' + (' ' * remain) +
                          f'| EST:{est} | Train: {metrics} | Val: {val_metrics}',
                          end='')
                time_estimator.sub_step()

            if validation_loader is not None:
                model.eval()
                with torch.no_grad():
                    for batch, data in enumerate(validation_loader):
                        data = list(map(lambda x: x.to(device), data))
                        labels = data[-1]
                        if self.is_segmentation:
                            labels = labels.squeeze(1)
                        outputs = model(*data[:-1])

                        if self.is_segmentation:
                            predicted = torch.zeros(labels.size()).to(labels.get_device())
                            idx = outputs[:, 1, :, :] > 0.5
                            predicted[idx] = 1
                            predicted[predicted != 1] = 0
                        else:
                            _, predicted = torch.max(outputs.data, dim=1)


                        validation_confusion.update(predicted, labels)
                        loss = loss_fn(outputs, labels)
                        validation_loss.update(loss.item())

            train_accuracy_value = train_confusion.get_value("accuracy")
            train_sens_value = train_confusion.get_value("sensitivity")
            train_spec_value = train_confusion.get_value("specificity")
            train_dice_value = train_confusion.get_value("dice")
            train_loss_value = train_loss.get_loss()
            if validation_loader is not None:
                validation_accuracy_value = validation_confusion.get_value("accuracy")
                validation_sens_value = validation_confusion.get_value("sensitivity")
                validation_spec_value = validation_confusion.get_value("specificity")
                validation_dice_value = validation_confusion.get_value("dice")
                validation_loss_value = validation_loss.get_loss()

            if self.is_segmentation:
                train_dice_saver.update_value(train_dice_value, model)
                train_sens_saver.update_value(train_sens_value, model)
            else:
                train_accuracy_saver.update_value(train_accuracy_value, model)
                train_dice_saver.update_value(train_dice_value, model)
                train_sens_saver.update_value(train_sens_value, model)
                train_specificity_saver.update_value(train_spec_value, model)
            train_loss_saver.update_value(train_loss_value, model)
            if validation_loader is not None:
                if self.is_segmentation:
                    val_dice_saver.update_value(validation_confusion.get_best('dice'), model)
                    val_sens_saver.update_value(validation_sens_value, model)
                else:
                    val_accuracy_saver.update_value(validation_confusion.get_best('accuracy'), model)
                    val_dice_saver.update_value(validation_confusion.get_best('dice'), model)
                    val_sens_saver.update_value(validation_sens_value, model)
                    val_specificity_saver.update_value(validation_confusion.get_best('specificity'), model)

                val_loss_saver.update_value(validation_loss_value, model)

            time_estimator.sub_step()
            remaining_time = time_estimator.get_time()

            if self.is_segmentation:
                if kwargs.get("save_best_dice", False):
                    train_dice_saver.save_path(model, train_dice_value)
                if kwargs.get("save_best_sensitivity", False):
                    train_sens_saver.save_path(model, train_sens_value)
                if kwargs.get("save_best_loss", False):
                    train_loss_saver.save_path(model, train_loss_value)
                if kwargs.get("save_best_accuracy", False):
                    train_accuracy_saver.save_path(model, train_accuracy_value)
            else:
                if kwargs.get("save_best_accuracy", False):
                    train_accuracy_saver.save_path(model, train_accuracy_value)
                if kwargs.get("save_best_loss", False):
                    train_loss_saver.save_path(model, train_loss_value)
                if kwargs.get("save_best_dice", False):
                    train_dice_saver.save_path(model, train_dice_value)
                if kwargs.get("save_best_sensitivity", False):
                    train_sens_saver.save_path(model, train_sens_value)
                if kwargs.get("save_best_loss", False):
                    train_loss_saver.save_path(model, train_loss_value)
                if kwargs.get("save_best_accuracy", False):
                    train_accuracy_saver.save_path(model, train_accuracy_value)
                if kwargs.get("save_best_specificity", False):
                    train_specificity_saver.save_path(model, train_spec_value)

            end_time = datetime.datetime.now()
            elapsed_time = end_time - start_time
            elapsed_time = elapsed_time.total_seconds() / 60

            if "log_fn" in kwargs:
                kwargs["log_fn"](f"Epoch {epoch_number + 1}/{epochs} finished in {elapsed_time:.2f} minutes")

            train_confusion.reset()
            train_loss.reset()
            if validation_loader is not None:
                validation_confusion.reset()
                validation_loss.reset()


            train_confusion.save(os.path.join(save_path, 'fold_{}_train_history'.format(kwargs['fold'])))
            train_loss.save(os.path.join(save_path, 'fold_{}_train_loss_history.csv'.format(kwargs['fold'])))
            validation_confusion.save(os.path.join(save_path, 'fold_{}_val_history'.format((kwargs['fold']))))
            validation_loss.save(os.path.join(save_path, 'fold_{}_val_loss_history.csv'.format(kwargs['fold'])))
            model.save_all(os.path.join(save_path, 'fold_{}_last_model.pth'.format(kwargs['fold'])))

        # print(train_specificity_saver)
        return train_accuracy_saver, train_loss_saver, train_dice_saver,train_sens_saver,train_specificity_saver, val_accuracy_saver, val_loss_saver, val_dice_saver, val_sens_saver, val_specificity_saver
    def save_fold_results(self, history, fold, save_path):
        train_accuracy_saver, train_loss_saver, train_dice_saver, train_sens_saver, val_accuracy_saver, val_loss_saver, val_dice_saver, val_sens_saver, val_specificity_saver, train_specificity_saver = history
        with open(os.path.join(save_path, f'fold_{fold}_results.txt'), 'w') as f:
            f.write(f'Fold {fold} Results:\n')
            f.write(f'Best Train Accuracy: {train_accuracy_saver.value}\n')
            f.write(f'Best Train Loss: {train_loss_saver.value}\n')
            f.write(f'Best Train Dice: {train_dice_saver.value}\n')
            f.write(f'Best Train Sensitivity: {train_sens_saver.value}\n')
            f.write(f'Best Train Specifity: {train_specificity_saver.value}\n')
            f.write(f'Best Validation Accuracy: {val_accuracy_saver.value}\n')
            f.write(f'Best Validation Loss: {val_loss_saver.value}\n')
            f.write(f'Best Validation Dice: {val_dice_saver.value}\n')
            f.write(f'Best Validation Sensitivity: {val_sens_saver.value}\n')
            f.write(f'Best Validation Specifity: {val_specificity_saver.value}\n')
            f.write('----------------------------------------------------------')



    def evaluate(self, device, loss_fn, test_loader, save_path, **kwargs):
        model = self.model.to(device)
        model.eval()
        test_confusion = ConfusionMatrix()
        test_loss = Loss()

        with torch.no_grad():
            for data in test_loader:
                data = list(map(lambda x: x.to(device), data))
                labels = data[-1]
                if self.is_segmentation:
                    labels = labels.squeeze(1)

                outputs = model(*data[:-1])

                if self.is_segmentation:
                    predicted = torch.zeros(labels.size()).to(labels.get_device())
                    idx = outputs[:, 1, :, :] > 0.5
                    predicted[idx] = 1
                    predicted[predicted != 1] = 0
                else:
                    _, predicted = torch.max(outputs.data, dim=1)

                test_confusion.update(predicted, labels)
                loss = loss_fn(outputs, labels)
                test_loss.update(loss.item())

        test_accuracy_value = test_confusion.get_value("accuracy")
        test_sens_value = test_confusion.get_value("sensitivity")
        test_spec_value = test_confusion.get_value("specificity")
        test_dice_value = test_confusion.get_value("dice")
        test_loss_value = test_loss.get_loss()

        print(f'Test Accuracy: {test_accuracy_value}')
        print(f'Test Sensitivity: {test_sens_value}')
        print(f'Test Specificity: {test_spec_value}')
        print(f'Test Dice: {test_dice_value}')
        print(f'Test Loss: {test_loss_value}')

        # Save evaluation results to a text file
        with open(os.path.join(kwargs.get('save_path', '.'), 'evaluation_results.txt'), 'w') as f:
            f.write('Test Results:\n')
            f.write(f'Test Accuracy: {test_accuracy_value}\n')
            f.write(f'Test Sensitivity: {test_sens_value}\n')
            f.write(f'Test Specificity: {test_spec_value}\n')
            f.write(f'Test Dice: {test_dice_value}\n')
            f.write(f'Test Loss: {test_loss_value}\n')
