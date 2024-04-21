import datetime
import os
from math import ceil

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from packages.utils.Loss import Loss
from packages.utils.Sensetivity import Sensitivity
from packages.utils.TrackBest import TrackBest
from packages.utils.TimeEstimator import TimeEstimator
from packages.utils.Metrics import ConfusionMatrix
import torch.nn.functional as F


def visualize_images_with_masks(inputs, labels, predicted):
    # Convert PyTorch tensors to NumPy arrays
    inputs = inputs.cpu().numpy()
    labels = labels.cpu().numpy()
    predicted = predicted.cpu().numpy()

    # Assuming inputs, labels, and predicted have shape (batch_size, channels, height, width)
    batch_size = inputs.shape[0]

    for i in range(batch_size):
        plt.figure(figsize=(12, 4))

        # Plot the input image
        plt.subplot(1, 3, 1)
        plt.imshow(np.transpose(inputs[i], (1, 2, 0)))
        plt.title('Input Image')

        # Plot the ground truth mask
        plt.subplot(1, 3, 2)
        plt.imshow(labels[i], cmap='gray')
        plt.title('Ground Truth Mask')

        # Plot the predicted segmentation
        plt.subplot(1, 3, 3)
        plt.imshow(predicted[i], cmap='gray')
        plt.title('Predicted Segmentation')

        plt.show()


class SimpleRun:
    def __init__(self, train_data, validation_data, test_data, model, is_segmentation=False):
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.model = model
        self.is_segmentation = is_segmentation

    def visualize_images_with_masks(self, inputs, labels, predicted):
        # Convert PyTorch tensors to NumPy arrays
        inputs = inputs.cpu().numpy()
        labels = labels.cpu().numpy()
        predicted = predicted.cpu().numpy()

        # Assuming inputs, labels, and predicted have shape (batch_size, channels, height, width)
        batch_size = inputs.shape[0]

        for i in range(batch_size):
            plt.figure(figsize=(12, 4))

            # Plot the input image
            plt.subplot(1, 3, 1)
            plt.imshow(np.transpose(inputs[i], (1, 2, 0)))
            plt.title('Input Image')

            # Plot the ground truth mask
            plt.subplot(1, 3, 2)
            plt.imshow(labels[i], cmap='gray')
            plt.title('Ground Truth Mask')

            # Plot the predicted segmentation
            plt.subplot(1, 3, 3)
            plt.imshow(predicted[i], cmap='gray')
            plt.title('Predicted Segmentation')

            plt.show()

    def get_loader(self, batch_size=1, **_):
        train_loader = DataLoader(dataset=self.train_data, batch_size=batch_size, drop_last=False, shuffle=True)
        validation_loader = DataLoader(dataset=self.validation_data, batch_size=batch_size, drop_last=False,
                                       shuffle=True)
        test_loader = DataLoader(dataset=self.test_data, batch_size=batch_size, drop_last=False, shuffle=True)
        return train_loader, validation_loader, test_loader, \
            ceil(len(self.train_data) / batch_size), \
            ceil(len(self.validation_data) / batch_size), \
            ceil(len(self.test_data) / batch_size)

    def train(self, device, loss_fn, optimizer, save_path, **kwargs):
        if 'l1_lambda' in kwargs:
            print('--[Training with L1 regularization]--')
            print('If it is not what you expected, check that l1_lambda is not provided')
        train_loader, validation_loader, test_loader, \
            train_n_batch, val_n_batch, test_n_batch = self.get_loader(**kwargs)
        history = self.train_loop(device, loss_fn, optimizer, save_path, train_loader, validation_loader, test_loader,
                                  train_n_batch, val_n_batch, **kwargs)

        train_confusion = history["train_confusion"]
        validation_confusion = history["val_confusion"]
        best_result = \
            f'''
            Best Train Loss: {history["train_loss"].get_best():.4f}
            Best Validation Loss: {history["val_loss"].get_best():.4f}
            
            Best Train Accuracy: {train_confusion.get_best("accuracy"):.2f}%
            Best Validation Accuracy: {validation_confusion.get_best("accuracy"):.2f}%
            
            Best Train Sensitivity: {train_confusion.get_best("sensitivity"):.2f}%
            Best Validation Sensitivity: {validation_confusion.get_best("sensitivity"):.2f}%
            
            Best Train Specificity Score: {train_confusion.get_best("specificity"):.4f}
            Best Validation Specificity: {validation_confusion.get_best("specificity"):.2f}%
            
            Best Train Dice Score: {train_confusion.get_best("dice"):.4f}
            Best Validation Dice Score: {validation_confusion.get_best("dice"):.4f}
            '''
        with open(f'{save_path}/best_result.txt', 'w') as file:
            print(best_result)
            file.write(best_result)
        print('----------------------------------------------------------------')

        return history

    def train_loop(self, device, loss_fn, optimizer, save_path, train_loader, validation_loader, test_loader,
                   train_n_batch, val_n_batch, epochs=100, **kwargs):
        model = self.model.to(device)  # Move model to device

        # ----------------------------- Metrics
        train_confusion = ConfusionMatrix()
        validation_confusion = ConfusionMatrix()

        train_loss = Loss()
        validation_loss = Loss()

        # ----------------------------- Checkpoint Savers
        train_accuracy_saver = TrackBest(0, lambda x, y: y > x, os.path.join(save_path, 'best_train', 'accuracy'))
        train_loss_saver = TrackBest(float('inf'), lambda x, y: y < x, os.path.join(save_path, 'best_train', 'loss'))
        train_dice_saver = TrackBest(0, lambda x, y: y > x, os.path.join(save_path, 'best_train', 'dice'))
        train_sens_saver = TrackBest(0, lambda x, y: y > x, os.path.join(save_path, 'best_train', 'sensitivity'))

        val_accuracy_saver = TrackBest(0, lambda x, y: y > x, os.path.join(save_path, 'best_val', 'accuracy'))
        val_loss_saver = TrackBest(float('inf'), lambda x, y: y < x, os.path.join(save_path, 'best_val', 'loss'))
        val_dice_saver = TrackBest(0, lambda x, y: y > x, os.path.join(save_path, 'best_val', 'dice'))
        val_sens_saver = TrackBest(0, lambda x, y: y > x, os.path.join(save_path, 'best_val', 'sensitivity'))

        # ----------------------------- Time Estimator
        time_estimator = TimeEstimator(epochs, train_n_batch)
        remaining_time = 0

        # ----------------------------- Initial Values
        train_accuracy_value, validation_accuracy_value = 0, 0
        train_sens_value, validation_sens_value = 0, 0
        train_spec_value, validation_spec_value = 0, 0
        train_dice_value, validation_dice_value = 0, 0
        train_loss_value, validation_loss_value = float('inf'), float('inf')

        # -----------------------------Training Loop
        for epoch_number in range(epochs):
            start_time = datetime.datetime.now()
            model.train()  # Set model to train mode
            # ----------------------------- Train each batch
            for i, data in enumerate(train_loader):
                data = list(map(lambda x: x.to(device), data))  # Move all data to device

                labels = data[-1]

                # TODO: move to Dataset
                if self.is_segmentation:
                    labels.squeeze(1)

                outputs = model(*data[:-1])  # FeedForward

                # ----------------------------- BackPropagation
                optimizer.zero_grad()

                # ----------------------------- L1 Regularization
                l1_lambda = kwargs['l1_lambda'] if 'l1_lambda' in kwargs else 0
                l1_regularization = torch.tensor(0.0).to(device)
                for param in model.parameters():
                    l1_regularization += torch.norm(param, p=1)

                loss = loss_fn(outputs, labels) + l1_lambda * l1_regularization
                loss.backward()
                optimizer.step()

                # ----------------------------- Update Metrics
                if self.is_segmentation:
                    predicted = torch.zeros(labels.size()).to(labels.get_device())
                    idx = outputs[:, 1, :, :] > 0.5
                    predicted[idx] = 1
                    predicted[predicted != 1] = 0
                    # self.visualize_images_with_masks(data[0], labels, predicted)


                else:
                    _, predicted = torch.max(outputs.data, dim=1)
                train_confusion.update(predicted, labels)
                train_loss.update(loss.item())

                # ----------------------------- Print Metrics
                # TODO: Move this to somewhere else FGS
                if (i == 0 or i % 2 == 1 or i == train_n_batch - 1) and \
                        ('verbose' not in kwargs or kwargs['verbose'] is False):
                    done = epoch_number * 10 // epochs
                    remain = 10 - done
                    est = datetime.timedelta(seconds=time_estimator.get_time()) if remaining_time else "infinity"
                    if self.is_segmentation:
                        metrics = (f'Dice[{train_confusion.get_value("dice"):0.2f}|' +
                                   f'{train_dice_value:0.2f}|{train_confusion.get_best("dice"):0.2f}]' +
                                   f'Sens[{train_confusion.get_value("sensitivity"):0.2f}|{train_sens_value:0.2f}|' +
                                   f'{train_confusion.get_best("sensitivity"):0.2f}]')
                        val_metrics = (f'Dice[{validation_confusion.get_value("dice"):0.2f}|' +
                                       f'{validation_dice_value:0.2f}|{validation_confusion.get_best("dice"):0.2f}]' +
                                       f'Sens[{validation_confusion.get_value("sensitivity"):0.2f}|' +
                                       f'{validation_sens_value:0.2f}|' +
                                       f'{validation_confusion.get_best("sensitivity"):0.2f}]')

                    else:
                        metrics = (f'Acc[{train_confusion.get_value("accuracy"):0.2f}|' +
                                   f'{train_accuracy_value:0.2f}|{train_confusion.get_best("accuracy"):0.2f}] ' +
                                   f'Sens[{train_confusion.get_value("sensitivity"):0.2f}|' +
                                   f'{train_sens_value:0.2f}|{train_confusion.get_best("sensitivity"):0.2f}] ' +
                                   f'Spec[{train_confusion.get_value("specificity"):0.2f}|' +
                                   f'{train_spec_value:0.2f}|{train_confusion.get_best("specificity"):0.2f}]')
                        val_metrics = (f'Acc[{validation_confusion.get_value("accuracy"):0.2f}|' +
                                       f'{validation_accuracy_value:0.2f}|' +
                                       f'{validation_confusion.get_best("accuracy"):0.2f}] ' +
                                       f'Sens[{validation_confusion.get_value("sensitivity"):0.2f}|' +
                                       f'{validation_sens_value:0.2f}|{validation_confusion.get_best("sensitivity"):0.2f}] ' +
                                       f'Spec[{validation_confusion.get_value("specificity"):0.2f}|' +
                                       f'{validation_spec_value:0.2f}|' +
                                       f'{validation_confusion.get_best("specificity"):0.2f}]')

                    print(('\r' + f'Epoch [{epoch_number + 1}/{epochs}] | Batch [{i + 1}/{train_n_batch}] | '
                           + '=' * (done - 1)) + '>' + (' ' * remain) +
                          f'| EST:{est} | Train: {metrics} | Val: {val_metrics}',
                          end='')
                time_estimator.sub_step()

            if validation_loader is not None:
                model.eval()  # Set model to evaluation mode
                with torch.no_grad():
                    # ----------------------------- Validate Model
                    for batch, data in enumerate(validation_loader):
                        data = list(map(lambda x: x.to(device), data))
                        labels = data[-1]
                        if self.is_segmentation:
                            labels = labels.squeeze(1)

                            # #change for refine-net
                            # labels = labels.long()
                            # labels = labels.float()
                            # labels = torch.nn.functional.avg_pool2d(labels, kernel_size=4, stride=4, padding=0)
                        outputs = model(*data[:-1])

                        # True one
                        if self.is_segmentation:
                            predicted = torch.zeros(labels.size()).to(labels.get_device())
                            idx = outputs[:, 1, :, :] > 0.5
                            predicted[idx] = 1
                            predicted[predicted != 1] = 0
                            # predicted = torch.argmax(outputs, dim=1)
                            # predicted = predicted.unsqueeze(1)
                            train_confusion.update(predicted, labels)
                            train_loss.update(loss.item())


                        else:
                            _, predicted = torch.max(outputs.data, dim=1)

                        if not self.is_segmentation and kwargs.get('verbose', False):
                            print('output:', predicted)
                            print('labels:', labels)
                            print('--------------------------------------')

                        validation_confusion.update(predicted, labels)
                        loss = loss_fn(outputs, labels)
                        validation_loss.update(loss.item())
                        if 'verbose' not in kwargs or kwargs['verbose'] is False:
                            done = epoch_number * 10 // epochs
                            remain = 10 - done
                            est = datetime.timedelta(
                                seconds=time_estimator.get_time()) if remaining_time else "infinity"
                            if self.is_segmentation:
                                metrics = (f'Dice[{train_confusion.get_value("dice"):0.2f}|' +
                                           f'{train_dice_value:0.2f}|{train_confusion.get_best("dice"):0.2f}]' +
                                           f'Sens[{train_confusion.get_value("sensitivity"):0.2f}|{train_sens_value:0.2f}|' +
                                           f'{train_confusion.get_best("sensitivity"):0.2f}]')
                                val_metrics = (f'Dice[{validation_confusion.get_value("dice"):0.2f}|' +
                                               f'{validation_dice_value:0.2f}|{validation_confusion.get_best("dice"):0.2f}]' +
                                               f'Sens[{validation_confusion.get_value("sensitivity"):0.2f}|' +
                                               f'{validation_sens_value:0.2f}|' +
                                               f'{validation_confusion.get_best("sensitivity"):0.2f}]')

                            else:
                                metrics = (f'Acc[{train_confusion.get_value("accuracy"):0.2f}|' +
                                           f'{train_accuracy_value:0.2f}|{train_confusion.get_best("accuracy"):0.2f}] ' +
                                           f'Sens[{train_confusion.get_value("sensitivity"):0.2f}|' +
                                           f'{train_sens_value:0.2f}|{train_confusion.get_best("sensitivity"):0.2f}] ' +
                                           f'Spec[{train_confusion.get_value("specificity"):0.2f}|' +
                                           f'{train_spec_value:0.2f}|{train_confusion.get_best("specificity"):0.2f}]')
                                val_metrics = (f'Acc[{validation_confusion.get_value("accuracy"):0.2f}|' +
                                               f'{validation_accuracy_value:0.2f}|' +
                                               f'{validation_confusion.get_best("accuracy"):0.2f}] ' +
                                               f'Sens[{validation_confusion.get_value("sensitivity"):0.2f}|' +
                                               f'{validation_sens_value:0.2f}|' +
                                               f'{validation_confusion.get_best("sensitivity"):0.2f}] ' +
                                               f'Spec[{validation_confusion.get_value("specificity"):0.2f}|' +
                                               f'{validation_spec_value:0.2f}|' +
                                               f'{validation_confusion.get_best("specificity"):0.2f}]')

                            print(
                                '\r' + f'Epoch [{epoch_number + 1}/{epochs}]' +
                                f'| Validating...[{batch + 1}/{val_n_batch}] | '
                                + '=' * (done - 1) + '>' + (' ' * remain) +
                                f'| EST:{est} | Train: {metrics} | Val: {val_metrics}',
                                end='')

            # ----------------------------- Get final metric results for epoch
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

            train_loss_saver.update_value(train_loss_value, model)
            if validation_loader is not None:
                if self.is_segmentation:
                    val_dice_saver.update_value(validation_dice_value, model)
                    val_sens_saver.update_value(validation_sens_value, model)
                else:
                    val_accuracy_saver.update_value(validation_accuracy_value, model)

                val_loss_saver.update_value(validation_loss_value, model)

            end_time = datetime.datetime.now()
            time_estimator.update((end_time - start_time).total_seconds())
            remaining_time = time_estimator.get_time()
            if kwargs.get('verbose', False):
                print(
                    f"Epoch [{epoch_number + 1}/{epochs}]" +
                    f" | Estimated Remaining Time: {datetime.timedelta(seconds=remaining_time)}")
                print(f'Train Loss: {train_loss_value:.4f} | Train Accuracy: {train_accuracy_value:.2f}%' +
                      f'[{train_confusion.get_best("accuracy")}%]')
                if validation_loader is not None:
                    print(
                        f'Validation Loss: {validation_loss_value:.4f}% |' +
                        f'Validation Accuracy: {validation_accuracy_value:.2f}%' +
                        f'[{validation_confusion.get_best("accuracy"):0.2f}%]')
                print(
                    '=================================================================================================')
            else:
                done = epoch_number * 10 // epochs
                remain = 10 - done
                est = datetime.timedelta(seconds=time_estimator.get_time()) if remaining_time else "infinity"
                if self.is_segmentation:
                    metrics = (f'Dice[{train_confusion.get_value("dice"):0.2f}|' +
                               f'{train_dice_value:0.2f}|{train_confusion.get_best("dice"):0.2f}]' +
                               f'Sens[{train_confusion.get_value("sensitivity"):0.2f}|{train_sens_value:0.2f}|' +
                               f'{train_confusion.get_best("sensitivity"):0.2f}]')
                    val_metrics = (f'Dice[{validation_confusion.get_value("dice"):0.2f}|' +
                                   f'{validation_dice_value:0.2f}|{validation_confusion.get_best("dice"):0.2f}]' +
                                   f'Sens[{validation_confusion.get_value("sensitivity"):0.2f}|' +
                                   f'{validation_sens_value:0.2f}|' +
                                   f'{validation_confusion.get_best("sensitivity"):0.2f}]')

                else:
                    metrics = (f'Acc[{train_confusion.get_value("accuracy"):0.2f}|' +
                               f'{train_accuracy_value:0.2f}|{train_confusion.get_best("accuracy"):0.2f}] ' +
                               f'Sens[{train_confusion.get_value("sensitivity"):0.2f}|' +
                               f'{train_sens_value:0.2f}|{train_confusion.get_best("sensitivity"):0.2f}] ' +
                               f'Spec[{train_confusion.get_value("specificity"):0.2f}|' +
                               f'{train_spec_value:0.2f}|{train_confusion.get_best("specificity"):0.2f}]')
                    val_metrics = (f'Acc[{validation_confusion.get_value("accuracy"):0.2f}|' +
                                   f'{validation_accuracy_value:0.2f}|' +
                                   f'{validation_confusion.get_best("accuracy"):0.2f}] ' +
                                   f'Sens[{validation_confusion.get_value("sensitivity"):0.2f}|' +
                                   f'{validation_sens_value:0.2f}|' +
                                   f'{validation_confusion.get_best("sensitivity"):0.2f}] ' +
                                   f'Spec[{validation_confusion.get_value("specificity"):0.2f}|' +
                                   f'{validation_spec_value:0.2f}|' +
                                   f'{validation_confusion.get_best("specificity"):0.2f}]')
                print(('\r' + f'Epoch [{epoch_number + 1}/{epochs}] | Batch [{train_n_batch}/{train_n_batch}] | '
                       + '=' * (done - 1)) + '>' + (' ' * remain) +
                      f'| EST:{est} | Train: {metrics} | Val: {val_metrics}',
                      end='')

            train_confusion.reset()
            train_loss.reset()
            if validation_loader is not None:
                validation_confusion.reset()
                validation_loss.reset()
        train_confusion.save(os.path.join(save_path, 'train_history'))
        train_loss.save(os.path.join(save_path, 'train_loss_history.csv'))
        validation_confusion.save(os.path.join(save_path, 'val_history'))
        validation_loss.save(os.path.join(save_path, 'val_loss_history.csv'))

        model.save_all(os.path.join(save_path, 'last_model.pth'))

        if test_loader is not None:
            print('\n================= Running on test dataset =================')
            test_loss = Loss()
            test_confusion = ConfusionMatrix()
            model.eval()

            with torch.no_grad():
                for data in test_loader:
                    data = list(map(lambda x: x.to(device), data))
                    labels = data[-1]

                    # if self.is_segmentation:
                    #     # change for refine-net
                    #     labels = labels.long()
                    #     labels = labels.float()
                    #     labels = torch.nn.functional.avg_pool2d(labels, kernel_size=4, stride=4, padding=0)
                    outputs = model(*data[:-1])

                    # TODO: Move this block to where you process predictions
                    # if self.is_segmentation:
                    #     predicted = torch.zeros(labels.size()).to(labels.get_device())
                    #     idx = outputs[:, 1, :, :] > 0.5
                    #     predicted[:, 0, :, :][idx] = 1  # Assuming you have a single channel output
                    #     predicted[:, 0, :, :][predicted[:, 0, :, :] != 1] = 0
                    #     labels = labels.squeeze(1)

                    if self.is_segmentation:
                        predicted = torch.zeros(labels.size()).to(labels.get_device())
                        idx = outputs[:, 1, :, :] > 0.5
                        predicted[idx] = 1
                        predicted[predicted != 1] = 0



                    else:
                        _, predicted = torch.max(outputs.data, dim=1)
                    if not self.is_segmentation:
                        print('output:', predicted)
                        print('labels:', labels)
                        print('--------------------------------------')
                    test_confusion.update(predicted, labels)

                    loss = loss_fn(outputs, labels)
                    test_loss.update(loss.item())

            test_loss_value = test_loss.get_loss()
            test_sensitivity_value = test_confusion.get_value("sensitivity")
            result = ''
            name = 'test_result'
            if not self.is_segmentation:
                test_accuracy_value = test_confusion.get_value("accuracy")
                test_specificity_value = test_confusion.get_value("specificity")
                result += f'Test Loss: {test_loss_value:.4f} | Test Accuracy: {test_accuracy_value:.2f}%\n'
                result += f'Test Sensitivity: {test_sensitivity_value:.2f}% |'
                result += f'Test Specificity: {test_specificity_value:.2f}%'
            else:
                test_dice_value = test_confusion.get_value("dice")
                result += f'Test Loss: {test_loss_value:.4f} | Test Dice: {test_dice_value:.2f}%\n'
                result += f'Test Sensitivity: {test_sensitivity_value:.2f}%'
                name += '_seg'

            with open(f'{save_path}/{name}.txt', 'w') as file:
                file.write(result)
            print(result)

        return {
            'train_confusion': train_confusion,
            'train_loss': train_loss,
            'val_confusion': validation_confusion,
            'val_loss': validation_loss
        }

    def evaluate(self, device, loss_fn, save_path, model_path=None):
        if model_path is not None:
            self.model.load_all(model_path)
        _, _, test_loader, _, _, _ = self.get_loader()
        model = self.model
        print('\n================= Running on test dataset [Evaluation] =================')
        test_confusion = ConfusionMatrix()
        test_loss = Loss()

        with torch.no_grad():
            for i, data in enumerate(test_loader):
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
                if not self.is_segmentation:
                    print('output:', predicted)
                    print('labels:', labels)
                    print('--------------------------------------')
                test_confusion.update(predicted, labels)

                loss = loss_fn(outputs, labels)
                test_loss.update(loss.item())

                # Visualize images every 10 batches

                # if i % 10 == 0:
                #     self.visualize_images_with_masks(data[0], labels, predicted)

        test_loss_value = test_loss.get_loss()
        test_sensitivity_value = test_confusion.get_value("sensitivity")
        result = ''
        name = 'test_best_result'
        if not self.is_segmentation:
            test_accuracy_value = test_confusion.get_value("accuracy")
            test_specificity_value = test_confusion.get_value("specificity")
            result += f'Test Loss: {test_loss_value:.4f} | Test Accuracy: {test_accuracy_value:.2f}%\n'
            result += f'Test Sensitivity: {test_sensitivity_value:.2f}% |'
            result += f'Test Specificity: {test_specificity_value:.2f}%'
        else:
            test_dice_value = test_confusion.get_value("dice")
            result += f'Test Loss: {test_loss_value:.4f} | Test Dice: {test_dice_value:.2f}%\n'
            result += f'Test Sensitivity: {test_sensitivity_value:.2f}%'
            name += '_seg'

        with open(f'{save_path}/{name}.txt', 'w') as file:
            file.write(result)
        print(result)
