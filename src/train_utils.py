import torch
import wandb
from tqdm import tqdm
import numpy as np
from pathlib import Path
import logging
from src.dataset_utils import *
from typing import Tuple


logger = logging.getLogger()


def get_batch_accuracy_mse(predictions: torch.Tensor, targets: torch.Tensor, mask: Dict, decoder: torch.Tensor, debug: bool = True) -> Tuple[int, int]:
    """Batch Accuracy for MSE (Scope for optimization here)

    Args:
        predictions (torch.tensor): Shape - (batch, seq, vocab_size)
        targets (torch.tensor): Shape - (batch, seq, vocab_size)
        mask (torch.tensor): Tensor of the shape (batch, seq). Contains False as an entry, if target has a padding vector there
        decoder (dict): Contains mappings from the index to the actual character in the vocabulary
        debug (bool, optional): If True, save logs with decoded output to understand the patterns of errors

    Returns:
        int: The number of correct matches b/w targets and predictions
        int: The number of correct matches b/w targets and predictions
    """
    target_batch = targets.to("cpu").detach().numpy()
    predicted_batch = predictions.to("cpu").detach().numpy()
    mask_batch = mask.to("cpu").detach().numpy()
    
    curr_correct_relax, curr_correct_strict = 0, 0
    for target, predicted, mask_curr in zip(target_batch, predicted_batch, mask_batch):
        # apply mask to remove padding tensors, hence size of the sequence shortened
        target_filtered = target[mask_curr]
        # argmax to get index and not compare one hot vectors
        target_seq = np.argmax(target_filtered, axis=1)
        # repeat for predicted tensor
        predicted_filtered = predicted[mask_curr]
        # note - argmax taken, instead of using 0.5 as a threshold for shifting to 1 / 0. 
        # Arbitrary choice, Might be changed, to experiment / compare performance later
        predicted_seq = np.argmax(predicted_filtered, axis=1)
        if debug:
            # Decode based on vocab
            target_str = "".join([str(decoder.get(i, '#')) for i in target_seq])
            predicted_str = "".join([str(decoder.get(i, '#')) for i in predicted_seq])
            logger.info(" Target: {} Predicted : {}".format(target_seq, predicted_seq))
            logger.info('Predicted: {} Target: {} Equal: {}'.format(predicted_str, target_str, np.equal(target_seq, predicted_seq).all()))

        compared_seq = np.equal(target_seq, predicted_seq)
        curr_correct_relax += compared_seq.mean()
        curr_correct_strict += compared_seq.all()
    return curr_correct_relax, curr_correct_strict


def train_model(model, lossfn, device, epochs, optimizer, scheduler, dataloader_dict,
                dataset, use_wandb: bool=False, debug: bool=False, pad_tensor=None):
    """
    Args:
        model (torch.nn.Module): Model (1, 2 or 3 layers of Mamba)
        lossfn (torch.nn.MSELoss): Loss function
        device (torch.device): The GPU device on which to run the pipeline
        epochs (int): Number of epochs to train for
        optimizer (torch.optim.Optimizer): Pytorch Optimizer, currently AdamW
        scheduler (torch.optim.lr_scheduler, None): LR Scheduler, might be OneCycleLR or None
        dataloader_dict (dict): Key - train, or val_bin_number, and the values are the corresponding dataloaders
        dataset (torch.utils.data.Dataset): The Dataset object, whose decoder will be used in the debug mode
        use_wandb (bool, optional): If True, save logging information for wandb, else not. Defaults to False.
        debug (bool, optional): If True, save intermediate logs for help. Defaults to False.
        pad_tensor (torch.tensor, optional): In case of MSELoss, It's of shape (vocab_size) with each entry being the padding index, to filter
            out the padding entries while computing loss
    """
    model.train()
    for epoch in tqdm(range(int(epochs))):
        logging_dict = {}
        for category, dataloader in dataloader_dict.items():
            if category == 'train':
                model.train()
            else:
                # No need for going over validation sets, when learning hasn't even properly started
                # Go for validation sets, once training is about to finish
                if epochs - epoch > 2:
                    continue
                model.eval()

            epoch_correct_relax, epoch_correct_strict, epoch_total, epoch_loss = 0, 0, 0, 0
            for data in dataloader:
                inputs, targets = data[0], data[1]
                inputs = inputs.to(device)
                targets = targets.to(device)

                if category == 'train':
                    optimizer.zero_grad()

                predicted = model(inputs)
                #print(" Shape inputs - {} targets - {} predicted - {} pad_tensor - {}".format(inputs.shape, targets.shape, predicted.shape, pad_tensor.shape))
                # Find all the padding vector locations in case of MSE
                mask = torch.all(torch.eq(targets, pad_tensor), dim=-1)
                # Invert the mask, since we are more interested in the tokens that are not padding tokens
                mask = ~mask
                # Apply Masks to both targets and predictions
                mask_expanded = mask.unsqueeze(-1).repeat(1, 1, targets.shape[-1])
                # Hence after applying these masks, the padding_tensors will be replaced by (0, 0, .. 0), hence they won't contribute to the Loss function
                masked_target = targets * mask_expanded
                #print(" Mask expand {} mask target - {}".format(mask_expanded.shape, masked_target.shape))
                masked_prediction = predicted * mask_expanded
                #print("mask expanded -{} masked_target - {} maked prediction - {}".format(mask_expanded.size(), masked_target.size(), masked_prediction.size()))
                # Accuracy & Loss computation
                correct_relax, correct_strict = get_batch_accuracy_mse(predicted, targets, mask, dataset.
                decoder_out, debug=debug)
                loss = lossfn(masked_prediction, masked_target)

                if category == 'train':
                    loss.backward()
                    # Optimizer Step and scheduler step
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()

                epoch_correct_relax += correct_relax
                epoch_correct_strict += correct_strict
                # Total is just the total number of examples across the entire training set
                epoch_total += inputs.shape[0]
                epoch_loss += loss.item()

            relax_accuracy, strict_accuracy = 100*epoch_correct_relax/ epoch_total, 100*epoch_correct_strict/ epoch_total
            logging_dict[category + "_loss"] = epoch_loss
            logging_dict[category + "_accuracy_relax"] = relax_accuracy
            logging_dict[category + "_accuracy_strict"] = strict_accuracy
            logger.info(" Category - {} Relax Accuracy - {} Strict Accuracy - {} ".format(category, relax_accuracy, strict_accuracy))

            if use_wandb is True:
                wandb.log(logging_dict)


def get_batch_accuracy_flip_flop(predictions: torch.Tensor, targets: torch.Tensor, debug: bool = False) -> int:
    """Batch Accuracy for MSE
    Args:
        predictions (torch.tensor): Shape - (batch, seq, vocab_size)
        targets (torch.tensor): Shape - (batch, seq, vocab_size)
        debug (bool, optional): If True, save logs with decoded output to understand the patterns of errors

    Returns:
        int: The number of correct matches b/w targets and predictions
    """
    target_batch = targets.to("cpu").detach().numpy()
    predicted_batch = predictions.to("cpu").detach().numpy()

    # After taking argmax, from (batch, seq, channels) -> (batch, seq)
    target_batch = np.argmax(target_batch, axis=2)
    predicted_batch = np.argmax(predicted_batch, axis=2)


    if debug is True:
        target_str, pred_str =  ''.join(target_batch[0].astype(str)), ''.join(predicted_batch[0].astype(str))
        num_differs = sum(c1 != c2 for c1, c2 in zip(target_str, pred_str))
        logger.info("Target - {} Predicted - {} Diff - {}".format(target_str, pred_str, num_differs))

    # Check equality per sequence, entire sequence should be correct for increment
    return np.equal(target_batch, predicted_batch).all(axis=1).sum()


def execution_loop_flip_flop(data, lossfn, device, optimizer, model, scheduler, curr_total, curr_correct, curr_loss, idx, is_train=True):
    inputs, targets = data['inputs'], data['outputs']
    inputs = inputs.to(device)
    targets = targets.to(device)

    if is_train is True:
        optimizer.zero_grad()

    inputs = inputs.to(torch.int32)
    predicted = model(inputs)
    debug = True if is_train is True and idx % 97 == 0 else False
    correct = get_batch_accuracy_flip_flop(predicted, targets, debug=debug)
    loss = lossfn(predicted, targets)
    
    if is_train is True:
        loss.backward()
        # Optimizer Step and scheduler step
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    # # Total is just the total number of examples across the entire training set
    curr_loss += loss.item()
    curr_correct += correct
    curr_total += inputs.shape[0]

    return curr_loss, curr_correct, curr_total


def log_flip_flop(curr_correct, curr_total, curr_loss, use_wandb, category):

    logging_dict = {}
    curr_accuracy = 100*curr_correct/ curr_total
    logging_dict[category + "_loss"] = curr_loss
    logging_dict[category + "_accuracy"] = curr_accuracy

    if use_wandb is True:
        wandb.log(logging_dict)


def train_flip_flop(model, lossfn, device, epochs, optimizer, scheduler, dataloader_dict, use_wandb: bool=False, debug: bool=False):
    """
    Args:
        model (torch.nn.Module): Model (1, 2 or 3 layers of Mamba with / or without embedding & Linear layer around it)
        lossfn (torch.nn.MSELoss): Loss function
        device (torch.device): The GPU device on which to run the pipeline
        epochs (int): Number of epochs to train for
        optimizer (torch.optim.Optimizer): Pytorch Optimizer, currently AdamW
        scheduler (torch.optim.lr_scheduler, None): LR Scheduler, might be OneCycleLR or None
        dataloader_dict (dict): Key - train, or val_bin_number, and the values are the corresponding dataloaders
        use_wandb (bool, optional): If True, save logging information for wandb, else not. Defaults to False.
        debug (bool, optional): If True, save intermediate logs for help. Defaults to False.
    """
    model.train()
    for epoch in tqdm(range(int(epochs))):
        logging_dict = {}

        model.train()
        # Reset values of loss, accuracy every epoch
        curr_correct, curr_total, curr_loss = 0, 0, 0        
        for idx, data in tqdm(enumerate(dataloader_dict['train'])):

            if idx % 100 == 0:
                if idx > 0:
                    # Switch to evaluation mode for the val_sparse and the val splits
                    model.eval()
                    # Start logging all the test values
                    for category, dataloader in dataloader_dict.items():
                        # According to the flip flop paper, only val_sparse, and val are reported every 100 steps 
                        if category == 'train':
                            continue
                        test_correct, test_total, test_loss = 0, 0, 0
                        # Iterate over the test dataloader, and collect the stats
                        for test_data in dataloader:
                            test_loss, test_correct, test_total = execution_loop_flip_flop(test_data, lossfn, device, optimizer, model, scheduler,
                                                                                           test_total, test_correct, test_loss, idx, is_train=False)
                        log_flip_flop(test_correct, test_total, test_loss, use_wandb, category)

                model.train()

            # Carry out the execution of training loop normally
            curr_loss, curr_correct, curr_total = execution_loop_flip_flop(data, lossfn, device, optimizer, model, scheduler,
                                                                           curr_total, curr_correct, curr_loss, idx, is_train=True)
        
        # Log the training values collected per 100 steps 
        log_flip_flop(curr_correct, curr_total, curr_loss, use_wandb, category='train')