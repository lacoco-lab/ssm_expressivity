import torch
import wandb
import hydra
from tqdm import tqdm

import src.train_utils as train_utils
from src.dataset_utils import create_dataloader, build_lang_config

from omegaconf import DictConfig, OmegaConf
from config import settings


def get_model(cfg, model_name, vocab_size_in, vocab_size_out, device):
    # Import the necessary model
    from src.mamba_model import MambaModel

    # Get the parameters to run the Mamba model
    d_conv, d_expand, layers = cfg.model.d_conv, cfg.model.d_expand, cfg.model.num_layers
    d_state, d_channels, model_debug = cfg.model.d_state, cfg.model.d_channels, cfg.model.debug
    # Initialise the model with all the parameters from the Yaml file
    model = MambaModel(vocab_size_in, vocab_size_out, d_channels, d_state, layers, d_conv, d_expand, device, model_debug)

    ## Add other SSM imports here if necessary
    model.to(device)
    return model


def train_with_ce(model, vocab_size_out, lossfn, device, epochs, optimizer, dataloader_dict, use_wandb):
    """
    Args:
        model (torch.nn.Module): Model (1, 2 or 3 layers of Mamba)
        lossfn (torch.nn.MSELoss): Loss function
        device (torch.device): The GPU device on which to run the pipeline
        epochs (int): Number of epochs to train for
        optimizer (torch.optim.Optimizer): Pytorch Optimizer, currently AdamW
        dataloader_dict (dict): Key - train, or val_bin_number, and the values are the corresponding dataloaders
    """
    model.train()
    for epoch in tqdm(range(int(epochs))):
        logging_dict = {}
        for category, dataloader in dataloader_dict.items():
            if category == 'train':
                model.train()
            else:
                # Go for validation sets, once training is about to finish
                model.eval()

            epoch_correct, epoch_total = 0, 0, 
            for data in dataloader:
                inputs, targets = data[0], data[1]
                inputs = inputs.to(device).long()
                targets = targets.to(device).long()

                predicted = model(inputs)
                loss = lossfn(predicted.view(-1, vocab_size_out), targets.view(-1)).mean()
                # Accuracy & Loss computation
                reqd_shape = targets.size()
                accuracy = (torch.argmax(predicted.view(-1, vocab_size_out), dim=1) == targets.view(-1)).reshape(reqd_shape).all(dim=1).float()
                epoch_correct += accuracy.sum().item()
                epoch_total += reqd_shape[0]
                if category == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    # Optimizer Step and scheduler step
                    optimizer.step()
            
            logging_dict = {
                f'{category}_Loss': loss.item(),
                f'{category}_Accuracy': epoch_correct/epoch_total
            }
            if use_wandb:
                wandb.log(logging_dict)
            else:
                print(logging_dict)



@hydra.main(config_path='configs', config_name="defaults", version_base=None)
def run_pipeline(cfg: DictConfig) -> None:

    use_wandb = cfg.basic.use_wandb
    if use_wandb is True:
        # Login into the wandb system
        cfg_copy = OmegaConf.to_container(cfg, resolve=True)
        wandb.login(key=settings.WANDB_API_KEY)
        wandb.init(project=cfg.dataset.name, entity=settings.WANDB_TEAM,
                   name=cfg.basic.wandb_run, config=cfg_copy)

    lossfn = torch.nn.CrossEntropyLoss(reduction='none')
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    print(" DEVICE & LOSSFN DONE ")
    
    pad_value = cfg.basic.pad_value
    max_seq_size, generate_dataset = cfg.basic.max_seq_size, cfg.basic.generate_dataset
    is_multiple, num_val_bins = cfg.dataset.multiple, cfg.dataset.num_val_bins
    dataset_name, batch_size, model_name = cfg.dataset.name, cfg.dataset.batch_size, cfg.model.name

    # If required, build the language config from the hydra dataset config
    lang_params = build_lang_config(cfg.dataset) if generate_dataset else None

    # Need alternate dataloader here for cross entropy
    dataloader_dict, dataset = create_dataloader(base_folder=dataset_name, batch_size=batch_size, lang_params=lang_params, is_multiple=is_multiple, pad_value=pad_value, max_size=max_seq_size, num_val_bins=num_val_bins, generate=generate_dataset, is_entropy_loss=True)
    print(" Dataloader done ")

    # To account for padding token 
    vocab_size_in = dataset.vocab_inp.shape[0] + 1
    vocab_size_out = dataset.vocab_out.shape[0] + 1
    #print(vocab_size_in, vocab_size_out)

    model = get_model(cfg, model_name, vocab_size_in, vocab_size_out, device)
    # The learning rate is added to the optimizer by default, model parameters added manually
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())

    num_epochs = cfg.train.epochs
    print(" STARTING TRAINING ")

    # Add argument to train_model
    train_with_ce(model=model, vocab_size_out=vocab_size_out, lossfn=lossfn, device=device, epochs=num_epochs, optimizer=optimizer, dataloader_dict=dataloader_dict, use_wandb=use_wandb)
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    run_pipeline()

