import torch
import wandb
import hydra

import src.train_utils as train_utils
from src.dataset_utils import create_dataloader, build_lang_config, get_flip_flop_dataset

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


@hydra.main(config_path='configs', config_name="defaults", version_base=None)
def run_pipeline(cfg: DictConfig) -> None:
    use_wandb = cfg.basic.use_wandb
    if use_wandb is True:
        # Login into the wandb system
        cfg_copy = OmegaConf.to_container(cfg, resolve=True)
        wandb.login(key=settings.WANDB_API_KEY)
        wandb.init(project=cfg.dataset.name, entity=settings.WANDB_TEAM,
                   name=cfg.basic.wandb_run, config=cfg_copy)

    lossfn = hydra.utils.instantiate(cfg.loss)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    # device = 'cpu'

    print(" DEVICE & LOSSFN DONE ")

    if cfg.basic.is_flip_flop is True:
        run_flip_flop_pipeline(cfg, lossfn, device, use_wandb)
    else:
        run_formal_lang_pipeline(cfg, lossfn, device, use_wandb)

    if use_wandb is True:
        wandb.finish()


def run_flip_flop_pipeline(cfg: DictConfig, lossfn, device, use_wandb: bool) -> None:

    dataset_name, batch_size, model_name = cfg.dataset.name, cfg.dataset.batch_size, cfg.model.name
    # Get the dataset, dataloader from the datasets library transformers
    dataloader_dict = get_flip_flop_dataset(batch_size=batch_size)
    
    # For flip flop, just have to remember, Either 0 or 1 or ignore, but shape has to be the same as input
    vocab_size_in, vocab_size_out = 6, 3
    
    # Initialise the model with all the parameters from the Yaml file
    model = get_model(cfg, model_name, vocab_size_in, vocab_size_out, device)

    # The learning rate is added to the optimizer by default, model parameters added manually
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())


    # for name, param in model.named_parameters():
    #     if param.requires_grad is True:
    #         print(f"Layer: {name} | Size: {param.size()} \n")

    # Add all the other non sense about hyperparameters here
    if cfg.basic.use_scheduler is True:
        # scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)
        scheduler = None
    else:
        scheduler = None

    num_epochs = cfg.train.epochs
    # Add argument to train_model
    train_utils.train_flip_flop(model=model, lossfn=lossfn, device=device, epochs= num_epochs, optimizer=optimizer,
                                scheduler=scheduler, dataloader_dict=dataloader_dict, use_wandb=use_wandb, debug=cfg.basic.debug)


def run_formal_lang_pipeline(cfg: DictConfig, lossfn, device, use_wandb: bool) -> None:
    pad_value = cfg.basic.pad_value
    max_seq_size, generate_dataset = cfg.basic.max_seq_size, cfg.basic.generate_dataset
    is_multiple, num_val_bins = cfg.dataset.multiple, cfg.dataset.num_val_bins
    dataset_name, batch_size, model_name = cfg.dataset.name, cfg.dataset.batch_size, cfg.model.name

    # If required, build the language config from the hydra dataset config
    lang_params = build_lang_config(cfg.dataset) if generate_dataset else None

    dataloader_dict, dataset = create_dataloader(base_folder=dataset_name, batch_size=batch_size, lang_params=lang_params,
                                                 is_multiple=is_multiple, pad_value=pad_value, max_size=max_seq_size,
                                                 num_val_bins=num_val_bins, generate=generate_dataset, is_entropy_loss=False)

    # To account for padding token 
    vocab_size_in = dataset.vocab_inp.shape[0] + 1
    vocab_size_out = dataset.vocab_out.shape[0]
    # To create the mask and separate genuine one hot vectors from the padding tensors
    pad_tensor_out = torch.tensor([pad_value for _ in range(vocab_size_out)]).to(device)
    model = get_model(cfg, model_name, vocab_size_in, vocab_size_out, device)

    # The learning rate is added to the optimizer by default, model parameters added manually
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    if cfg.basic.use_scheduler is True:
        # scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)
        scheduler = None
    else:
        scheduler = None

    num_epochs = cfg.train.epochs
    print(" STARTING TRAINING ")
    # Add argument to train_model
    train_utils.train_model(model=model, lossfn=lossfn, device=device, epochs= num_epochs, optimizer=optimizer,
                            scheduler=scheduler, dataloader_dict=dataloader_dict, use_wandb=use_wandb, dataset=dataset,
                            debug=cfg.basic.debug, pad_tensor=pad_tensor_out)

    if use_wandb is True:
        wandb.finish()

if __name__ == "__main__":
    run_pipeline()
