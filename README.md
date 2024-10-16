## The Expressive Capacity of State Space Models: A Formal Language Perspective

This is the official repository of the [NeurIPS 2024 paper](https://nips.cc/virtual/2024/poster/94264) "The Exprgiessive Capacity of State Space Models: A Formal Language Perspective".

## Installation

To set up the environment and install the necessary packages, follow these steps:

1. Create a new conda environment:
   ```bash
   conda create -n mtest
   ```
2. Activate the conda environment:
   ```bash
   conda activate mtest
   ```
3. Install CUDA NVCC:
   ```bash
   conda install -c 'Nvidia/label/cuda-11.7.0' cuda-nvcc
   ```
4. Install PyTorch and related packages:
   ```bash
   conda install PyTorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 PyTorch-cuda=11.7 -c pytorch -c Nvidia
   ```
5. Install additional Python packages:
   ```bash
   pip install packaging
   pip install mamba-ssm
   pip install pandas hydra-core pydantic-settings python-decouple wandb
   ```

Save a `.env` file locally with the following parameters (if Wandb logging is to be enabled):
```
WANDB_API_KEY=__
WANDB_TEAM=__
```

## Running Experiments

### Bhattamishra Suite

#### With Cross Entropy Loss
To run an experiment with Cross Entropy Loss:
```bash
python train_with_ce.py basic.use_wandb=False train.epochs=100 dataset=dyck1
```

#### With Mean Squared Error Loss
To run an experiment with Mean Squared Error Loss:
```bash
python train_with_mse.py basic.use_wandb=False train.epochs=100 dataset=dyck1
```

You can change the `dataset` parameter to train on a specific dataset out of the 27 languages.

### Flip Flop Experiment
To run the Flip Flop experiment:
```bash
python train_with_mse.py basic.is_flip_flop=True dataset=flip_flop train.epochs=100
```

### Bounded Dyck Experiments
 
1. Install additional Python packages:
  ```bash
  pip install matplotlib seaborn
  ```

2. Switch to the correct folder
```bash
  cd bounded_dyck_expt
```

3. Run the experiment for a specific layer and d_model combination

```bash
python src/run_lm.py experiments/layer2_80.yaml
```

## Acknowledgements

Code co-authored by [Yash Sarrof](https://github.com/yashYRS) and [Yana Veitsman](https://github.com/yan-vei)
Code used here has been heavily inspired by the following repositories:

* [Mamba Official Implementation - Gu et al](https://github.com/state-spaces/mamba)
* [Recognising Formal Languages - Bhattmishra et al](https://github.com/satwik77/Transformer-Simplicity)
* [Dyck Transformer - Yao et al](https://github.com/princeton-nlp/dyck-transformer)
