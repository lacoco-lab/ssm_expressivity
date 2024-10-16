import torch
from mamba_ssm.models.mixer_seq_simple import MixerModel


class MambaModel(torch.nn.Module):
    def __init__(self, args, input_size, hidden_size, num_layers):
        super(MambaModel, self).__init__()
        self.device = args['device'] 
        vocab_size = args['language']['vocab_size']
        self.mixer_model = MixerModel(d_model=hidden_size, n_layer=num_layers, vocab_size=vocab_size, device=self.device)
        # The final model should be made already here.
        self.mixer_model.to(self.device)

    def forward(self, input):
        # Expects return type of (batch_len, seq_type, hidden_size), hence no need for an extra decode linear layer
        return self.mixer_model(input), 0
