import math
import torch
import logging

from mamba_ssm import Mamba
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MixerModel

logger = logging.getLogger()


class MambaModel(torch.nn.Module):
    def __init__(self, vocab_size_in, vocab_size_out, d_channels, d_state, layers,
                 d_conv=4, d_expand=2, device='cpu', debug=False):
        super(MambaModel, self).__init__()
        self.debug = debug
        ssm_cfg = dict()
        ssm_cfg['d_conv'] = d_conv
        ssm_cfg['expand'] = d_expand
        ssm_cfg['d_state'] = d_state
        self.linear = torch.nn.Linear(d_channels, vocab_size_out)

        self.mixer_model = MixerModel(d_model=d_channels,n_layer=layers, vocab_size=vocab_size_in,
                                      ssm_cfg=ssm_cfg, device=device)

    def forward(self, input):
        # pass the input ids and the attention mask to the pretrained model
        if self.debug is True:
          logger.info("shapes - Input : {} ".format(input.shape)) 
        mamba_output = self.mixer_model(input)
        output = self.linear(mamba_output)
        if self.debug is True:
          logger.info("shapes - Input : {} Temp : {} Final : {}".format(input.shape, mamba_output.shape, output.shape))
        return output


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class AlteranteMambaTrans(torch.nn.Module):
    def __init__(self, vocab_size_in, vocab_size_out, d_channels, d_state, layers, positional_mask, num_attn_heads,
                 d_conv=4, d_expand=2, device='cpu', debug=False):
        super(AlteranteMambaTrans, self).__init__()
        # Since the mamba and transformer layers are supposed to alternate, the number of layers has to be even
        assert layers % 2 == 0
        self.each_num_layers = int(layers/2)
        self.debug = debug
        self.d_channels = d_channels

        self.positional_mask = positional_mask
        
        self.token_embedding = torch.nn.Embedding(vocab_size_in, d_channels)
        self.pos_embedding = PositionalEncoding(d_channels)
        self.out_decode = torch.nn.Linear(d_channels, vocab_size_out)
        
        self.layers = torch.nn.ModuleList()
        
        # Add token and positional embeddings first
        self.layers.append(self.token_embedding)
        self.layers.append(self.pos_embedding)

        for x in range(self.each_num_layers):
            self.layers.append(Mamba(d_model=d_channels, d_state=d_state, d_conv=d_conv, expand=d_expand))
            self.layers.append(torch.nn.TransformerEncoderLayer(d_model=d_channels, nhead=num_attn_heads))

        # Add the output decode layer in the end 
        self.layers.append(self.out_decode)

    def forward(self, input, device='cpu'):
        # pass the input ids and the attention mask to the pretrained model
        if self.debug is True:
            logger.info("shapes - Input : {} ".format(input.shape)) 

        # Input token embedding 
        input = self.layers[0](input) * math.sqrt(self.d_channels)
        if self.positional_mask is not True:
            # Add the positional embedding to the token embedding
            self.layers[1](input)

        for layer_num, layer in enumerate(self.layers[:-1]):
            if layer_num < 2:
                # Since they correspond to the input & positional embedding
                continue
            if layer_num % 2 == 0:
                # Then its a Mamba layer, as they are added before the Transformers                
                input = layer(input)
                # RMSNorm. Refine later, again to check dim=2 vs dim=1
                input = input / (1e-8 + input.pow(2).sum(dim=2, keepdim=True)).sqrt()
            else:
                # We are using the transformer layer
                if self.positional_mask is True:
                    # with causal masking
                    src_mask = torch.nn.Transformer.generate_square_subsequent_mask(input.size()[0]).to(device)
                    input = layer(input, src_mask=src_mask, is_causal=True)
                else: 
                    # without causal masking
                    input = layer(input)

        # Output decode back
        output = self.layers[-1](input)

        if self.debug is True:
          logger.info("shapes - Input : {} Temp : {} Final : {}".format(input.shape, mamba_output.shape, output.shape))
        return output


class TransformerOnly(torch.nn.Module):
    def __init__(self, vocab_size_in, vocab_size_out, d_channels, d_state, layers, positional_mask, num_attn_heads,
                 d_conv=4, d_expand=2, device='cpu', debug=False):
        super(TransformerOnly, self).__init__()
        # Since the mamba and transformer layers are supposed to alternate, the number of layers has to be even
        assert layers % 2 == 0
        self.each_num_layers = int(layers/2)
        self.debug = debug
        self.d_channels = d_channels

        self.positional_mask = positional_mask
        
        self.token_embedding = torch.nn.Embedding(vocab_size_in, d_channels)
        self.pos_embedding = PositionalEncoding(d_channels)
        self.out_decode = torch.nn.Linear(d_channels, vocab_size_out)
        
        self.layers = torch.nn.ModuleList()
        
        # Add token and positional embeddings first
        self.layers.append(self.token_embedding)
        self.layers.append(self.pos_embedding)

        for x in range(self.each_num_layers):
            self.layers.append(torch.nn.TransformerEncoderLayer(d_model=d_channels, nhead=num_attn_heads))
            self.layers.append(torch.nn.TransformerEncoderLayer(d_model=d_channels, nhead=num_attn_heads))            

        # Add the output decode layer in the end 
        self.layers.append(self.out_decode)

    def forward(self, input, device='cpu'):
        # pass the input ids and the attention mask to the pretrained model
        if self.debug is True:
            logger.info("shapes - Input : {} ".format(input.shape)) 

        # Input token embedding 
        input = self.layers[0](input) * math.sqrt(self.d_channels)
        if self.positional_mask is not True:
            # Add the positional embedding to the token embedding
            self.layers[1](input)

        for layer_num, layer in enumerate(self.layers[:-1]):
            if layer_num < 2:
                # Since they correspond to the input & positional embedding
                continue
            # We are using the transformer layer
            if self.positional_mask is True:
                # with causal masking
                src_mask = torch.nn.Transformer.generate_square_subsequent_mask(input.size()[0]).to(device)
                input = layer(input, src_mask=src_mask, is_causal=True)
            else: 
                # without causal masking
                input = layer(input)

        # Output decode back
        output = self.layers[-1](input)
        return output
