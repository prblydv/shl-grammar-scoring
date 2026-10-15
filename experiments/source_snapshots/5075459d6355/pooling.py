"""Explicit masked pooling definitions approved for configurable experiments."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

SUPPORTED=('mean','cls','attention','concat','weighted_layer','gem','lstm')


def masked_mean(hidden,mask):
    return (hidden.float()*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)


class Pooling(nn.Module):
    def __init__(self,config,hidden,layers):
        super().__init__();self.kind=config['pooling'];self.output_dim=hidden
        if self.kind not in SUPPORTED: raise ValueError('Unknown pooling: '+self.kind)
        self.needs_layers=self.kind in {'concat','weighted_layer'}
        if self.kind=='attention': self.attention=nn.Linear(hidden,1)
        if self.kind=='concat':
            self.count=config.get('concat_last_n_layers',4)
            if type(self.count) is not int or not 1<=self.count<=layers: raise ValueError('Invalid concat layer count')
            self.output_dim=hidden*self.count
        if self.kind=='weighted_layer':
            self.start=config.get('weighted_layer_start',0)
            if type(self.start) is not int or not 0<=self.start<layers: raise ValueError('Invalid weighted layer start')
            self.weights=nn.Parameter(torch.zeros(layers-self.start))
        if self.kind=='gem':
            p=float(config.get('gem_initial_p',3.0));self.epsilon=float(config.get('gem_epsilon',1e-6))
            if not math.isfinite(p) or p<=1 or not 0<self.epsilon<1: raise ValueError('GeM requires p>1 and epsilon in (0,1)')
            self.raw_p=nn.Parameter(torch.tensor(math.log(math.expm1(p-1))),requires_grad=config.get('gem_learnable_p',True))
        if self.kind=='lstm':
            size=config.get('lstm_hidden_size',128);count=config.get('lstm_num_layers',1)
            if type(size) is not int or size<1 or type(count) is not int or count<1: raise ValueError('Invalid LSTM shape')
            self.lstm=nn.LSTM(hidden,size,num_layers=count,batch_first=True,bidirectional=True,dropout=0.)
            self.output_dim=2*size

    def forward(self,output,mask):
        hidden=output.last_hidden_state;mask=mask.bool()
        if self.kind=='mean': return masked_mean(hidden,mask)
        if self.kind=='cls': return hidden[:,0].float()
        if self.kind=='attention':
            weights=self.attention(hidden).squeeze(-1).float().masked_fill(~mask,-1e4).softmax(1)
            return (hidden.float()*weights.unsqueeze(-1)).sum(1)
        if self.kind=='concat':
            return torch.cat([masked_mean(h,mask) for h in output.hidden_states[-self.count:]],dim=-1)
        if self.kind=='weighted_layer':
            layers=torch.stack(output.hidden_states[1+self.start:],dim=0).float()
            mixed=(layers*self.weights.softmax(0).view(-1,1,1,1)).sum(0)
            return masked_mean(mixed,mask)
        if self.kind=='gem':
            p=1+F.softplus(self.raw_p)
            hidden=hidden.float()
            powered=hidden.sign()*hidden.abs().clamp_min(self.epsilon).pow(p)
            mean=masked_mean(powered,mask)
            return mean.sign()*mean.abs().clamp_min(self.epsilon).pow(1/p)
        if self.kind=='lstm':
            lengths=mask.sum(1).cpu()
            if not torch.all(mask==(torch.arange(mask.shape[1],device=mask.device)[None,:]<lengths.to(mask.device)[:,None])):
                raise ValueError('LSTM pooling requires right padding')
            with torch.autocast(device_type=hidden.device.type,enabled=False):
                packed=pack_padded_sequence(hidden.float(),lengths,batch_first=True,enforce_sorted=False)
                packed,_=self.lstm(packed)
                sequence,_=pad_packed_sequence(packed,batch_first=True,total_length=mask.shape[1])
            return masked_mean(sequence,mask)
