import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, dim, activation=nn.GELU(), use_norm=True):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.activation = activation
        self.use_norm = use_norm
        if use_norm:
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        identity = x  
        out = self.fc1(x)
        if self.use_norm:
            out = self.norm1(out)
        out = self.activation(out)
        out = self.fc2(out)
        if self.use_norm:
            out = self.norm2(out)
        out = self.activation(out)
        return identity + out  
    

class LogDistributionModel(nn.Module):
    def __init__(self, data_dim=2, hidden_dim=256, num_res_blocks=1, time_encoding=True):
        super().__init__()
        self.time_encoding = time_encoding
        input_dim = data_dim + 2 if self.time_encoding else data_dim + 1
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.res_blocks = nn.Sequential(
        *[ResBlock(hidden_dim, activation=self.activation) for _ in range(num_res_blocks)]
        )
        self.output_layer = nn.Linear(hidden_dim, 1)


    def forward(self, t, x):
        batch_size = x.shape[0]

        if not torch.is_tensor(t):
            t = torch.tensor(t, device=x.device, dtype=x.dtype)
        
        if self.time_encoding:
            sin_t = torch.sin(0.1 * t)
            cos_t = torch.cos(0.1 * t)
            sin_t_tensor = torch.ones((batch_size, 1), device=x.device, dtype=x.dtype) * sin_t
            cos_t_tensor = torch.ones((batch_size, 1), device=x.device, dtype=x.dtype) * cos_t
            t_encoding = torch.cat([sin_t_tensor, cos_t_tensor], dim=1)  # [batch_size, 2]
            x_t = torch.cat([x, t_encoding], dim=1)  
        else:

            t_tensor = torch.ones((batch_size, 1), device=x.device, dtype=x.dtype) * t
            x_t = torch.cat([x, t_tensor], dim=1) 
        
        h = self.activation(self.input_layer(x_t))
        h = self.res_blocks(h)
        out = self.output_layer(h)
        return out


class MultiplierModel(nn.Module):
    def __init__(self, data_dim=2, hidden_dim=512, num_res_blocks=1, num_frequencies=0):

        super().__init__()
        self.num_frequencies = num_frequencies

        self.input_dim = data_dim + 1

        self.encoded_dim = self.input_dim + 2 * self.input_dim * self.num_frequencies



        self.input_layer = nn.Linear(self.encoded_dim, hidden_dim)

        self.res_blocks = nn.Sequential(*[ResBlock(hidden_dim) for _ in range(num_res_blocks)])
        self.output_layer = nn.Linear(hidden_dim, 1)

        self.activation = nn.GELU()

    # self._init_weights()


    def fourier_encode(self, x):

        out = [x]

        for i in range(self.num_frequencies):

            freq = 0.2 ** i
            out.append(torch.sin(x * freq))
            out.append(torch.cos(x * freq))

        return torch.cat(out, dim=-1)

    def forward(self, t, x):

        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=x.dtype, device=x.device)

        t_tensor = torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype) * t

        x_t = torch.cat([x, t_tensor], dim=1)

        encoded = self.fourier_encode(x_t)
        h = self.activation(self.input_layer(encoded))
        h = self.res_blocks(h)
        out = self.output_layer(h)
        return out
        




class InteractionModel(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.input_layer = nn.Linear(2, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.activation = nn.Tanh()

    def compute_net(self, x):
        h = self.activation(self.input_layer(x))
        h = self.activation(self.hidden_layer(h)) + h  
        out = self.output_layer(h)
        return out

    def forward(self, x_t):

        zero_input = torch.zeros((1, x_t.shape[-1]), device=x_t.device, dtype=x_t.dtype)
        baseline = self.compute_net(zero_input)
        potential = self.compute_net(x_t) + self.compute_net(-x_t) - 2 * baseline
        return potential

