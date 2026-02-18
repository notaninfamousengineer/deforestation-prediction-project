"""
ConvLSTM Model with optional CBAM Attention.
Supports multi-step prediction and separate input/output dimensions
(input has temporal encoding channels, output does not).
"""

import torch
import torch.nn as nn
from .attention import CBAM


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=True, bias=True, return_all_layers=False,
                 prediction_horizon=1, output_dim=None,
                 output_activation="tanh", use_attention=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim else input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.prediction_horizon = prediction_horizon
        self.output_activation = output_activation
        self.use_attention = use_attention

        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(
                input_dim=cur_input_dim,
                hidden_dim=self.hidden_dim[i],
                kernel_size=self.kernel_size[i],
                bias=self.bias
            ))

        self.cell_list = nn.ModuleList(cell_list)

        # Optional CBAM Attention
        if self.use_attention:
            self.attention = CBAM(self.hidden_dim[-1])

        # Output layer
        # Output channels = output_dim * prediction_horizon
        self.final_conv = nn.Conv2d(
            self.hidden_dim[-1],
            self.output_dim * prediction_horizon,
            kernel_size=1
        )

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: 5-D Tensor of shape (b, t, c, h, w) if batch_first=True
        
        Returns
        -------
        pred: Tensor of shape (b, output_dim, h, w) if horizon=1
              or (b, horizon, output_dim, h, w) if horizon > 1
        """
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, seq_len, _, h, w = input_tensor.size()

        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h_c_tuple = hidden_state[layer_idx]
            output_inner = []

            for t in range(seq_len):
                h_c_tuple = self.cell_list[layer_idx](cur_layer_input[:, t, :, :, :], h_c_tuple)
                output_inner.append(h_c_tuple[0])

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append(h_c_tuple)

        if not self.return_all_layers:
            # Take the last hidden state from the last layer
            last_frame_emb = layer_output_list[-1][:, -1, :, :, :]  # [b, hidden, h, w]

            # Apply CBAM attention
            if self.use_attention:
                last_frame_emb = self.attention(last_frame_emb)

            # Predict
            pred = self.final_conv(last_frame_emb)  # [b, out_dim*horizon, h, w]

            if self.prediction_horizon > 1:
                pred = pred.view(b, self.prediction_horizon, self.output_dim, h, w)

            # Activation
            if self.output_activation == "tanh":
                pred = torch.tanh(pred)
            elif self.output_activation == "sigmoid":
                pred = torch.sigmoid(pred)

            return pred

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            return [param] * num_layers
        return param
