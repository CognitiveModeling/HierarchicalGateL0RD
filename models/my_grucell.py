import torch
import torch.nn as nn

class MyGRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, output_size=-1, device=None):
        """
        GRU cell that adds a read-out layer if the hidden size != output dim to align it with GateL0RD where the
        dimensionality of the latent state is independent of the output dimension.
        """

        super(MyGRUCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        if output_size == -1:
            self.output_size = hidden_size
        else:
            self.output_size = output_size

        self.cell = torch.nn.GRUCell(input_size=input_size, hidden_size=hidden_size)

        if output_size == hidden_size:
            self.read_out = torch.nn.Identity()
        else:
            self.read_out = torch.nn.Linear(hidden_size, output_size)

        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = device

    def forward(self, x_t, h_tminus1=None):

        h_t = self.cell(x_t, h_tminus1)
        y_t = self.read_out(h_t)
        return y_t, h_t, torch.ones((h_t.size()), device=self.device)

    def loss(self, loss_task, Theta=None):
        """
        Added loss wrapper to align with GateL0RDCell
        """
        return loss_task

