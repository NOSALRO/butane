import torch


class EMA(torch.nn.Module):

    def __init__(self, model: torch.nn.Module, decay: float):
        super().__init__()
        self.model = model
        self.decay = decay
        for name, param in model.named_parameters():
            if param.requires_grad:
                buffer_name = name.replace('.', '_')
                self.register_buffer(buffer_name, param.data.clone())

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                buffer_name = name.replace('.', '_')
                buffer = getattr(self, buffer_name)
                buffer.copy_(self.decay * buffer + (1 - self.decay) * param.data)

    def apply(self):
        # Store the current parameters and apply EMA
        self.saved_params = {name.replace('.', '_'): param.data.clone() for name, param in self.model.named_parameters()}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                buffer_name = name.replace('.', '_')
                param.data.copy_(getattr(self, buffer_name))

    def undo(self):
        # Reset the model parameters to the original values
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                buffer_name = name.replace('.', '_')
                param.data.copy_(self.saved_params[buffer_name])
