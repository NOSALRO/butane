from typing import Callable, Optional, Union, Tuple
from functools import partial
import torch

from ..._typedefs import *
from ..modules.residual_blocks import *
from ..modules.conv_blocks import Conv1dBlock, Conv2dBlock, Conv3dBlock
from ..modules.mlp_block import MLPBlock
from ..modules.embeddings import SinusoidalEmbeddings, LearnableEmbeddings
from ..utils import utils


class TimeMLP(torch.nn.Module):

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        hidden_dims: IntParams,
        *,
        time_embedding_size: Optional[int] = None,
        activation_function=torch.nn.SiLU(),
        condition_input_dims: Optional[int] = None,
        condition_hidden_dims: Optional[int] = None,
        self_condition: bool = False,
        n_classes: Optional[int] = None,
    ):

        super().__init__()
        self.__input_dims = input_dims
        self.__output_dims = output_dims
        self.__condition_input_dims = (
            input_dims if condition_input_dims is None else condition_input_dims
        )
        self.__condition_hidden_dims = (
            hidden_dims if condition_hidden_dims is None else condition_hidden_dims
        )
        self.__self_condition = self_condition

        self.__time_dependent = time_embedding_size is not None

        if self.__time_dependent:
            # self.time_embeddings = LearnableEmbeddings(time_embedding_size)
            self.time_embeddings = SinusoidalEmbeddings(time_embedding_size)
            self.time_projection = MLPBlock(
                input_dims=time_embedding_size,
                output_dims=self.__input_dims,
                hidden_dims=[time_embedding_size * 4],
                activation_function=[torch.nn.SiLU()],
            )
            # self.__input_dims = self.__input_dims + input_dims

        if self_condition:
            self.condition_projection = MLPBlock(
                input_dims=self.__condition_input_dims,
                output_dims=self.__input_dims,
                hidden_dims=self.__condition_hidden_dims,
                activation_function=[activation_function],
            )
        if n_classes is not None:
            _condition_embeddings = torch.nn.Embedding(n_classes, time_embedding_size)
            self.condition_projection = torch.nn.Sequential(
                _condition_embeddings,
                MLPBlock(
                    input_dims=time_embedding_size,
                    output_dims=self.__input_dims,
                    hidden_dims=[],
                    activation_function=[torch.nn.SiLU()],
                ),
            )

        if self_condition or n_classes is not None:
            self.__input_dims = self.__input_dims + input_dims + input_dims

        self.mlp = MLPBlock(
            input_dims=self.__input_dims,
            output_dims=self.__output_dims,
            hidden_dims=hidden_dims,
            activation_function=[activation_function],
        )

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if t is not None:
            t = self.time_projection(self.time_embeddings(t))
            while t.dim() != x.dim():
                t = [..., None]
            x = torch.cat([x, t], dim=-1)

        if c is not None and hasattr(self, "condition_projection"):
            c = self.condition_projection(c)
            if self.__self_condition:
                while c.dim() != x.dim():
                    c = c[..., None]
                x = torch.cat([x, c], dim=-1)

        return self.mlp(x)
