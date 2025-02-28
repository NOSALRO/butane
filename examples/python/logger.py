import torch
import butane

if __name__ == "__main__":

    logger = butane.logger.ModelLogger('.tmp/test_logger', overwrite = True)

    model = butane.nn.MLPBlock(
        input_dims = 3,
        output_dims=2,
        hidden_dims=[64, 64],
        activation_function=[torch.nn.ReLU(), torch.nn.ReLU(), torch.nn.Tanh()],
        output_activation=False).to('cuda')
    optim = torch.optim.Adam(model.parameters())

    logger.checkpoint(100, model, optim)

    model, _ = butane.nn.utils.load_state('.tmp/test_logger/checkpoint_100', model, optim)
    print(logger.last_path)

