import unittest
import torch
import butane

class TestLogger(unittest.TestCase):

    def test_logger(self):
        logger = butane.logger.ModelLogger(".tmp/test/", overwrite=True, use_tb=True)
        logger.add_stats(loss=0.9)
        logger.add_stats(loss=0.8, y=100)
        logger.add_stats(loss=0.7, x=100.439409234802348)
        logger.add_stats(loss=0.7, x=100.439409234802348)
        logger.add_stats(loss=0.7, x=100.439409234802348)
        logger.add_stats(loss=0.7, x=100.439409234802348)
        logger.add_stats(loss=0.6)

        model = torch.nn.Sequential(torch.nn.Linear(10, 10))
        print(torch.nn.utils.parameters_to_vector(model.parameters()).norm())
        scaler = butane.data.StandardScaler()
        scaler.fit(torch.randn(100, 2, 3))
        optim = torch.optim.Adam(model.parameters(), betas=(0.8, 0.7))
        lrs=torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=100)
        logger.add_logs(f=100, model=model)
        logger.checkpoint(1, model=model, optimizer=optim, lr_scheduler=lrs, scaler=scaler)
        # logger.load_stats(1)
        states = butane.nn.utils.load_state(".tmp/test/checkpoint_1", model=model, lr_scheduler=lrs, optimizer=optim, scaler=scaler)
        # states = logger.load_checkpoint(1, model=model, lr_scheduler=lrs, optimizer=optim, scaler=scaler)
        print(torch.nn.utils.parameters_to_vector(model.parameters()).norm())
        # print(states)
        # model = states.get("model")
        # scaler = states.get("scaler")

    # def test_monitoring(self):
    #     mm = butane.logger.ModelMonitor(
    #         increase_keys=["x1", "x2"],
    #         # decrease_keys=["y1", "y2"],
    #         tolerance=0.1
    #     )
    #     s1 = dict(
    #         x1=1,
    #         x2=2,
    #         y1=3,
    #         y2=4
    #     )
    #     mm(1, s1)

    #     s2 = dict(
    #         x1=1.5,
    #         x2=1.8,
    #         y1=2.9,
    #         y2=3.5,
    #     )
    #     mm(2, s2)


if __name__ == '__main__':
    unittest.main()
