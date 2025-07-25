import unittest
import torch
import butane

class TestLogger(unittest.TestCase):

    def test_logger(self):
        logger = butane.logger.ModelLogger(".tmp/test/", overwrite=True)
        logger.add_stats(loss=0.9)
        logger.add_stats(loss=0.8, y=100)
        logger.add_stats(loss=0.7, x=100.439409234802348)
        logger.add_stats(loss=0.7, x=100.439409234802348)
        logger.add_stats(loss=0.7, x=100.439409234802348)
        logger.add_stats(loss=0.7, x=100.439409234802348)
        logger.add_stats(loss=0.6)

        model = torch.nn.Sequential(torch.nn.Linear(10, 10))
        scaler = butane.data.StandardScaler()
        scaler.fit(torch.randn(100, 2, 3))
        # optim = torch.optim.Adam(list(model.parameters()) + list(model.parameters()))
        # print(optim.param_groups[0])
        logger.add_logs(f=100, model=model)
        print(logger.stats)
        print(logger.log['model'])
        logger.checkpoint(1, model, scaler=scaler)
        logger.load_stats(1)
        states = butane.nn.utils.load_state(".tmp/test/checkpoint_1", model=model, scaler=scaler)
        model = states.get("model")
        scaler = states.get("scaler")

if __name__ == '__main__':
    unittest.main()
