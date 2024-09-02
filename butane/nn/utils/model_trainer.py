class ModelTrainer:
    def __init__(self, model, dl, optimizer,  scheduler = None):
        self.model = model
        self.dl = dl
        self.optimizer = optimizer
        self.scheduler = scheduler

    def __call__(self, epochs, loss_fn, eval_period = 0, eval_dl = None):
        for epoch in range(epochs):
            print(f"Epoch {epoch} -> ", end='')
            self.model.step(self.dl, self.optimizer, loss_fn, self.scheduler)
            if eval_period and eval_dl and not ((epoch + 1) % eval_period):
                eval(eval_dl.value(), loss)

    def eval(eval_dl, loss_fn):
        self.model.eval()
        print("Evaluation -> ", end='')
        self.model.step(eval_dl, self.optimizer, loss_fn, self.scheduler)
        self.model.train()
