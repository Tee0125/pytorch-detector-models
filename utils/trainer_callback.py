
class TrainerCallback:
    def step_start(self, trainer, epoch):
        pass

    def step_end(self, trainer, epoch, loss):
        pass

    def minibatch_start(self, trainer, epoch, idx):
        pass

    def minibatch_end(self, trainer, epoch, idx, loss):
        pass

    def fit_start(self, trainer):
        pass

    def fit_end(self, trainer, loss):
        pass

