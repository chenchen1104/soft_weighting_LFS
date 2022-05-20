from lfs_core.utils.loss import loss_search, loss_baseline
from lfs_core.utils.loss_agent import LFSAgent
import lfs_core.link_utils as link
import torch


def softmax(x):
    exp_x = torch.exp(x)
    sum_exp_x = torch.sum(exp_x)
    y = exp_x / sum_exp_x
    return y


class LossFuncSearch(object):
    def __init__(self, do_search=True):
        self.model = None
        self.lr = 0.05
        self.sample_step = 2
        self.val_freq = 2
        self.scale = 0.2
        self.best_acc = 0
        self.best_epoch = -1
        self.__init_agent()
        self.do_search = do_search

    def __init_agent(self):
        self.agent = LFSAgent(self.lr, self.scale)
        a = torch.randn(3)
        self.a = softmax(a)

    def set_model(self, model):
        self.model = model

    def get_loss(self, outputs, targets, inputs):
        if self.do_search:
            loss = loss_search(outputs, targets, inputs, self.a)
        else:
            loss = loss_baseline(outputs, targets)
        return loss

    def set_loss_parameters(self, epoch):
        if epoch >= 2:
            self.a, self.mu = self.agent.sample_subfunction()

    def update_lfs(self, reward):
        self.agent.step(reward=reward)
