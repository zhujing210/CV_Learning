import visdom
import numpy as np
from mmcv.runner import Hook
from torch.utils.data import DataLoader


class Visualizer(Hook):
    def __init__(self, dataloader, interval=1, **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a pytorch DataLoader, but got'
                            f' {type(dataloader)}')
        self.vis = visdom.Visdom(env='test')
        self.flag={}
        self.log_text = ""

    def after_train_epoch(self, runner):
        self.plot_many(runner.log_buffer.output, runner.epoch)


    def plot_many(self, d,epoch):
        for k, v in d.items():
            if type(v) is float:
                self.plot(k, epoch,v)

    def plot(self, name, x, y, **kwargs):
        flag = self.flag.get(name,True)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if flag else 'append',
                      **kwargs)
        self.flag[name]=False

