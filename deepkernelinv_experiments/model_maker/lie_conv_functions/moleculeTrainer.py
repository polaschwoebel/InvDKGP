# import torch
# from oil.model_trainers import Trainer
# from lie_conv.lieConv import LieResNet
# from lie_conv.utils import export
# from lie_conv.datasets import SO3aug, SE3aug
# from lie_conv.lieGroups import Trivial


# @export
# class MoleculeTrainer(Trainer):
#     def __init__(self, *args, task='cv', ds_stats=None, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.hypers['task'] = task
#         self.ds_stats = ds_stats
#         if hasattr(self.lr_schedulers[0], 'setup_metrics'):  # setup lr_plateau if exists
#             self.lr_schedulers[0].setup_metrics(self.logger, 'valid_MAE')

# def qm_loss(self, minibatch):
#     y = self.model(minibatch)
#     target = minibatch[self.hypers['task']]

#     if self.ds_stats is not None:
#         median, mad = self.ds_stats
#         target = (target - median) / mad

#     return (y-target).abs().mean()




# def mae(mb):
#     target = mb[task]
#     y = self.model(mb) * mad + median
#     return (y-target).abs().mean().cpu().data.numpy()


# def metrics(self, loader):
#     task = self.hypers['task']

#     #mse = lambda mb: ((self.model(mb)-mb[task])**2).mean().cpu().data.numpy()
#     if self.ds_stats is not None:
#         median, mad = self.ds_stats
#         def mae(mb):
#             target = mb[task]
#             y = self.model(mb) * mad + median
#             return (y-target).abs().mean().cpu().data.numpy()
#     else:
#         mae = lambda mb: (self.model(mb)-mb[task]).abs().mean().cpu().data.numpy()
#     return {'MAE': self.evalAverageMetrics(loader,mae)}

#     # def logStuff(self,step,minibatch=None):
#     #     super().logStuff(step,minibatch)                            
