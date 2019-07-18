import time
from options.vis_options import VisOptions
from data import create_dataset
from models import create_model
import sys
import torch
import torch.nn as nn
from torchviz import make_dot, make_dot_from_trace

if __name__ == '__main__':
    opt = VisOptions().parse()   # get options
    opt.isTrain = True
    
    dataset = create_dataset(opt)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    data = None
    for i, d in enumerate(dataset):
        data = d
        break
    model.set_input(data)
    
    model.forward()
    G = model.fake_B
    
    fake_AB = torch.cat((model.real_A, model.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
    D = model.netD(fake_AB.detach())
    
    G_graph = make_dot(G, params=dict(model.netG.named_parameters()))
    D_graph = make_dot(D, params=dict(model.netD.named_parameters()))

    G_graph.render(opt.outG)
    D_graph.render(opt.outD)