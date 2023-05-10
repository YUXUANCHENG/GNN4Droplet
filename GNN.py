import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F

## adapted from https://github.com/BaratiLab/FGN
## https://github.com/BaratiLab/GAMD/tree/main/code
## andhttps://github.com/YunzhuLi/DPI-Net

## note: this only contains the model,
## actual training is not done due to time
## limitation. The results part uses demo
## model from https://github.com/YunzhuLi/DPI-Net

class EdgeEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return x


class NodeEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x

class Propagator(nn.Module):
    def __init__(self, input_size, output_size, residual=False):
        super().__init__()

        self.residual = residual

        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x, res=None):
        x = self.fc(x)
        if self.residual and res is not None:
            x += res

        return F.relu(x)


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class GNN(nn.Module):
    def __init__(self, args, stat, phases_dict, residual=False):
        super(GNN, self).__init__()

        # system properties such as number of particles
        # system dimentions etc.
        self.args = args
        state_dim = args.state_dim
        attr_dim = args.attr_dim
        relation_dim = args.relation_dim
        nf_particle = args.nf_particle
        nf_relation = args.nf_relation
        nf_effect = args.nf_effect

        self.nf_effect = args.nf_effect
        self.stat = stat
        self.residual = residual

        self.pi = nn.Parameter(torch.FloatTensor([np.pi]))
        self.dt = nn.Parameter(torch.FloatTensor([args.dt]))
        self.mean_v = nn.Parameter(torch.FloatTensor(stat[1][:, 0]))
        self.std_v = nn.Parameter(torch.FloatTensor(stat[1][:, 1]))
        self.mean_p = nn.Parameter(torch.FloatTensor(stat[0][:3, 0]))
        self.std_p = nn.Parameter(torch.FloatTensor(stat[0][:3, 1]))

        # encoders for nodes (particles)
        self.particle_encoder_list = nn.ModuleList()
        # encoders for edges (interactions)
        self.interaction_encoder_list = nn.ModuleList()
        # massage passing block for edges
        self.relation_propagator_list = nn.ModuleList()
        # massage passing block for nodes
        self.particle_propagator_list = nn.ModuleList()

        for i in range(args.n_stages):
            self.particle_encoder_list.append(
                NodeEncoder(attr_dim + state_dim * 2, nf_particle, nf_effect))
            self.interaction_encoder_list.append(
                EdgeEncoder(2 * attr_dim + 4 * state_dim + relation_dim, nf_relation, nf_relation))
            self.relation_propagator_list.append(Propagator(nf_relation + 2 * nf_effect, nf_effect))
            self.particle_propagator_list.append(Propagator(2 * nf_effect, nf_effect, self.residual))

        # decoder
        self.predictor = Decoder(nf_effect, nf_effect, args.position_dim)


    def forward(self, attr, state, Rr, Rs, Ra, n_particles, node_r_idx, node_s_idx, pstep,
                instance_idx, phases_dict, verbose=0):
        # construct particle attribute encoding
        particle_effect = torch.zeros((attr.size(0), self.nf_effect)).to(attr.device)

        num_stages = len(Rr)
        for stage in range(num_stages):
            Rrp = Rr[stage].t()
            Rsp = Rs[stage].t()
            
            # receiver and sender
            attr_r = attr[node_r_idx[stage]]
            attr_s = attr[node_s_idx[stage]]
            attr_r_rel = Rrp.mm(attr_r)
            attr_s_rel = Rsp.mm(attr_s)

            # state of receiver and sender
            state_r = state[node_r_idx[stage]]
            state_s = state[node_s_idx[stage]]
            state_r_rel = Rrp.mm(state_r)
            state_s_rel = Rsp.mm(state_s)

            particle_encode = self.particle_encoder_list[stage](torch.cat([attr_r, state_r], 1))
            edge_encode = self.interaction_encoder_list[stage](torch.cat([attr_r_rel, attr_s_rel, 
                                                            state_r_rel, state_s_rel, Ra[stage]], 1))

            for step in range(pstep[stage]):
                effect_p_r = particle_effect[node_r_idx[stage]]
                effect_p_s = particle_effect[node_s_idx[stage]]

                receiver_effect = Rrp.mm(effect_p_r)
                sender_effect = Rsp.mm(effect_p_s)

                effect_rel = self.relation_propagator_list[stage](
                    torch.cat([edge_encode, receiver_effect, sender_effect], 1))

                effect_p_r_agg = Rr[stage].mm(effect_rel)

                effect_p = self.particle_propagator_list[stage](
                    torch.cat([particle_encode, effect_p_r_agg], 1),
                    res=effect_p_r)

                particle_effect[node_r_idx[stage]] = effect_p

        pred = []
        for i in range(len(instance_idx) - 1):
            st, ed = instance_idx[i], instance_idx[i + 1]
            pred.append(self.predictor(particle_effect[st:ed]))

        pred = torch.cat(pred, 0)

        return pred

