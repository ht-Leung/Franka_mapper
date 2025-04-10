import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




# output is T
def FK(thetas):
    # iiwa
    alpha_1 = torch.tensor(-np.pi * 0.5)
    alpha_2 = torch.tensor(np.pi * 0.5)
    alpha_3 = torch.tensor(np.pi * 0.5)
    alpha_4 = torch.tensor(-np.pi * 0.5)
    alpha_5 = torch.tensor(-np.pi * 0.5)
    alpha_6 = torch.tensor(np.pi * 0.5)
    alpha_7 = torch.tensor(0.0)

    a_1 = 0.0
    a_2 = 0.0
    a_3 = 0.0
    a_4 = 0.0
    a_5 = 0.0
    a_6 = 0.0
    a_7 = 0.0

    d_1 = 340.0
    d_2 = 0.0
    d_3 = 400.0
    d_4 = 0.0
    d_5 = 400.0
    d_6 = 0.0
    d_7 = 126.0 + 120.

    t1 = thetas[0]
    t2 = thetas[1]
    t3 = thetas[2]
    t4 = thetas[3]
    t5 = thetas[4]
    t6 = thetas[5]
    t7 = thetas[6]

    T0_1 = torch.zeros((4, 4))
    T0_1[0, 0] = torch.cos(t1)
    T0_1[0, 1] = -1 * torch.sin(t1) * torch.cos(alpha_1)
    T0_1[0, 2] = torch.sin(t1) * torch.sin(alpha_1)
    T0_1[0, 3] = a_1 * torch.cos(t1)
    T0_1[1, 0] = torch.sin(t1)
    T0_1[1, 1] = torch.cos(t1) * torch.cos(alpha_1)
    T0_1[1, 2] = -1 * torch.cos(t1) * torch.sin(alpha_1)
    T0_1[1, 3] = a_1 * torch.sin(t1)
    T0_1[2, 0] = 0
    T0_1[2, 1] = torch.sin(alpha_1)
    T0_1[2, 2] = torch.cos(alpha_1)
    T0_1[2, 3] = d_1
    T0_1[3, 0] = 0
    T0_1[3, 1] = 0
    T0_1[3, 2] = 0
    T0_1[3, 3] = 1.0

    T1_2 = torch.zeros((4, 4))
    T1_2[0, 0] = torch.cos(t2)
    T1_2[0, 1] = -1 * torch.sin(t2) * torch.cos(alpha_2)
    T1_2[0, 2] = torch.sin(t2) * torch.sin(alpha_2)
    T1_2[0, 3] = a_2 * torch.cos(t2)
    T1_2[1, 0] = torch.sin(t2)
    T1_2[1, 1] = torch.cos(t2) * torch.cos(alpha_2)
    T1_2[1, 2] = -1 * torch.cos(t2) * torch.sin(alpha_2)
    T1_2[1, 3] = a_2 * torch.sin(t2)
    T1_2[2, 0] = 0
    T1_2[2, 1] = torch.sin(alpha_2)
    T1_2[2, 2] = torch.cos(alpha_2)
    T1_2[2, 3] = d_2
    T1_2[3, 0] = 0
    T1_2[3, 1] = 0
    T1_2[3, 2] = 0
    T1_2[3, 3] = 1.0

    T2_3 = torch.zeros((4, 4))
    T2_3[0, 0] = torch.cos(t3)
    T2_3[0, 1] = -torch.sin(t3) * torch.cos(alpha_3)
    T2_3[0, 2] = torch.sin(t3) * torch.sin(alpha_3)
    T2_3[0, 3] = a_3 * torch.cos(t3)
    T2_3[1, 0] = torch.sin(t3)
    T2_3[1, 1] = torch.cos(t3) * torch.cos(alpha_3)
    T2_3[1, 2] = -torch.cos(t3) * torch.sin(alpha_3)
    T2_3[1, 3] = a_3 * torch.sin(t3)
    T2_3[2, 0] = 0
    T2_3[2, 1] = torch.sin(alpha_3)
    T2_3[2, 2] = torch.cos(alpha_3)
    T2_3[2, 3] = d_3
    T2_3[3, 0] = 0
    T2_3[3, 1] = 0
    T2_3[3, 2] = 0
    T2_3[3, 3] = 1.0

    T3_4 = torch.zeros((4, 4))
    T3_4[0, 0] = torch.cos(t4)
    T3_4[0, 1] = -torch.sin(t4) * torch.cos(alpha_4)
    T3_4[0, 2] = torch.sin(t4) * torch.sin(alpha_4)
    T3_4[0, 3] = a_4 * torch.cos(t4)
    T3_4[1, 0] = torch.sin(t4)
    T3_4[1, 1] = torch.cos(t4) * torch.cos(alpha_4)
    T3_4[1, 2] = -torch.cos(t4) * torch.sin(alpha_4)
    T3_4[1, 3] = a_4 * torch.sin(t4)
    T3_4[2, 0] = 0
    T3_4[2, 1] = torch.sin(alpha_4)
    T3_4[2, 2] = torch.cos(alpha_4)
    T3_4[2, 3] = d_4
    T3_4[3, 0] = 0
    T3_4[3, 1] = 0
    T3_4[3, 2] = 0
    T3_4[3, 3] = 1.0

    T4_5 = torch.zeros((4, 4))
    T4_5[0, 0] = torch.cos(t5)
    T4_5[0, 1] = -torch.sin(t5) * torch.cos(alpha_5)
    T4_5[0, 2] = torch.sin(t5) * torch.sin(alpha_5)
    T4_5[0, 3] = a_5 * torch.cos(t5)
    T4_5[1, 0] = torch.sin(t5)
    T4_5[1, 1] = torch.cos(t5) * torch.cos(alpha_5)
    T4_5[1, 2] = -torch.cos(t5) * torch.sin(alpha_5)
    T4_5[1, 3] = a_5 * torch.sin(t5)
    T4_5[2, 0] = 0
    T4_5[2, 1] = torch.sin(alpha_5)
    T4_5[2, 2] = torch.cos(alpha_5)
    T4_5[2, 3] = d_5
    T4_5[3, 0] = 0
    T4_5[3, 1] = 0
    T4_5[3, 2] = 0
    T4_5[3, 3] = 1.0

    T5_6 = torch.zeros((4, 4))
    T5_6[0, 0] = torch.cos(t6)
    T5_6[0, 1] = -torch.sin(t6) * torch.cos(alpha_6)
    T5_6[0, 2] = torch.sin(t6) * torch.sin(alpha_6)
    T5_6[0, 3] = a_6 * torch.cos(t6)
    T5_6[1, 0] = torch.sin(t6)
    T5_6[1, 1] = torch.cos(t6) * torch.cos(alpha_6)
    T5_6[1, 2] = -torch.cos(t6) * torch.sin(alpha_6)
    T5_6[1, 3] = a_6 * torch.sin(t6)
    T5_6[2, 0] = 0
    T5_6[2, 1] = torch.sin(alpha_6)
    T5_6[2, 2] = torch.cos(alpha_6)
    T5_6[2, 3] = d_6
    T5_6[3, 0] = 0
    T5_6[3, 1] = 0
    T5_6[3, 2] = 0
    T5_6[3, 3] = 1.0

    T6_7 = torch.zeros((4, 4))
    T6_7[0, 0] = torch.cos(t7)
    T6_7[0, 1] = -torch.sin(t7) * torch.cos(alpha_7)
    T6_7[0, 2] = torch.sin(t7) * torch.sin(alpha_7)
    T6_7[0, 3] = a_7 * torch.cos(t7)
    T6_7[1, 0] = torch.sin(t7)
    T6_7[1, 1] = torch.cos(t7) * torch.cos(alpha_7)
    T6_7[1, 2] = -torch.cos(t7) * torch.sin(alpha_7)
    T6_7[1, 3] = a_7 * torch.sin(t7)
    T6_7[2, 0] = 0
    T6_7[2, 1] = torch.sin(alpha_7)
    T6_7[2, 2] = torch.cos(alpha_7)
    T6_7[2, 3] = d_7
    T6_7[3, 0] = 0
    T6_7[3, 1] = 0
    T6_7[3, 2] = 0
    T6_7[3, 3] = 1.0

    T0_2 = torch.mm(T0_1, T1_2)
    T0_3 = torch.mm(T0_2, T2_3)
    T0_4 = torch.mm(T0_3, T3_4)
    T0_5 = torch.mm(T0_4, T4_5)
    T0_6 = torch.mm(T0_5, T5_6)
    T0_7 = torch.mm(T0_6, T6_7)

    return torch.cat([T0_7[0:3, 0], T0_7[0:3, 1], T0_7[0:3, 2], T0_7[0:3, 3]/100])   # 相当于T的转置


class Mapper(nn.Module):
    def __init__(self):
        super(Mapper, self).__init__()

        self.input = torch.nn.Linear(2, 60)
        self.hid = torch.nn.Linear(60, 100)
        self.out = torch.nn.Linear(100, 7)

    def forward(self, x):
        input_out = torch.tanh(self.input(x))
        hid = torch.tanh(self.hid(input_out))
        out = torch.tanh(self.out(hid))
        return out





class UR5e_Mapper(nn.Module):
    def __init__(self):
        super(UR5e_Mapper, self).__init__()

        self.input = torch.nn.Linear(2, 60)
        self.hid = torch.nn.Linear(60, 100)
        self.out = torch.nn.Linear(100, 6)

    def forward(self, x):
        input_out = torch.tanh(self.input(x))
        hid = torch.tanh(self.hid(input_out))
        out = torch.tanh(self.out(hid))
        return out

class panda_Mapper(nn.Module):
    def __init__(self):
        super(panda_Mapper, self).__init__()

        self.input = torch.nn.Linear(2, 60)
        self.hid = torch.nn.Linear(60, 100)
        self.out = torch.nn.Linear(100, 7)

    def forward(self, x):
        input_out = torch.tanh(self.input(x))
        hid = torch.tanh(self.hid(input_out))
        out = torch.tanh(self.out(hid))
        return out
