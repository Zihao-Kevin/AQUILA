import torch
from config import cfg
# import numpy as np
# import minpy.numpy as np
from collections import OrderedDict
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import copy
import math

device = cfg['device']

def cal_ksi():
    D = 10
    ck = 0.8
    ksi = torch.ones((D, D + 1))  # ksi=np.matrix(ksi);
    ksi = ksi.to(device)
    for i in range(0, D + 1):
        if (i == 0):
            ksi[:, i] = torch.ones(D)
        if (i <= D and i > 0):
            ksi[:, i] = 1 / i * torch.ones(D)
    ksi = ck * ksi
    return ksi


def cal_weight_bits_CNN(model):
    # le = len(model.state_dict())
    nv = 0
    for k, v in model.state_dict().items():
        if 'weight' in k or 'bias' in k:
            da = 1
            for i in range(0, len(v.shape)):
                da = da * int(v.shape[i])
            nv = nv + da
    return nv


def gradtovec(grads):
    vec = torch.tensor([]).to(device)
    if type(grads) == OrderedDict:
        for k, v in grads.items():
            if "weight" not in k and 'bias' not in k:
                continue
            para_name = k.split('.')[-1]
            da = 1
            if para_name == "weight":
                for i in range(len(v.shape)):
                    da *= int(v.shape[i])
                v = v.reshape(da)
            # elif para_name == "bias":
            vec = torch.cat([vec, v], dim=0)
    else:
        for i in range(0, len(grads)):
            v = grads[i]
            da = 1
            if len(v.shape) > 1:
                for i in range(len(v.shape)):
                    da *= int(v.shape[i])
                v = v.reshape(da)
            vec = torch.cat([vec, v], dim=0)
    return vec


def gradtovec_new(grads):
    vec = torch.tensor([]).to(device)
    for param in grads:
        if len(param.shape) == 1:
            vec = torch.cat([vec, param], dim=0)
        else:
            da = 1
            for i in range(len(param.shape)):
                da *= int(param.shape[i])
            param = param.reshape(da)
            vec = torch.cat([vec, param], dim=0)

    return vec


def vectograd(vec, model):
    tmp = []
    for k, v in model.state_dict().items():
        if 'weight' in k or 'bias' in k:
            if len(v.shape) == 4:
                da = int(v.shape[0])
                db = int(v.shape[1])
                dc = int(v.shape[2])
                dd = int(v.shape[3])
                c = vec[0:da * db * dc * dd]
                c = c.reshape((da, db, dc, dd))
                vec = vec[da * db * dc * dd:]
            elif len(v.shape) == 2:
                da = int(v.shape[0])
                db = int(v.shape[1])
                c = vec[0:da * db]
                c = c.reshape(da, db)
                vec = vec[da * db:]
            elif len(v.shape) == 1:
                da = int(v.shape[0])
                c = vec[0:da]
                vec = vec[da:]
            c = c.to(device)
            tmp.append(c)
    return tmp


def quantd(vec, v2, b):
    r = torch.max(abs(vec - v2)).to(device)
    delta = (r / (2 ** b - 1)).to(device)
    quantv = (v2 - r + 2 * delta * torch.floor((vec - v2 + r + delta) / (2 * delta))).to(device)
    return quantv

def quantd_aquila(vec, v2):
    R_m_k = torch.max(abs(vec - v2)).item()
    R_M_K = torch.ones_like(vec).to(device) * R_m_k
    d = vec.shape[0]
    s_k = max(math.floor(math.log2(R_m_k * math.sqrt(d) /torch.norm((vec - v2), 2) + 1)), 1)
    if s_k >= 16:
        s_k = 16
    print("Adaptive Quantization Bit is {}".format(s_k))
    tau = 1 / (2 ** s_k - 1)
    # tau = torch.sum(v2 - vec + R_M_K) / (2 * R_m_k.item() * vec.shape[0])
    delta = (R_M_K * tau).to(device)
    quantv = (v2 - R_M_K + 2 * delta * torch.floor((vec - v2 + R_M_K + delta) / (2 * tau * R_m_k))).to(device)
    return quantv, s_k


def quant_qsgd(x, b):
    x_norm = torch.norm(x, 2).to(device)
    sign_x = torch.sign(x).int().to(device)
    p = torch.div(torch.abs(x), x_norm).to(device)
    renormalize_p = torch.mul(p, b).to(device)
    floor_p = torch.floor(renormalize_p).to(device)
    compare = torch.rand_like(floor_p).to(device)
    final_p = (renormalize_p - floor_p).to(device)
    margin = (compare < final_p).float().to(device)
    xi = ((floor_p + margin) / b).to(device)
    res = (x_norm * sign_x * xi).to(device)

    return res


def L2_Loss(model, decay):
    loss = 0
    for k, v in model.state_dict().items():
        loss = torch.sum(torch.pow(v, 2))
    return decay / 2 * loss


def in_stack(x, y):
    for i in range(1, 10):
        x[i - 1, :] = x[i, :]
    x[9, :] = y


def lazily_aggre_aquila(gamma, beta, global_model_diff, s_k, vec, k, m, ksi, dtheta, local, dsa, weight_bit, Bit, Ind, me):
    user_num = int(np.ceil(cfg['frac'] * cfg['num_users']))
    # gamma = 5
    # beta = 0.5
    C = cfg['C_aquila']
    Q_diff = torch.norm(local[m].mgr - local[m].gr) ** 2
    error = local[m].err
    if local[m].clock == C or Q_diff + gamma * error > beta / (cfg['lr'] ** 2*0.5) * global_model_diff or k == 0:
        Ind[m, k] = 1

    if (Ind[m, k] == 1):
        local[m].mgr = (local[m].gr).to(device)
        local[m].clock = 0
        Bit[k] = Bit[k] + s_k * local[m].local_weight_bit
        local[m].ehat = (local[m].err).to(device)
        local[m].ehat_vec = (local[m].err_vec).to(device)
        local[m].grad_pre = vec.to(device)

    if (Ind[m, k] == 0):
        local[m].clock += 1
        print("Fail")
    return dsa.to(device)

def lazily_aggre_aquila_uuu(gamma, beta, global_model_diff, s_k, vec, k, m, ksi, dtheta, local, dsa, weight_bit, Bit, Ind, me):
    # gamma = 5
    # beta = 0.5
    C = cfg['C_aquila']
    Q_diff = torch.norm(local[m].mgr) ** 2
    error = local[m].ehat
    if local[m].clock == C or Q_diff + error > beta / (cfg['lr'] ** 2) * global_model_diff or k == 0:
        Ind[m, k] = 1

    if (Ind[m, k] == 1):
        local[m].mgr = (local[m].gr).to(device)
        local[m].ehat = (local[m].err).to(device)
        local[m].clock = 0
        Bit[k] = Bit[k] + s_k * local[m].local_weight_bit
        local[m].ehat_vec = (local[m].err_vec).to(device)
        local[m].grad_pre = vec.to(device)

    if (Ind[m, k] == 0):
        local[m].clock += 1
        print("Fail")
    judge_res = Ind[m, k]
    return dsa.to(device), judge_res


def lazily_aggre(s_k, vec, k, m, ksi, dtheta, local, dsa, weight_bit, Bit, Ind, me, exp_name):
    D = 10
    user_num = int(np.ceil(cfg['frac'] * cfg['num_users']))
    dL = torch.zeros((user_num, weight_bit)).to(device)
    for d in range(0, D):
        if (k - d >= 0):
            if (k <= D - 1):
                me[m] = me[m] + ksi[d, k] * (dtheta[k - d, :].dot(dtheta[k - d, :]))
            if (k > D):
                me[m] = me[m] + ksi[d, D] * (dtheta[9 - d, :].dot(dtheta[9 - d, :]))
    dL[m, :local[m].local_weight_bit] = local[m].gr - local[m].mgr

    alpha = cfg['alpha']
    C = cfg['C']
    tmp_t = torch.norm(local[m].err_vec) ** 2 + torch.norm(local[m].ehat_vec) ** 2
    if local[m].clock == C or ((dL[m, :].dot(dL[m, :])) > (1 / (alpha ** 2 * user_num ** 2)) * me[m] + tmp_t):
        Ind[m, k] = 1

    if (Ind[m, k] == 1):
        dsa = dsa + dL[m, :]
        local[m].mgr = (local[m].gr).to(device)
        local[m].clock = 0
        Bit[k] = Bit[k] + s_k * local[m].local_weight_bit
        local[m].ehat = (local[m].err).to(device)
        local[m].ehat_vec = (local[m].err_vec).to(device)
        local[m].grad_pre = vec.to(device)

    if (Ind[m, k] == 0):
        local[m].clock += 1
        print("Fail")
    return dsa.to(device), Ind[m, k]


def No_Judge(s_k, k, m, local, dsa, weight_bit, Bit, Ind):
    user_num = int(np.ceil(cfg['frac'] * cfg['num_users']))
    dL = torch.zeros((user_num, weight_bit)).to(device)
    dL[m, :local[m].local_weight_bit] = local[m].gr - local[m].mgr
    Ind[m, k] = 1

    if (Ind[m, k] == 1):
        dsa = dsa + dL[m, :]
        local[m].mgr = local[m].gr
        Bit[k] = Bit[k] + s_k * local[m].local_weight_bit
        local[m].ehat = local[m].err

    local[m].mmgr = copy.deepcopy((local[m].mgr).to(device))
    local[m].mgr = copy.deepcopy((local[m].gr).to(device))
    return dsa, Ind[m, k]
