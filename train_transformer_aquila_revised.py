import argparse
import copy
import datetime
import models
import numpy as np
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
from config import cfg
from data import fetch_dataset, make_data_loader, split_dataset, SplitDataset, BatchDataset
from fed import Federation
from metrics import Metric
from utils import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume, collate
from logger import Logger
import math
from numpy.random import RandomState
from AQG_utils import *
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
warnings.filterwarnings("ignore", category=UserWarning)
args = vars(parser.parse_args())
for k in cfg:
    cfg[k] = args[k]
if args['control_name']:
    cfg['control'] = {k: v for k, v in zip(cfg['control'].keys(), args['control_name'].split('_'))} \
        if args['control_name'] != 'None' else {}
cfg['control_name'] = '_'.join([cfg['control'][k] for k in cfg['control']])
cfg['pivot_metric'] = 'Global-Perplexity'
cfg['pivot'] = float('inf')
cfg['metric_name'] = {'train': {'Local': ['Local-Loss', 'Local-Perplexity', 'Total-Bit']},
                      'test': {'Global': ['Global-Loss', 'Global-Perplexity']}}
current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
device = cfg['device']

torch.cuda.set_device(1)
grad_update_save = []
grad_save = []

def main():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['data_name'], cfg['subset'], cfg['model_name'], cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    seed = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    dataset = fetch_dataset(cfg['data_name'], cfg['subset'])
    process_dataset(dataset)
    model = eval('models.{}(model_rate=cfg["global_model_rate"]).to(cfg["device"])'.format(cfg['model_name']))
    optimizer = make_optimizer(model, cfg['lr'])
    scheduler = make_scheduler(optimizer)
    if cfg['resume_mode'] == 1:
        last_epoch, data_split, label_split, model, optimizer, scheduler, logger = resume(model, cfg['model_tag'],
                                                                                          optimizer, scheduler)
    elif cfg['resume_mode'] == 2:
        last_epoch = 1
        _, data_split, label_split, model, _, _, _ = resume(model, cfg['model_tag'])
        logger_path = os.path.join('output', 'runs', '{}'.format(cfg['model_tag']))
        logger = Logger(logger_path)
    else:
        last_epoch = 1
        data_split, label_split = split_dataset(dataset, cfg['num_users'], cfg['data_split_mode'])

        logger_path = 'output/runs/{}_{}_{}_{}_{}_{}_{}'.format(cfg['exp_name'], cfg['data_split_mode'], cfg['num_users'],
                                                       cfg['model_mode'], cfg['data_split_mode'], cfg['beta'], current_time)
        logger = Logger(logger_path)
    if data_split is None:
        data_split, label_split = split_dataset(dataset, cfg['num_users'], cfg['data_split_mode'])
    global_parameters = model.state_dict()
    federation = Federation(global_parameters, cfg['model_rate'], label_split)

    ksi = cal_ksi()
    weight_bits = cal_weight_bits_CNN(model)

    act_user_num = int(np.ceil(cfg['frac'] * cfg['num_users']))
    dtheta = torch.zeros((10, weight_bits)).to(device)
    Ind = torch.zeros((act_user_num, cfg['num_epochs']['global']))
    Bit = torch.zeros(cfg['num_epochs']['global'])
    dsa = torch.zeros(weight_bits).to(device)

    # prob = 1 / (cfg['num_users'])
    prob = 1 / (32 * (math.pow(2, 6) - 1))
    rs = RandomState(2022)

    # local = [None for _ in range(act_user_num)]
    local = [Local(1, 0, None) for _ in range(act_user_num)]
    global_model_diff = 0
    for epoch in range(last_epoch - 1, cfg['num_epochs']['global']):
        logger.safe(True)

        # me = torch.zeros(act_user_num).to(device)
        # var = gradtovec(model.state_dict()).to(device)
        # if (epoch >= 1):
        #     if epoch >= 10:
        #         in_stack(dtheta, var - theta)
        #     else:
        #         dtheta[epoch, :] = var - theta
        # theta = copy.deepcopy(var).to(device)

        dsa, local, global_model_diff = train(global_model_diff, local, dataset['train'], data_split['train'],
                                              label_split, federation, model, optimizer,
                                        logger, epoch, ksi, dsa, dtheta, weight_bits, Bit, Ind, rs, prob)
        test(dataset['test'], model, logger, epoch)
        # if cfg['scheduler_name'] == 'ReduceLROnPlateau':
        #     scheduler.step(metrics=logger.mean['train/{}'.format(cfg['pivot_metric'])])
        # else:
        #     scheduler.step()
        bit_info = {'Total-Bit': float(Bit[epoch])}
        logger.append(bit_info, 'train', n=5)
        logger.write('train', cfg['metric_name']['train']['Local'][2])
    logger.safe(False)
    return


def train(global_model_diff, local, dataset, data_split, label_split, federation, global_model, optimizer, logger, epoch, ksi, dsa,
              dtheta, weight_bits, Bit, Ind, rs, prob, me=None):
    global_model.load_state_dict(federation.global_parameters)
    global_params = copy.deepcopy(global_model.state_dict())
    global_model.train(True)
    local, local_parameters, user_idx, param_idx = make_local(local, epoch, dataset, data_split, label_split, federation)
    num_active_users = len(local)
    lr = optimizer.param_groups[0]['lr']
    start_time = time.time()
    local_grads = []
    c_k = rs.binomial(1, prob, 1)[0]
    save_bits = 0
    update_idx = []
    for m in range(num_active_users):
        judge_res = 0
        if args['exp_name'] in 'marina':
            tmp, s_k, vec, local_parameters_copy = local[m].train(epoch, local_parameters[m], lr, logger, c_k)
        else:
            tmp, s_k, vec, local_parameters_copy = local[m].train(epoch, local_parameters[m], lr, logger)
        local_parameters[m] = copy.deepcopy(local_parameters_copy)
        local_grads.append(tmp)
        if args['exp_name'] == 'aquila':
            beta = cfg['beta']
            dsa, judge_res = lazily_aggre_aquila_uuu(beta, global_model_diff, s_k, vec, epoch, m, ksi, dtheta, local, dsa, weight_bits, Bit, Ind, me)
        elif args['exp_name'] in ['adap_plus_laq', 'aquila', 'laq']:
            dsa, judge_res = lazily_aggre(s_k, vec, epoch, m, ksi, dtheta, local, dsa, weight_bits, Bit, Ind, me, args['exp_name'])
        elif args['exp_name'] in ["adap", 'fedavg', 'qsgd', 'marina', 'lena']:
            dsa, judge_res = No_Judge(s_k, epoch, m, local, dsa, weight_bits, Bit, Ind)
        # local_parameters[m] = copy.deepcopy(local[m].train(local_parameters[m], lr, logger))
        if judge_res == 1:
            update_idx.append(m)
        if m == 0:
            local_time = (time.time() - start_time) / (m + 1)
            epoch_finished_time = datetime.timedelta(seconds=local_time * (num_active_users - m - 1))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg['num_epochs']['global'] - epoch) * local_time * num_active_users))
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * m / num_active_users),
                             'ID: {}({}/{})'.format(user_idx[m], m + 1, num_active_users),
                             'Learning rate: {}'.format(lr),
                             'Rate: {}'.format(federation.model_rate[user_idx[m]]),
                             'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            logger.write('train', cfg['metric_name']['train']['Local'][:2])
    update_params = []
    update_param_idx = []
    update_user_idx = []
    # if no client is chosen, continue
    if update_idx == []:
        for k, v in federation.global_parameters.items():
            global_model_diff += torch.norm(federation.global_parameters[k] - global_params[k]) ** 2
        return dsa, local, global_model_diff
    else:
        for i in update_idx:
            update_params.append(copy.deepcopy(local_parameters[i]))
            update_param_idx.append(copy.deepcopy(param_idx[i]))
            update_user_idx.append(copy.deepcopy(user_idx[i]))
        federation.combine(update_params, update_param_idx, update_user_idx)
        global_model_diff = 0
        for k, v in federation.global_parameters.items():
            global_model_diff += torch.norm(federation.global_parameters[k] - global_params[k]) ** 2
        global_model.load_state_dict(federation.global_parameters)

        # global_model_diff: the deflection between theta^{k + 1} and theta^k
        return dsa, local, global_model_diff


def test(dataset, model, logger, epoch):
    with torch.no_grad():
        metric = Metric()
        model.train(False)
        batch_dataset = BatchDataset(dataset, cfg['bptt'])
        for i, input in enumerate(batch_dataset):
            input_size = input['label'].size(0)
            input = to_device(input, cfg['device'])
            output = model(input)
            output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
            evaluation = metric.evaluate(cfg['metric_name']['test']['Global'], input, output)
            logger.append(evaluation, 'test', input_size)
        info = {'info': ['Model: {}'.format(cfg['model_tag']),
                         'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        logger.write('test', cfg['metric_name']['test']['Global'])
    return


def make_local(local, epoch, dataset, data_split, label_split, federation):
    num_active_users = int(np.ceil(cfg['frac'] * cfg['num_users']))
    user_idx = torch.arange(cfg['num_users']).tolist()
    local_parameters, param_idx = federation.distribute(user_idx)
    if epoch == 0:
        for m in range(num_active_users):
            model_rate_m = federation.model_rate[user_idx[m]]
            dataset_m = SplitDataset(dataset, data_split[user_idx[m]])
            local[m] = Local(model_rate_m, label_split[user_idx[m]], dataset_m)
    else:
        local_new = [None for _ in range(num_active_users)]
        for m in range(num_active_users):
            model_rate_m = federation.model_rate[user_idx[m]]
            dataset_m = SplitDataset(dataset, data_split[user_idx[m]])
            local_new[m] = Local(model_rate_m, label_split[user_idx[m]], dataset_m)
            local[m].update_AQG_data(local_new[m])
    return local, local_parameters, user_idx, param_idx


class Local:
    def __init__(self, model_rate, label_split, dataset=None):
        if dataset != None:
            self.dataset = BatchDataset(dataset, cfg['bptt'])
        self.model_rate = model_rate
        # self.data_loader = data_loader
        self.label_split = label_split
        self.local_weight_bit = 0
        self.mgr = None
        self.mmgr = None
        self.gr = None
        self.ehat = 0
        self.err = 0
        self.cal_local_weight_bits()
        self.clock = 0
        self.grad_update_0 = None
        self.grad_pre = None
        self.pre_v_m_t = None  # LENA param
        self.ehat_vec = None
        self.err_vec = None

    def update_AQG_data(self, new):
        self.model_rate = new.model_rate
        self.dataset = new.dataset
        self.label_split = new.label_split

    def cal_local_weight_bits(self):
        model = eval('models.{}(model_rate=self.model_rate).to(cfg["device"])'.format(cfg['model_name']))
        self.local_weight_bit = cal_weight_bits_CNN(model)
        self.mgr = torch.zeros(self.local_weight_bit).to(device)
        self.gr = torch.zeros(self.local_weight_bit).to(device)

    def train(self, global_epoch, local_parameters, lr, logger, c_k=None):
        metric = Metric()
        model = eval('models.{}(model_rate=self.model_rate).to(cfg["device"])'.format(cfg['model_name']))
        model.load_state_dict(local_parameters)
        model.train(True)
        optimizer = make_optimizer(model, lr)
        for local_epoch in range(1, cfg['num_epochs']['local'] + 1):
            for (i, input) in enumerate(self.dataset):
                input_size = input['label'].size(0)
                input['label_split'] = torch.tensor(self.label_split)
                input = to_device(input, cfg['device'])
                optimizer.zero_grad()
                output = model(input)
                output['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                break
            print("Finish, Loss = {}".format(output['loss']))
            break

        grads = []
        for param in model.named_parameters():
            grads.append(param[1].grad)
        vec = gradtovec_new(grads).to(device)  # Trans gradient to vector

        if args['exp_name'] == 'aquila':
            v1, s_k = quantd_aquila(vec, self.mgr)
        elif args['exp_name'] == 'adap_plus_laq' or args['exp_name'] == 'adap':
            s_0 = 6
            if global_epoch == 0:
                self.f0 = output['loss']
                s_k = s_0
            else:
                s_k = math.ceil((self.f0 / output['loss']) ** 0.5 * s_0)
            if s_k > 16:
                s_k = 16
        elif args['exp_name'] == 'fedavg':
            s_k = 32
        elif args['exp_name'] == 'qsgd':
            s_k = 6
        elif args['exp_name'] == 'laq':
            s_k = 6
        evaluation = metric.evaluate(cfg['metric_name']['train']['Local'][:2], input, output)
        logger.append(evaluation, 'train', n=input_size)

        if args['exp_name'] == 'lena':
            beta = 200
            if global_epoch == 0 or torch.norm(self.e_lena + vec - self.pre_v_m_t) >= math.sqrt(beta) * torch.norm(
                    self.grad_pre):
                self.e_lena = torch.zeros_like(vec).to(device)
                s_k = 32
                v_m_t = self.e_lena + vec
            else:
                s_k = 6
                v_m_t = quantd(self.e_lena + vec, self.mgr, s_k)

            self.pre_v_m_t = v_m_t.to(device)
            self.grad_pre = vec.to(device)
            self.e_lena += vec - v_m_t
            print("Adaptive Quantization Bit is {}".format(s_k))
            quant_grads = vectograd(v_m_t, model)
            'OD: local grads after quantization'
            'vec: local grads before quantization'
            quant_grads_update = OrderedDict()
            i = 0
            for param in model.named_parameters():
                quant_grads_update[param[0]] = quant_grads[i]
                i += 1
            return quant_grads_update, s_k, vec, model.state_dict()
        elif args['exp_name'] == 'marina':
            if global_epoch == 0 or c_k == 1:
                s_k = 32
                print("Adaptive Quantization Bit is {}".format(s_k))
                v1 = quantd(vec, self.mgr, s_k)
                self.gr = v1.to(device)
                self.grad_pre = vec.to(device)
                quant_grads = vectograd(v1, model)
                quant_grads_update = OrderedDict()
                i = 0
                for param in model.named_parameters():
                    quant_grads_update[param[0]] = quant_grads[i]
                    i += 1
                return quant_grads_update, s_k, vec, model.state_dict()
            else:
                s_k = 6
                print("Adaptive Quantization Bit is {}".format(s_k))
                diff_grad = vec - self.grad_pre
                diff_quant = self.mgr - self.mmgr
                v1 = quantd(diff_grad, diff_quant, s_k)
                v1 += self.mgr
                self.gr = v1.to(device)
                self.grad_pre = vec.to(device)
                quant_grads = vectograd(v1, model)
                quant_grads_update = OrderedDict()
                i = 0
                for param in model.named_parameters():
                    quant_grads_update[param[0]] = quant_grads[i]
                    i += 1
                return quant_grads_update, s_k, vec, model.state_dict()
        elif args['exp_name'] == 'aquila':
            self.gr = v1.to(device)
            dvec = (v1 - vec).to(device)
            self.err = (dvec.dot(dvec))
            self.err_vec = dvec
            if global_epoch == 0:
                self.ehat_vec = torch.zeros(vec.shape).to(device)

            quant_grads = vectograd(v1, model)
            'OD: local grads after quantization'
            'vec: local grads before quantization'
            quant_grads_update = OrderedDict()
            i = 0
            for param in model.named_parameters():
                quant_grads_update[param[0]] = quant_grads[i]
                i += 1
            return quant_grads_update, s_k, vec, model.state_dict()
        else:
            print("Adaptive Quantization Bit is {}".format(s_k))
            v1 = quantd(vec, self.mgr, s_k)
            self.gr = v1.to(device)
            dvec = (v1 - vec).to(device)
            self.err = (dvec.dot(dvec))
            self.err_vec = dvec
            if global_epoch == 0:
                self.ehat_vec = torch.zeros(vec.shape).to(device)

            quant_grads = vectograd(v1, model)
            'OD: local grads after quantization'
            'vec: local grads before quantization'
            quant_grads_update = OrderedDict()
            i = 0
            for param in model.named_parameters():
                quant_grads_update[param[0]] = quant_grads[i]
                i += 1
            return quant_grads_update, s_k, vec, model.state_dict()


if __name__ == "__main__":
    main()