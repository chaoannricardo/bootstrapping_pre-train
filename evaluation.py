# -*- coding: utf8 -*-
from core.model import GBNEncoder, LNClassifier, GBNDecoder
from core.sim_metric import EDSim, SDPSim, CosSim
from core.util import get_optimizer, get_linear_schedule_with_warmup
from core.dataloader import load_graph
from torch_scatter import scatter_sum
import numpy as np
import os
import random
import torch
import torch.nn as nn

opt_encoder = {'data_dir': '../KnowledgeGraph_materials/data_kg/bootstrapnet_data/boot_pretrain_data_revised/',
       'output_model_file': './models/test',
       'input_model_file': './models/210919_selfSupervised_2',
       'sim_metric': CosSim(),
        'k_hop': 1,
       'nl_weight': 0.01,
       'local': False,
       'n_layer': 3,
       'dropout': 0.1,
       'negative_slope': 0.2,
       'bias': False,
       'device': torch.device(type='cuda', index=0),
       'cpu': True,
       'seed': 1,
       'n_epoch': 100,
       'optimizer': 'adam',
       'lr': 0.001,
       'decay': 0.001,
       'max_grad_norm': 1.0,
       'feature_type': 'random',
       'feature_dim': 50,
       'edge_feature_dim': 5}

opt_decoder = {'dataset': '../KnowledgeGraph_materials/data_kg/bootstrapnet_data/boot_pretrain_data/gum_train/',
               'input_model_file': './models/210914_fineTuned_decoder',
               'output_model_file': '',
               'method': 'multi_view',
               'sim_metric': 'cos',
               'n_iter': 20,
               'min_match': 1,
               'n_expansion': 10,
               'k_hop': 2,
               'un_weight': 0.01,
               'local': False,
               'mean_updated': False,
               'n_layer': 3,
               'dropout': 0.1,
               'negative_slope': 0.2,
               'bias': False,
               'device': torch.device(type='cuda', index=0),
               'cpu': True,
               'seed': 1,
               'init_encoder_epoch': 200,
               'init_decoder_epoch': 200,
               'encoder_epoch': 50,
               'decoder_epoch': 50,
               'optimizer': 'adam',
               'lr': 0.001,
               'decay': 0.001,
               'max_grad_norm': 1.0,
               'feature_type': 'glove',
               'feature_dim': 50,
               'edge_feature_dim': 5}


def comprise_data(opt, encoder, weight, load_classifier=False):
    ''' setting device '''
    device_encoder = "cpu"
    if not opt["cpu"]:
        device_encoder = opt['device']

    print('loading %s......' % opt['dataset'])
    pkl_path = 'graph_' + opt['feature_type'] + '.pkl'
    graph_data, graph = load_graph(opt, pkl_path, pyg=True)
    opt['n_class'] = len(graph.node_s.itol)

    graph_data = graph_data.to(device_encoder)
    graph_data.x = (graph_data.x[0].to(device_encoder),
                    graph_data.x[1].to(device_encoder))
    d_es = graph_data.x[0].size(-1)
    classifier = LNClassifier(d_es * 2, 1)

    if load_classifier:
        classifier.load_state_dict(torch.load(opt['input_model_file']+'_MLPClassifier.pth'))
        print("Classifier model file loaded!")

    classifier.to(device_encoder)

    parameters = [
        {'params': [p for p in encoder.parameters() if p.requires_grad]},
        {'params': [p for p in classifier.parameters() if p.requires_grad]}]
    optimizer = get_optimizer(opt['optimizer'], parameters,
                              opt['lr'], opt['decay'])
    n_epoch = opt['n_epoch'] * weight
    warm_step = n_epoch * 0.1
    scheduler = get_linear_schedule_with_warmup(optimizer, warm_step, n_epoch,
                                                min_ratio=0.1)
    print('loaded!')
    return classifier, optimizer, scheduler, graph_data, weight


def contrastive_loss(encoder_output, graph_data, sim_metric):
    es, ps = encoder_output
    e_size = graph_data.x[0].size(0)

    ee_pos = graph_data.node_pos_index
    ee_neg = _contrastive_sample(ee_pos.size(1), graph_data.node_neg_index)
    ep_pos = graph_data.edge_pos_index
    ep_neg = _contrastive_sample(ep_pos.size(1), graph_data.edge_neg_index)

    ep1, ep2 = es.index_select(0, ee_pos[0]), es.index_select(0, ee_pos[1])
    link_sim = sim_metric(ep1, ep2, flatten=True, method='exp')
    en1, en2 = es.index_select(0, ee_neg[0]), es.index_select(0, ee_neg[1])
    non_sim = sim_metric(en1, en2, flatten=True, method='exp')
    pp1, pp2 = es.index_select(0, ep_pos[0]), ps.index_select(0, ep_pos[1])
    pos_sim = sim_metric(pp1, pp2, flatten=True, method='exp')
    pn1, pn2 = es.index_select(0, ep_neg[0]), ps.index_select(0, ep_neg[1])
    neg_sim = sim_metric(pn1, pn2, flatten=True, method='exp')

    en_sum = scatter_sum(non_sim, ee_neg[0], dim=-1, dim_size=e_size)
    link_loss = link_sim / (link_sim + en_sum.index_select(0, ee_pos[0]))
    ep_sum = scatter_sum(neg_sim, ep_neg[0], dim=-1, dim_size=e_size)
    ep_loss = pos_sim / (pos_sim + ep_sum.index_select(0, ep_pos[0]))

    loss = torch.cat([-link_loss.log(), -ep_loss.log()], dim=-1).mean()
    return loss


def _contrastive_sample(pos_size, neg_index):
    # limited by the GPU memory, we randomly sample some neg-neighbors
    neg_size = pos_size * 5
    indices = torch.randperm(neg_index.size(1), device=neg_index.device)[:neg_size]
    neg_sample = neg_index[:, indices]
    return neg_sample


def neighbor_learning_pretrain(opt, encoder, data):
    sim_metric = opt['sim_metric']
    es, ps = encoder(data)
    loss = opt['nl_weight'] * contrastive_loss((es, ps), data, sim_metric)
    return loss


def edge_mask_loss(encoder_output, graph_data, masked_indice, classifier):
    edge_index = graph_data.edge_index
    edge_index = edge_index[:, masked_indice]
    size = (graph_data.x[0].size(0), graph_data.x[1].size(0))
    neg_edge_index = _negative_sample(graph_data.edge_index, size,
                                      num_neg=masked_indice.size(0))
    es, ps = encoder_output
    criterion = nn.BCEWithLogitsLoss()
    pos_score = classifier(torch.cat([es[edge_index[0]], ps[edge_index[1]]], dim=-1))
    neg_score = classifier(torch.cat([es[neg_edge_index[0]], ps[neg_edge_index[1]]], dim=-1))
    loss = criterion(pos_score, torch.ones_like(pos_score)) + \
        criterion(neg_score, torch.zeros_like(neg_score))
    loss = loss / 2
    return loss


def _negative_sample(edge_index, size, num_neg):
    # Handle '|V|^2 - |E| < |E|'.
    count = size[0] * size[1]
    num_neg = min(num_neg, count - edge_index.size(1))

    row, col = edge_index
    idx = row * size[1] + col

    alpha = 1 / (1 - 1.2 * (edge_index.size(1) / count))

    perm = sample(count, int(alpha * num_neg))
    mask = torch.from_numpy(np.isin(perm, idx.to('cpu'))).to(torch.bool)
    perm = perm[~mask][:num_neg].to(edge_index.device)
    row = perm // size[1]
    col = perm % size[1]
    neg_edge_index = torch.stack([row, col], dim=0)
    return neg_edge_index


def sample(high: int, size: int, device=None):
    size = min(high, size)
    return torch.tensor(random.sample(range(high), size), device=device)


def link_mask_pretrain(opt, encoder, classifier, data):
    x = data.x
    edge_attr = data.edge_attr
    edge_index = data.edge_index
    edge_size = edge_index.size(1)
    neg_size = int(edge_size * 0.1)

    indices = torch.randperm(edge_size, device=edge_index.device)
    masked_indices = indices[:neg_size]
    remain_indices = indices[neg_size:]

    output = encoder(x, edge_index[:, remain_indices],
                     edge_attr[remain_indices])
    loss = edge_mask_loss(output, data, masked_indices, classifier)
    return loss


def edge_mask(opt, encoder, batch, batch_id, ite):
    classifier, optimizer, scheduler, data, weight = batch

    total_loss = 0
    for i in range(weight):
        encoder.train()
        optimizer.zero_grad()
        loss_nl = neighbor_learning_pretrain(opt, encoder, data)
        loss_lm = link_mask_pretrain(opt, encoder, classifier, data)
        loss = (loss_nl + loss_lm) / weight
        loss.backward()
        # nn.utils.clip_grad_norm_(encoder.parameters(), opt['max_grad_norm'])
        # nn.utils.clip_grad_norm_(classifier.parameters(), opt['max_grad_norm'])
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    print('Ite[%d]-Batch[%d]--loss:%.5f, lr:%.7f' %
          (ite, batch_id, total_loss, scheduler.get_last_lr()[0]))

    return classifier


if __name__ == '__main__':
    ''' set device '''
    device_encoder = "cpu"
    if not opt_encoder["cpu"]:
        device_encoder = opt_encoder['device']

    ''' Load models '''
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    encoder = GBNEncoder(opt_encoder)
    encoder.load_state_dict(torch.load(opt_encoder['input_model_file'] + '_encoder.pth'))
    if not opt_encoder["cpu"]:
        encoder = encoder.to(device_encoder)

    ''' Load datasets '''
    datasets = []
    with open(os.path.join(opt_encoder['data_dir'], 'unsupervised_dataset.txt'), 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            assert line and line[0]
            dataset = line[0]
            # setting weight to 1 when predicting
            weight = 1
            datasets.append((dataset, weight))

    batches = []
    for dataset, weight in datasets:
        opt_encoder['dataset'] = os.path.join(opt_encoder['data_dir'], dataset)
        batches.append(comprise_data(opt_encoder, encoder, weight, load_classifier=True))

    ''' Begin predicting with model '''
    # for i, batch in enumerate(batches):
    #     edge_mask(opt_encoder, encoder, batch, i + 1)

    for ite in range(1, opt_encoder['n_epoch'] + 1):
        for i, batch in enumerate(batches):
            classifier = edge_mask(opt_encoder, encoder, batch, i+1, ite)
