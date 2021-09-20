# -*- coding: utf8 -*-
from core.model import GBNEncoder, LNClassifier, GBNDecoder
from core.sim_metric import EDSim, SDPSim, CosSim
from core.util import get_optimizer, get_linear_schedule_with_warmup
from core.dataloader import load_graph
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
    device = "cpu" if opt["cpu"] else opt['device']

    pkl_path = 'graph_' + opt['feature_type'] + '.pkl'
    graph_data, graph = load_graph(opt, pkl_path, pyg=True)
    opt['n_class'] = len(graph.node_s.itol)

    graph_data = graph_data.to(device)
    graph_data.x = (graph_data.x[0].to(device),
                    graph_data.x[1].to(device))

    d_es = graph_data.x[0].size(-1)
    classifier = LNClassifier(d_es * 2, 1)

    if load_classifier:
        classifier.load_state_dict(torch.load(opt['input_model_file']+'_MLPClassifier.pth'))
        print("Classifier model file loaded!")

    classifier.to(device)
    parameters = [
        {'params': [p for p in encoder.parameters() if p.requires_grad]},
        {'params': [p for p in classifier.parameters() if p.requires_grad]}]
    optimizer = get_optimizer(opt['optimizer'], parameters,
                              opt['lr'], opt['decay'])
    n_epoch = opt['n_epoch'] * weight
    warm_step = n_epoch * 0.1
    scheduler = get_linear_schedule_with_warmup(optimizer, warm_step, n_epoch,
                                                min_ratio=0.1)
    return classifier, optimizer, scheduler, graph_data, weight


def sigmoid(x):
    z = np.exp(-x)
    sig = 1 / (1 + z)

    return sig


def sample(high: int, size: int, device=None):
    size = min(high, size)
    return torch.tensor(random.sample(range(high), size), device=device)


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


def edge_mask_loss(encoder_output, graph_data, masked_indice, classifier):
    # get index of edges
    edge_index = graph_data.edge_index
    edge_index = edge_index[:, masked_indice]
    size = (graph_data.x[0].size(0), graph_data.x[1].size(0))
    neg_edge_index = _negative_sample(graph_data.edge_index, size,
                                      num_neg=masked_indice.size(0))
    es, ps = encoder_output
    pos_score = classifier(torch.cat([es[edge_index[0]], ps[edge_index[1]]], dim=-1))
    neg_score = classifier(torch.cat([es[neg_edge_index[0]], ps[neg_edge_index[1]]], dim=-1))

    return pos_score, neg_score


def link_mask_pretrain(opt, encoder, classifier, data):
    x = data.x
    edge_attr = data.edge_attr
    edge_index = data.edge_index
    edge_size = edge_index.size(1)
    neg_size = int(edge_size * 0.1)
    indices = torch.randperm(edge_size, device=edge_index.device)

    output = encoder(x, edge_index[:, indices],
                     edge_attr[indices])
    pos_score, neg_score = edge_mask_loss(output, data, indices, classifier)
    return pos_score, neg_score


def edge_mask(opt, encoder, batch, batch_id):
    classifier, optimizer, scheduler, data, weight = batch

    # activate evaluation mode
    encoder.eval()
    classifier.eval()

    pos_score, neg_score = link_mask_pretrain(opt, encoder, classifier, data)

    criterion = nn.BCEWithLogitsLoss()

    for scoreIndex, score in enumerate(pos_score):
        print(score, neg_score[scoreIndex])
        # pos_loss = criterion(score, torch.ones_like(score))
        # neg_loss = criterion(neg_score[scoreIndex], torch.ones_like(neg_score[scoreIndex]))
        # total_loss = (pos_loss + neg_loss) / 2
        # total_loss = total_loss.cpu().detach().numpy()
        # print("total loss", total_loss, "pos_loss", pos_loss, "neg_loss", neg_loss, "pos score", score, "neg score", neg_score[scoreIndex])
        # prob = sigmoid(score.detach().numpy())
        # if ((prob < 0 or prob > 1)):
        #     print("hi")


if __name__ == '__main__':
    ''' Load models '''
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    encoder = GBNEncoder(opt_encoder)
    encoder.load_state_dict(torch.load(opt_encoder['input_model_file'] + '_encoder.pth'))
    decoder = GBNDecoder(opt_decoder, opt_decoder['sim_metric'])
    decoder.load_state_dict(torch.load(opt_decoder['input_model_file'] + '.pth'))

    # print(encoder.eval(), decoder.eval())

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
    for i, batch in enumerate(batches):
        edge_mask(opt_encoder, encoder, batch, i + 1)
