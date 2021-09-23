# -*- coding: utf8 -*-
from core.sim_metric import EDSim, SDPSim, CosSim
from core.model import GBNEncoder, GBNDecoder
from core.dataloader import load_graph, load_seed
from tqdm import tqdm
import numpy as np
import codecs
import os
import random
import sys
import torch

K_HOP = 1
ENCODER_MODEL_PATH = './models/210922_KG_Semiconductor_Filtered_FineTuned_encoder'
DECODER_MODEL_PATH = './models/210922_KG_Semiconductor_Filtered_FineTuned_decoder'
OUTPUT_FILE = "./outputs/210923_Semiconductor"
DATA_DIR = "../KnowledgeGraph_materials/data_kg/baiduDatasetTranditional_GBN_wholeNode_dependency/doc_un_1/"
FEATURE_TYPE = "random"
IS_CPU = True
DEVICE = torch.device(type='cuda', index=0)
SIM_METRIC = CosSim()

opt_encoder = {'data_dir': DATA_DIR,
               'input_model_file': ENCODER_MODEL_PATH,
               'sim_metric': SIM_METRIC,
               'k_hop': K_HOP,
               'nl_weight': 0.01,
               'local': False,
               'n_layer': 3,
               'dropout': 0.1,
               'negative_slope': 0.2,
               'bias': False,
               'device': DEVICE,
               'cpu': IS_CPU,
               'seed': 1,
               'feature_type': FEATURE_TYPE,
               'feature_dim': 50,
               'edge_feature_dim': 5}

opt_decoder = {'dataset': DATA_DIR,
               'input_model_file': DECODER_MODEL_PATH,
               'method': 'multi_view',
               'sim_metric': SIM_METRIC,
               'n_iter': 20,
               'min_match': 1,
               'n_expansion': 10,
               'k_hop': K_HOP,
               'un_weight': 0.01,
               'local': False,
               'mean_updated': False,
               'n_layer': 3,
               'dropout': 0.1,
               'negative_slope': 0.2,
               'bias': False,
               'device': DEVICE,
               'cpu': IS_CPU,
               'seed': 1,
               'init_encoder_epoch': 1,
               'init_decoder_epoch': 1,
               'encoder_epoch': 50,
               'decoder_epoch': 50,
               'optimizer': 'adam',
               'lr': 0.001,
               'decay': 0.001,
               'max_grad_norm': 1.0,
               'feature_type': FEATURE_TYPE,
               'feature_dim': 50,
               'edge_feature_dim': 5}

SIM_TABLE = {
    'ed': EDSim(),
    'sdp': SDPSim(),
    'cos': CosSim()
}


def update_s(opt, encoder, decoder, graph_data, seeds, mv_iter=0):
    edge_index = graph_data.node_edge_index
    encoder.eval()

    n_iter = opt['n_iter']
    n_epoch = opt['decoder_epoch'] if mv_iter else opt['init_decoder_epoch']

    for i in tqdm(range(1, n_epoch + 1)):
        decoder.eval()
        es = encoder(graph_data)[0]
        es = es.detach()

        for parameter in decoder.parameters():
            parameter.requires_grad = False

        for parameter in encoder.parameters():
            parameter.requires_grad = False

        probs, selects, hxes = decoder.expand(es, edge_index, seeds, n_iter)
        loss = 0
        for ite, (iter_probs, iter_selects) in enumerate(zip(probs, selects)):
            score = np.exp(-ite / n_iter)
            select = torch.cat(iter_selects, dim=0)
            prob = torch.cat(iter_probs, dim=0)

        for index, item in enumerate(probs):
            for subIndex, subItem in enumerate(item):
                probs[index][subIndex] = probs[index][subIndex].cpu().detach().numpy().tolist()

        for index, item in enumerate(selects):
            for subIndex, subItem in enumerate(item):
                selects[index][subIndex] = selects[index][subIndex].cpu().detach().numpy().tolist()

        output_file_prob.write((str(i) + "+++++++++++++++++++++++++++++++++++++++++++++\n"))
        output_file_prob.write("Prob =============================================\n" + str(probs) + "\n\n\n\n")

        output_file_selection.write((str(i) + "+++++++++++++++++++++++++++++++++++++++++++++\n"))
        output_file_selection.write("Selects =============================================\n" + str(selects) + "\n\n\n\n")


def fine_tune_decoder(opt, encoder, decoder, graph_data, seeds,
                      dev_seeds=None):
    update_s(opt, encoder, decoder, graph_data, seeds, mv_iter=0)


if __name__ == '__main__':
    device = "cpu"

    output_file_prob = codecs.open(OUTPUT_FILE + "_probability.txt", mode="w", encoding="utf8")
    output_file_selection = codecs.open(OUTPUT_FILE + "_selection.txt", mode="w", encoding="utf8")

    if not opt_encoder["cpu"]:
        device = opt_encoder['device']

    if not opt_encoder["cpu"] and opt_encoder["seed"]:
        torch.cuda.manual_seed(opt_encoder["seed"])
        torch.cuda.manual_seed_all(opt_encoder["seed"])

    if opt_encoder['local']:
        opt_encoder['k_hop'] = 1

    if opt_encoder['feature_type'] == 'bert':
        opt_encoder['feature_dim'] = 768

    opt_encoder['sim_metric'] = opt_encoder['sim_metric'].to(device)

    encoder = GBNEncoder(opt_encoder)
    # load encoder model
    if opt_encoder['input_model_file']:
        encoder.load_state_dict(torch.load(opt_encoder['input_model_file'] + '.pth'))

    print('======================== Do Bootstrapping ==============================')
    pkl_path = 'graph_' + opt_decoder['feature_type'] + '.pkl'
    graph_data, graph = load_graph(opt_decoder, pkl_path, pyg=True)
    seed_file = opt_decoder['dataset'] + '/seeds.txt'
    seeds = load_seed(graph.node_s, seed_file)
    seeds = [torch.LongTensor(seed) for seed in seeds]
    opt_encoder['n_class'] = len(graph.node_s.itol)

    sim_metric = opt_encoder['sim_metric']
    decoder = GBNDecoder(opt_decoder, sim_metric)
    # load decode model
    if opt_decoder['input_model_file']:
        decoder.load_state_dict(torch.load(opt_decoder['input_model_file'] + '.pth'))

    graph_data = graph_data.to(opt_encoder['device'])
    graph_data.x = (graph_data.x[0].to(opt_encoder['device']),
                    graph_data.x[1].to(opt_encoder['device']))

    seeds = [seed.to(opt_encoder['device']) for seed in seeds]
    encoder = encoder.to(opt_encoder['device'])
    decoder = decoder.to(opt_decoder['device'])

    fine_tune_decoder(opt_decoder, encoder, decoder, graph_data, seeds)
