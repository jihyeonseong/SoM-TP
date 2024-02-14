import os
import random
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import torch
from torch.utils.data import DataLoader

from src.lrp_for_model import construct_lrp

def LRP_SoMTP(args, train_dataset, test_dataset, num, data_type, model, model_folder, result_folder, name):
    ### Computation of lrp ###
    def compute_lrp(lrp_model, x, y, class_specific):
        output = lrp_model.forward(x, y=y, class_specific=class_specific)
        all_relevnace = output['all_relevnaces']
        return all_relevnace
    
    batch_size = int(min(len(train_dataset)/10, args.batch_size))
    class_num = train_dataset.num_classes
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    perf = pd.read_csv(os.path.join(model_folder, f'{args.model}_{args.deep_extract}_{args.pool}_{args.pool_op}_uni_performance.csv'))
    acc = perf.iloc[num, 2]
    
    checkpoint = torch.load(os.path.join(model_folder, f'{args.model}-best-{data_type}-{num}.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()
    model = model.eval()
    n = model.protos_num
    lrp_model = construct_lrp(args, model, "cuda")
    
    os.makedirs(result_folder, exist_ok=True) 
    predictions = []
    lrp_list = []

    for i, batch in enumerate(test_loader):
        data, labels = batch['data'].cuda(), batch['labels'].cuda()
        x1, classes, ensem, one, op, attn = model(data)
        predictions.extend(torch.max(classes,1)[1].detach().cpu().numpy())
        lrps = compute_lrp(lrp_model, data.unsqueeze(2), labels, class_specific=True)[::-1] 
        lrp = lrps[0].squeeze(2)
        lrp_list.append(lrp)

        lrp_list_ = lrp_list[0].detach().cpu().numpy()
        relevance_ =np.abs(lrp_list_).mean(axis=1)
        attributions_occ_list = relevance_[:, 4:-4]

        fig, axs = plt.subplots(1, figsize=(6,2), sharex=True, sharey=True)

        if i == args.batch:
            x = np.linspace(0, len(data.detach().cpu().numpy()[args.sample][0]), 
            len(data.detach().cpu().numpy()[args.sample][0]))
            y = data.detach().cpu().numpy()[args.sample][0]
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            occlusion = attributions_occ_list[args.sample].reshape(-1, 1)[:, 0]

            norm = plt.Normalize(np.array(attributions_occ_list)[args.sample,:].min(),
                                 np.array(attributions_occ_list)[args.sample,:].max())

            lc = LineCollection(segments, cmap='Reds', norm=norm)
            lc.set_array(occlusion)
            lc.set_linewidth(3)
            line = axs.add_collection(lc)

            axs.set_xlim(x.min(), x.max())
            axs.set_ylim(y.min()-0.5, y.max()+0.5)

            axs.plot(y, color='black', alpha=0.1, linewidth=3)
            axs.set_title(f'SoM-TP', color='k', fontsize=25)
            axs.tick_params(axis='x', size=1, color='white')
            axs.tick_params(axis='y', size=1, color='white')
            axs.set_xticklabels([''])
            axs.set_yticklabels([''])
            fig.tight_layout()
            plt.savefig(os.path.join(result_folder,f'{name}_LRP_{args.batch}_{args.sample}.pdf'))
            break