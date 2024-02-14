import os
import random
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, precision_recall_curve

import torch
from torch import nn
from torch.utils.data import DataLoader

### Train and Validation ###
def train_ConvPool(args, train_dataset, valid_dataset, test_dataset, num, data_type, model, result_folder):
    batch_size = int(min(len(train_dataset)/10, args.batch_size))
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model.cuda()
    
    weight1 = torch.Tensor(train_dataset.weight).cuda()
    ce = torch.nn.CrossEntropyLoss(weight=weight1) 
    optim_h = torch.optim.Adam(model.parameters(), lr=args.lr)
    optim_p1 = torch.optim.Adam([model.protos], lr=args.lr)
    
    model.init_protos(train_loader)
    
    valid_loss_list = []
    valid_acc_list=[]
    performance = []
    
    correct = 0
    test_total = 1e-8
    
    for epoch in range(args.num_epoch):
        ### Training ###
        model.train()
        total_step = len(train_loader)
        total, total_ce_loss = 0, 0
       
        for batch in train_loader:
            data, labels = batch['data'].cuda(), batch['labels'].cuda()
            x1, logits, _ = model(data)

            ce_loss = ce(logits, labels)
            optim_h.zero_grad()
            ce_loss.backward(retain_graph=True)
            optim_h.step()

            dtw_loss = model.compute_aligncost(x1)
            optim_p1.zero_grad()
            dtw_loss.backward(retain_graph=True)
            optim_p1.step()
            
        ### Inference ###
        predictions = []
        answers = []
        correct_val, val_total = 0, 0
        total_step = len(valid_loader)
        total_ce_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in valid_loader:
                data, labels = batch['data'].cuda(), batch['labels'].cuda()
                answers.extend(labels.detach().cpu().numpy())
                
                x1, logits, _ = model(data)
                _, predicted = torch.max(logits, 1)
                predictions.extend(predicted.detach().cpu().numpy())
                
                val_total += data.size(0)
                correct_val += (predicted == labels).sum().item()
                
                ce_loss = ce(logits, labels)
                total_ce_loss += ce_loss.item() * data.size(0)
                
            valid_loss_list.append(total_ce_loss/val_total)
            valid_acc_list.append(correct_val/val_total)

        if (epoch==0) or (epoch>0 and (min(valid_loss_list[:-1])>valid_loss_list[-1])):
            performance, correct, test_total = _inference(args, model, ce, test_loader, epoch, valid_loss_list, valid_acc_list, result_folder, data_type, num)
    
    print('The Best Test Accuracy: {:.4f}'.format(correct/test_total))
    
    return performance


### Save best model and Get performance ###
def _inference(args, model, ce, test_loader, epoch, valid_loss_list, valid_acc_list, result_folder, data_type, num):
    torch.save({
                'epoch': epoch,
                'loss' : valid_loss_list[-1],
                'acc' : valid_acc_list[-1],
                'model_state_dict' : model.state_dict(),
    }, os.path.join(result_folder, f'{args.model}-best-{data_type}-{num}.pt'))

    predictions = []
    prob = []
    answers = []
    correct, test_total = 0, 0
    total_step = len(test_loader)
    total_ce_loss = 0
    with torch.no_grad():
        model.eval()
        for batch in test_loader:
            data, labels = batch['data'].cuda(), batch['labels'].cuda()
            answers.extend(labels.detach().cpu().numpy())

            x1, logits, _ = model(data)
            prob.extend(nn.Softmax()(logits).detach().cpu().numpy())
            _, predicted = torch.max(logits, 1)
            predictions.extend(predicted.detach().cpu().numpy())

            test_total += data.size(0)
            correct += (predicted == labels).sum().item()

            ce_loss = ce(logits, labels)
            total_ce_loss += ce_loss.item() * data.size(0)
            
    print('\tEpoch [{:3d}/{:3d}], Test Loss: {:.4f}, Test Accuracy: {:.4f}'
    .format(epoch+1, args.num_epoch, total_ce_loss/test_total, correct/test_total))

    c = len(np.unique(np.array(answers)))
    y_prob = np.array(prob)
    prauc_list = []
    auroc_list = []
    if c < 3: 
        prauc_list.append(average_precision_score(answers, np.array(prob)[:, 1]))
        auroc_list.append(roc_auc_score(answers, np.array(prob)[:, 1]))
    else:
        Y = label_binarize(answers, classes=[*range(c)])
        for i in range(c):
            prauc_list.append(average_precision_score(Y[:, i], np.array(prob)[:, i]))
            auroc_list.append(roc_auc_score(Y[:, i], np.array(prob)[:, i]))

    performance = [total_ce_loss/test_total,
                   correct/test_total, 
                   f1_score(answers, predictions, average='macro'),
                   f1_score(answers, predictions, average='micro'),
                   f1_score(answers, predictions, average='weighted'),
                   np.mean(np.array(auroc_list)),
                   np.mean(np.array(prauc_list))
                  ]
    
    return performance, correct, test_total