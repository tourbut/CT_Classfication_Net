from torch.autograd import Variable
import torch

from metrics import accuracy

import utils
from tqdm.auto import tqdm
from torchmetrics.functional.classification import multiclass_auroc
from torchmetrics.classification import MulticlassConfusionMatrix,MulticlassROC


def test(device, data_loader, model, criterion, logger,num_classes=2, best_yn=''):
    print('test')

    
    losses = utils.AverageMeter(name='losses')
    accuracies = utils.AverageMeter(name='accuracies')

    pred = []
    labels = []
    with torch.no_grad():
        model.eval()
        for i, (inputs, targets) in tqdm(enumerate(data_loader)):

            inputs = Variable(inputs).to(device)
            targets = Variable(targets).to(device)
            outputs = model(inputs)

            for j in range(len(outputs)):
                pred.append(outputs[j].data.to(device))
                labels.append(targets[j].data.to(device))    
            loss = criterion(outputs, targets)
            acc = accuracy(outputs.data, targets.data,device=device)

            losses.update(loss.data, inputs.size(0))
            accuracies.update(acc, inputs.size(0))
            
    pred   = torch.stack(pred).to(device)
    labels =  torch.stack(labels).to(device)
    
    metric = MulticlassConfusionMatrix(num_classes=num_classes).to(device)

    ConfusionMatrix = metric(pred, labels)

    auroc = multiclass_auroc(pred, labels, num_classes=num_classes, average=None, thresholds=None).to(device)

    roc_metric = MulticlassROC(num_classes=num_classes, thresholds=None).to(device)
    fpr, tpr, thresholds = roc_metric(pred, labels)
    _fpr = []
    _tpr = []
    _thresholds = []
    for i in range(num_classes):
        _fpr.append(fpr[i].tolist())
        _tpr.append(tpr[i].tolist())
        _thresholds.append(thresholds[i].tolist())
    logger.log({
        'best_yn': best_yn,
        'loss': losses.avg.item(),
        'acc': accuracies.avg.item(),
        'ConfusionMatrix' : ConfusionMatrix.tolist(),
        'auroc' : auroc.tolist(),
        'fpr'   : _fpr,
        'tpr'   : _tpr,
        'thresholds' : _thresholds
    })


    print('Loss : {loss.avg:.4f}\t Acc : {acc.avg:.5f}\t'.format(loss=losses, acc=accuracies))
    print(ConfusionMatrix)
    print(auroc)
    return losses.avg.item(), accuracies.avg.item(), ConfusionMatrix, auroc,_fpr,_tpr,_thresholds
