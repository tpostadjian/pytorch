from torch.autograd import Variable
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


def metrics(preds, targets, classes):

    confusion = confusion_matrix(
        targets,
        preds,
        range(len(classes)))
    print("*******************")

    print("Confusion matrix: ")
    print(confusion)

    print("*******************")

    # Overall accuracy
    total = sum(sum(confusion))
    accuracy = sum([confusion[x][x] for x in range(len(confusion))])
    accuracy *= 100 / float(total)
    print("Evaluation on {} pixels: ".format(total))
    print("Overall accuracy: {:.2f}%".format(accuracy))

    print("*******************")

    # F1-score / class
    F1Score = np.zeros(len(classes))
    for cls in range(len(classes)):
        try:
            F1Score[cls] = 2.*confusion[cls, cls]/(np.sum(confusion[cls, :])+np.sum(confusion[:, cls]))
        except:
            pass
    print("F1Score: ")
    for cls, score in enumerate(F1Score):
        print("{}: {:.2f}".format(classes[cls], score))

    print("*******************")

    # Kappa
    total = np.sum(confusion)
    Pa = np.trace(confusion) / float(total)
    Pe = np.sum(np.sum(confusion, axis=0) * np.sum(confusion, axis=1)) / float(total * total)
    kappa = (Pa - Pe) / (1 - Pe)
    print("Kappa: {:.2f}".format(kappa))

    print("*******************")

    # IoU = TP/(TP+FN+FP)
    iou = np.zeros(len(classes))
    for cls in range(len(classes)):
        preds_inds = preds == cls
        targets_inds = targets == cls
        intersection = np.sum(preds_inds[targets_inds]*1)
        union = np.sum(preds_inds*1) + np.sum(targets*1) - intersection
        if union == 0:
            iou[cls] = np.nan  # no target / groundtruth for class cls
        else:
            iou[cls] = intersection/union
    print("IoU: ")
    for cls, score in enumerate(iou):
        print("{}: {:.2f}".format(classes[cls], score))

    return accuracy


class Tester:

    def __init__(self, data_loader, model, criterion, classes, mode='cuda'):
        super(Tester, self).__init__()
        self.data_loader = data_loader
        self.model = model
        self.criterion = criterion
        self.classes = classes
        self.mode = mode
        self.params = model.parameters()
        self.avg_loss = 10000

    def test(self):

        loss_list = np.zeros(1000000)
        all_preds = []
        all_targets = []

        for it, batch in enumerate(tqdm(self.data_loader)):
            data = Variable(batch['image'])
            target = batch['class_code']
            # forward
            if self.mode == 'cuda':
                data = data.cuda()
                target = target.cuda()
            output = self.model(data)
            loss = self.criterion(output.float(), target)
            output = output.data.cpu().numpy()
            target = target.data.cpu().numpy()

            pred = np.argmax(output, axis=-1)
            all_preds.append(pred)
            all_targets.append(target)
            loss_list[it] = loss.item()
        self.avg_loss = np.mean(loss_list[np.nonzero(loss_list)])

        accuracy = metrics(np.concatenate([p for p in all_preds]), np.concatenate([p for p in all_targets]), self.classes)

        return accuracy
