from torch.autograd import Variable
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


def metrics(predictions, gts, label_values):
    cm = confusion_matrix(
        gts,
        predictions,
        range(len(label_values)))

    print("Confusion matrix :")
    print(cm)

    print("---")

    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    print("{} pixels processed".format(total))
    print("Total accuracy : {:.2f}%".format(accuracy))

    print("---")

    # Compute F1 score
    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except:
            # Ignore exception if there is no element in class i for test set
            pass
    print("F1Score :")
    for l_id, score in enumerate(F1Score):
        print("{}: {:.2f}".format(label_values[l_id], score))

    print("---")

    # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe)
    print("Kappa: {:.2f}".format(kappa))
    return accuracy


class Tester:

    def __init__(self, data_loader, model, classes, mode='cuda'):
        super(Tester, self).__init__()
        self.model = model
        self.data_loader = data_loader
        self.mode = mode
        self.classes = classes
        self.params = model.parameters()

    def test(self, acc_only=False):

        all_preds = []
        all_gts = []

        for _, batch in enumerate(tqdm(self.data_loader)):
            data = Variable(batch['image'])
            target = batch['class_code']
            # forward
            if self.mode == 'cuda':
                data = data.cuda()
            output = self.model.forward(data)
            output = output.data.cpu().numpy()

            pred = np.argmax(output, axis=-1)
            all_preds.append(pred)
            all_gts.append(target)

        #class_name = list(set(batch['class_name']))
        # metrics(pred, target, self.classes)
        accuracy = metrics(np.concatenate([p for p in all_preds]),\
                           np.concatenate([p for p in all_gts]).ravel()-1, self.classes)

        if not acc_only:
            return accuracy, all_preds, all_gts
        else:
            return accuracy
