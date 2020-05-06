from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch


class Trainner:
    def __init__(self, network, optimizer, train_loader, val_loader=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.train_loss_hist = []
        self.val_loss_hist = []
        self.val_acc_hist = []
        self.val_fscore_hist = []

        self.optimizer = optimizer
        self.network = network.to(self.device)

    def train(self, n_epochs, verbose=True, log_interval=10, save=False, save_best_only=False):
        # coloca a rede em modo de treino (habilita regularização)
        self.network.train()

        if save_best_only:
            save = False

        elif save:
            save_best_only = False

        loss_min = float('inf')
        best_acc = None
        best_epoch = None
        best_loss = None
        net_best_weights = None
        opt_best_weights = None
        # loop de treinamento
        for epoch in range(1, n_epochs + 1):
            loss_batch = 0.
            for batch_idx, (data, target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                output = self.network(data.to(self.device))

                loss = F.nll_loss(output, target.to(self.device))
                # armazena a soma das loss de todos os batches
                loss_batch += loss.data.cpu().item()
                loss.backward()

                self.optimizer.step()

                if batch_idx % log_interval == 0 and verbose:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(self.train_loader.dataset),
                               100. * batch_idx / len(self.train_loader), loss.data.cpu().item()))

            # guarda a loss da época
            loss_epoch = loss_batch / len(self.train_loader)
            self.train_loss_hist.append(loss_epoch)

            if verbose:
                print('---\n\tEpoch {} trainning: \tMean Loss: {:.6f}'.format(epoch, self.train_loss_hist[epoch - 1]))

            if not self.val_loader is None:
                val_loss, acc, _, _, f1, _ = self.evaluate(self.val_loader)
                self.val_loss_hist.append(val_loss)
                self.val_acc_hist.append(acc)
                self.val_fscore_hist.append(f1)

                if verbose:
                    print(
                        '\tEpoch {} validation: [{}] \tMean Loss: {:.6f}\tAcc: {:.2f}\tf-score: {:.2f}\n---'.format(
                            epoch,
                            len(self.val_loader.dataset),
                            val_loss,
                            acc,
                            f1))

            # salva os pesos da rede
            if save_best_only and val_loss <= loss_min:
                best_acc = acc
                best_epoch = epoch
                best_loss = loss_epoch
                net_best_weights = self.network.state_dict()
                opt_best_weights = self.optimizer.state_dict()

                loss_min = val_loss

            elif save:
                torch.save(self.network.state_dict(),
                           'model_weights/model_epoch-{}_loss-{:.5f}_valacc-{:.2f}.pth'.format(epoch, loss_epoch,
                                                                                               acc))
                torch.save(self.optimizer.state_dict(),
                           'model_weights/optimizer_epoch-{}_loss-{:.5f}_valacc-{:.2f}.pth'.format(epoch, loss_epoch,
                                                                                               acc))

        if verbose:
            print("\n\n\tDone!!!")

        if save_best_only:
            torch.save(net_best_weights,
                       'model_weights/model_epoch-{}_loss-{:.5f}_valacc-{:.2f}.pth'.format(best_epoch, best_loss, best_acc))
            torch.save(opt_best_weights,
                       'model_weights/optimizer_epoch-{}_loss-{:.5f}_valacc-{:.2f}.pth'.format(best_epoch, best_loss,
                                                                                               best_acc))

    def evaluate(self, data_loader):
        self.network.eval()

        with torch.no_grad():
            targets = []
            preds = []
            loss_batch = 0.
            for data, target in data_loader:
                output = self.network(data.to(self.device))

                loss = F.nll_loss(output, target.to(self.device))
                loss_batch += loss.data.cpu().item()

                pred = output.data.max(1, keepdim=True)[1]

                preds.extend(pred.data.cpu().tolist())
                targets.extend(target.data.cpu().tolist())

        cr = classification_report(targets, preds)
        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average='weighted')
        prec = precision_score(targets, preds, average='weighted')
        rec = recall_score(targets, preds, average='weighted')
        loss = loss_batch / len(data_loader)

        return loss, acc, prec, rec, f1, cr

    def predict(self, data_loader):
        self.network.eval()

        with torch.no_grad():
            preds = []
            for data, _ in data_loader:
                output = self.network(data.to(self.device))
                pred = output.data.max(1, keepdim=True)[1].data.cpu().item()

                preds.extend(pred)

        return preds

    def load_weights(self, model_name, optimizer_name, continue_trainning=True):
        pretrained_dict = torch.load(model_name)
        model_dict = self.network.state_dict()
        model_dict.update(pretrained_dict)
        self.network.load_state_dict(model_dict)

        if continue_trainning:
            pretrained_dict = torch.load(optimizer_name)
            optimizer_dict = self.optmizer.state_dict()
            optimizer_dict.update(pretrained_dict)
            self.optmizer.load_state_dict(model_dict)
