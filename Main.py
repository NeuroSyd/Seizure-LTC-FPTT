import sys
import argparse
import time
import logging
import numpy as np
import torch.optim as optim
from sklearn import metrics
from sklearn.metrics import roc_auc_score, recall_score, precision_score, precision_recall_curve, auc, roc_curve, confusion_matrix
from spikingjelly import visualizing
import matplotlib.pyplot as plt
from cnn_model_LTC_model import *
from datasets import data_generator, extra_test_generator, Epilepsia_12s_STFT
import torchvision
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import seaborn as sns
# from syops import get_model_complexity_info

print(sys.version)
print("Version info.")
print(sys.version_info)

def get_stats_named_params(model):

    named_params = {}
    for name, param in model.named_parameters():
        sm, lm, dm = param.detach().clone(), 0.0 * param.detach().clone(), 0.0 * param.detach().clone()
        named_params[name] = (param, sm, lm, dm)
    return named_params

def post_optimizer_updates(named_params, args, epoch):

    alpha = args.alpha
    beta = args.beta
    rho = args.rho

    for name in named_params:
        param, sm, lm, dm = named_params[name]
        if args.debias:
            beta = (1. / (1. + epoch))
            sm.data.mul_((1.0 - beta))
            sm.data.add_(beta * param)

            rho = (1. / (1. + epoch))
            dm.data.mul_((1. - rho))
            dm.data.add_(rho * lm)
        else:
            lm.data.add_(-alpha * (param - sm))
            sm.data.mul_((1.0 - beta))
            sm.data.add_(beta * param - (beta / alpha) * lm)

def get_regularizer_named_params(named_params, args, _lambda=1.0):

    alpha = args.alpha
    rho = args.rho
    regularization = torch.zeros([], device=args.device)
    for name in named_params:
        param, sm, lm, dm = named_params[name]
        regularization += (rho - 1.) * torch.sum(param * lm)
        if args.debias:
            regularization += (1. - rho) * torch.sum(param * dm)
        else:
            r_p = _lambda * 0.5 * alpha * torch.sum(torch.square(param - sm))
            regularization += r_p
    return regularization

def reset_named_params(named_params, args):

    if args.debias: return
    for name in named_params:
        param, sm, lm, dm = named_params[name]
        param.data.copy_(sm.data)

def test (model, test_loader, logger):

    test_loss = 0
    predictions = []
    true_labels = []

    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        with torch.no_grad():
            model.eval()
            visua = data
            data = data.view(-1, seq_length, input_channels)
            hidden = model.init_hidden(data.size(0))
            outputs, hidden, _ = model(data, hidden)
            output = hidden[-1]
            output = (torch.sigmoid(output))  # 512,1
            test_loss += F.binary_cross_entropy(output.squeeze().float(), target.float())
            predictions.append(output.squeeze())
            true_labels.append(target.squeeze())

            spike_seq = hidden[1]  # [Batch size, filters, 28, 28]
            spike_seq = spike_seq.unsqueeze(0).repeat(args.parts, 1, 1, 1, 1)  # torch.Size([1, 128, 1, 28, 28])
            img = visua.cpu()  # torch.Size([128, 1, 28, 28])
            spike_seq = spike_seq.cpu()  # torch.Size([1, 128, 1, 28, 28])
            spike_seq1 = hidden [0]
            spike_seq1 = spike_seq1.unsqueeze(0).repeat(args.parts, 1, 1, 1, 1)
            spike_seq1 = spike_seq1.cpu()
            print (spike_seq.shape) # torch.Size([10, 128, 16, 11, 62])
            visualization_folder = 'TUH_visualization'
            os.makedirs(visualization_folder, exist_ok=True)  # Create the folder if it doesn't exist
            for i in range(target.shape[0]):
                for t in range(args.parts):

                    print(f'saving {i}-th sample with t={t}...')
                    cmap_custom = ListedColormap(['#440154', '#fde725'])  # Choose colors from the 'viridis' colormap
                    plt.imshow(spike_seq[i][t][1], cmap=cmap_custom, interpolation= 'nearest')
                    plt.colorbar(ticks=[0, 1], label='My Custom Colorbar')
                    plt.savefig(f'{visualization_folder}/spike_seq{i}_t_{t}.pdf')  # Save the figure as a JPEG
                    plt.close()

                    plt.title(f'saving {i}-th sample with t={t}...')
                    plt.imshow(spike_seq1[i][t][1], cmap='viridis')
                    plt.colorbar()
                    plt.savefig(f'{visualization_folder}/voltage_{i}_t_{t}.pdf')  # Save the figure as a JPEG
                    plt.close()

    output_1 = torch.cat(predictions, axis=0)
    target_1 = torch.cat(true_labels, axis=0)

    test_loss /= len(test_loader)
    print(test_loss)
    auroc = roc_auc_score(target_1.cpu(), output_1.cpu())
    precision, recall, thresholds = metrics.precision_recall_curve(target_1.cpu(), output_1.cpu())
    AUPRC = metrics.auc(recall, precision)

    print('AUROC', auroc)
    print('AUPRC', AUPRC)
    print ("test_loss", test_loss)
    fpr, tpr, thresholds = metrics.roc_curve(target_1.cpu(), output_1.cpu())

    for i in range(target.shape[0]):
        for t in range(args.parts):
            print(f'saving {i}-th sample with t={t}...')
            # visualizing.plot_2d_feature_map(spike_seq[t][i], 8, spike_seq.shape[2] // 8, 2, f'$S[{t}]$')
            plt.imshow(spike_seq[i][t][i], cmap='viridis')
            plt.show()
            plt.clf()
    # logger.info(
    #     '\nLeave-One-Out Cross-Validation: Average loss: {:.4f}, AUROC: {:.4f}, Recall: {:.4f}\n'.format(
    #         test_loss, auroc, recall, AUPRC))
    # sys.stdout.flush()

    return test_loss, auroc, recall

def test_Epilepsiae(model, test_loader, logger):

    test_loss = 0
    predictions = []
    true_labels = []

    folder_path = [
                    # 'Insert paths',
                   ]

    folder_path = [
        1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30
    ]

    for folders1 in folder_path:
        # patnames = os.listdir(folders1) RPA
        # print (patnames) RPA
        # year = folders1[-4:]  # Extract the year part /RPA
        auroc_final = 0
        # print (year) /RPA

        # for patname in patnames: RPA
        #     print (patname) RPA
            # test_loader = extra_test_generator (patname=patname ,batch_size=args.batch_size, year=year) / RPA
            
        test_loader = Epilepsia_12s_STFT(patname=str(folders1), batch_size=args.batch_size)
        test_loss = 0

        #TUH

    # test_loader = extra_test_generator (batch_size=args.batch_size)
    # test_loss = 0
    #batch_size1 = 0
        for data, target in test_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()

                with torch.no_grad():
                    model.eval()
                    data1 = data
                    data = data.view(-1, seq_length, input_channels)
                    hidden = model.init_hidden(data.size(0))
                    outputs, hidden, _ = model(data, hidden)
                    output = hidden[-1]
                    output = (torch.sigmoid(output)) # 512,1
                    test_loss += F.binary_cross_entropy(output.squeeze().float(), target.float())
                    predictions.append(output.squeeze())
                    true_labels.append(target.squeeze())


        output_1 = torch.cat (predictions,axis = 0)
        target_1 = torch.cat (true_labels,axis = 0)

        test_loss /= len(test_loader)
        print(test_loss)
        auroc = roc_auc_score(target_1.cpu(), output_1.cpu())
        precision, recall, thresholds = metrics.precision_recall_curve(target_1.cpu(), output_1.cpu())
        AUPRC = metrics.auc(recall,precision)

        print ('AUROC', auroc)
        print ('AUPRC', AUPRC)

        fpr, tpr, thresholds = metrics.roc_curve(target_1.cpu(), output_1.cpu())

        # Create ROC curve plot
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auroc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc='lower right')

        # Save the plot to a file
        # output_folder = './AUROClessneurons/'+ year+ '/'
        outputfold = './EpilepsiaResultsAuroc/'+ str(folders1)
        # os.makedirs(output_folder, exist_ok=True)
        # outputfold = os.path.join (output_folder + patname)

        auroc_final = auroc_final + auroc

        with open(outputfold + '.txt', "w") as file:
            file.write(str(auroc))

        plt.savefig(outputfold + '.png')  # Change the filename as needed. RPA
        plt.show()

    ###CONFUSION MATRIX

    # conf_matrix = confusion_matrix (target_1.cpu(),output_1.cpu())
    # # Create a confusion matrix plot using seaborn
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    # plt.title(f"Confusion Matrix")
    # plt.xlabel("Predicted")
    # plt.ylabel("True")
    #
    # plt.savefig('./AUROC/' + args.saveauroc + 'confusion.png')
    # plt.show()

    # logger.info(
    #     '\nLeave-One-Out Cross-Validation: Average loss: {:.4f}, AUROC: {:.4f}, Recall: {:.4f}\n'.format(
    #         test_loss, auroc, recall, AUPRC))
    # sys.stdout.flush()

        # with open(output_folder + 'FinalAuroc.txt', "w") as file:
        #     # file.write(str(auroc_final/len(patnames)))

    return test_loss, auroc, recall

def test_RPA(model, test_loader, logger):

    test_loss = 0
    predictions = []
    true_labels = []

    folder_path = [
                   #folders
                   ]

    for folders1 in folder_path:
        patnames = os.listdir(folders1)
        year = folders1[-4:]  # Extract the year part /RPA
        auroc_final = 0
        print (year)

        for patname in patnames:
            test_loader = extra_test_generator (patname=patname ,batch_size=args.batch_size, year=year)
            test_loss = 0
            cha = 0
            for data, target in test_loader:
                    if args.cuda:
                        data, target = data.cuda(), target.cuda()

                    with torch.no_grad():
                        model.eval()
                        data = data.view(-1, seq_length, input_channels)
                        hidden = model.init_hidden(data.size(0))
                        outputs, hidden, _ = model(data, hidden)
                        output = hidden[-1]
                        output = (torch.sigmoid(output)) # 512,1
                        cha = cha + args.batch_size
                        print (cha)
                        test_loss += F.binary_cross_entropy(output.squeeze().float(), target.float())
                        predictions.append(output.squeeze())
                        true_labels.append(target.squeeze())

            output_1 = torch.cat (predictions,axis = 0)
            target_1 = torch.cat (true_labels,axis = 0)

            test_loss /= len(test_loader)
            print(test_loss)
            auroc = roc_auc_score(target_1.cpu(), output_1.cpu())
            precision, recall, thresholds = metrics.precision_recall_curve(target_1.cpu(), output_1.cpu())
            AUPRC = metrics.auc(recall,precision)

            print ('AUROC', auroc)
            print ('AUPRC', AUPRC)

            fpr, tpr, thresholds = metrics.roc_curve(target_1.cpu(), output_1.cpu())

            # Create ROC curve plot
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auroc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC)')
            plt.legend(loc='lower right')

            # Save the plot to a file
            output_folder = './AUROClessNSTFTICA/'+ year+ '/'
            os.makedirs(output_folder, exist_ok=True)
            outputfold = os.path.join (output_folder + patname)

            auroc_final = auroc_final + auroc

            with open(outputfold + '.txt', "w") as file:
                file.write(str(auroc))

            plt.savefig(outputfold + '.png')  # Change the filename as needed. RPA
            # plt.show()

    with open(output_folder + 'FinalAuroc.txt', "w") as file:
        file.write(str(auroc_final/len(patnames)))

    return test_loss, auroc, recall

def train(epoch, args, train_loader, permute, n_classes, model, named_params, logger):

    global steps
    global estimate_class_distribution

    batch_size = args.batch_size
    alpha = args.alpha
    beta = args.beta

    PARTS = args.parts
    train_loss = 0
    total_clf_loss = 0
    total_regularizaton_loss = 0
    model.train()

    i = 1

    totalfr = 0

    for batch_idx, (data, target) in enumerate(train_loader):

        if args.cuda: data, target = data.cuda(), target.cuda()
        # print (data.shape) #(32,19,23,125),
        # print (data.dtype) #torch.float32 with TUH torch.float32. With MNIST is also torch.float32
        data = data.view(-1, seq_length, input_channels) # print (data.shape) # (32,2875,19)
        B = target.size()[0] #batch size
        step = model.network.step
        Delta = torch.zeros(B, dtype=data.dtype, device=data.device)

        _PARTS = PARTS

        for p in range(_PARTS):
            if p == 0:
                h = model.init_hidden(data.size(0))
            else:
                h = tuple(v.detach() for v in h)

            o, h, hs = model.network.forward(data, h)
            output = h[-1]
            optimizer.zero_grad()

            clf_loss = (p + 1) / (_PARTS) * F.binary_cross_entropy_with_logits(output.squeeze(-1).float(), target.float())

            regularizer = get_regularizer_named_params (named_params, args, _lambda=1.0)
            loss = clf_loss  + regularizer

            loss.backward()

            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            optimizer.step()
            post_optimizer_updates(named_params, args, epoch)

            train_loss += loss.item()
            total_clf_loss += clf_loss.item()
            total_regularizaton_loss += regularizer  # .item()

        steps += seq_length

        if batch_idx > 0:
            message = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tlr: {:.5f}\tLoss: {:.5f}' \
                      ' \tClf: {:.5f} \tReg: {:.5f} \tFr: {:.5f}\tSteps: {}'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), lr, train_loss,
                total_clf_loss, total_regularizaton_loss, model.network.fr, steps)

            print(message, end='\r', flush=True)

            totalfr = totalfr + model.network.fr
            print ('totalfiringrate: ', totalfr)

            train_loss = 0
            total_clf_loss = 0
            total_regularizaton_loss = 0

            sys.stdout.flush()

parser = argparse.ArgumentParser(description='Sequential Decision Making..')

parser.add_argument('--filepath', default="Visualization", type=str)
parser.add_argument('--alpha', type=float, default=.1, help='Alpha')
parser.add_argument('--beta', type=float, default=0.5, help='Beta')
parser.add_argument('--rho', type=float, default=0.0, help='Rho')
parser.add_argument('--lmbda', type=float, default=2.0, help='Lambda')
parser.add_argument('--debias', action='store_true', help='FedDyn debias algorithm')
parser.add_argument('--K', type=int, default=1, help='Number of iterations for debias algorithm')
parser.add_argument('--nhid', type=int, default=100,
                    help='number of hidden units per layer')
parser.add_argument('--lr', type=float, default=5e-3,
                    help='initial learning rate (default: 4e-3)')
parser.add_argument('--clip', type=float, default=1.,  # 0.5,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit (default: 200)')
parser.add_argument('--parts', type=int, default=10,
                    help='Parts to split the sequential input into (default: 10)')
parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.05,
                    help='output locked dropout (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.2,
                    help='input locked dropout (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.1,
                    help='dropout applied to weights (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.2,
                    help='dropout applied to hidden layers (0 = no dropout)')
parser.add_argument('--wnorm', action='store_false',
                    help='use weight normalization (default: True)')
parser.add_argument('--temporalwdrop', action='store_false',
                    help='only drop the temporal weights (default: True)')
parser.add_argument('--wdecay', type=float, default=0.0,
                    help='weight decay')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--log-interval', type=int, default=2 * 8, metavar='N',
                    help='report interval')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use')
parser.add_argument('--when', nargs='+', type=int, default=[10, 20, 40, 80, 120, 150],
                    help='When to decay the learning rate')
parser.add_argument('--load', type=str,
                    help='path to load the model')
parser.add_argument('--save', type=str, default='./models/',
                    help='path to load the model')
parser.add_argument('--per_ex_stats', action='store_true',
                    help='Use per example stats to compute the KL loss (default: False)')
parser.add_argument('--permute', action='store_true',
                    help='use permuted dataset (default: False)')
parser.add_argument('--dataset', type=str, default='TUH',
                    help='dataset to use')
parser.add_argument('--dataroot', type=str,
                    default='./data/',
                    help='root location of the dataset')
parser.add_argument('--save_es',type=str, default='./logs/saved/',
                    help='save the images')
parser.add_argument('--saveauroc', type=str, default='',help='')

args = parser.parse_args()
args.cuda = True

exp_name = args.dataset + '-nhid-' + str(args.nhid) + '-parts-' + str(args.parts) + '-optim-' + args.optim
exp_name += '-B-' + str(args.batch_size) + '-E-' + str(args.epochs) + '-K-' + str(args.K)
exp_name += '-alpha-' + str(args.alpha) + '-beta-' + str(args.beta)

if args.permute:
    exp_name += '-perm-' + str(args.permute)
if args.per_ex_stats:
    exp_name += '-per-ex-stats-'
if args.debias:
    exp_name += '-debias-'

prefix = args.save + exp_name + args.filepath

logger = logging.getLogger('trainer')

file_log_handler = logging.FileHandler('./logs/logfile-' + exp_name + args.filepath + '.log' )
logger.addHandler(file_log_handler)

stderr_log_handler = logging.StreamHandler()
logger.addHandler(stderr_log_handler)

# nice output format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_log_handler.setFormatter(formatter)
stderr_log_handler.setFormatter(formatter)

logger.setLevel('DEBUG')

logger.info('Args: {}'.format(args))
logger.info('Exp_name = ' + exp_name)
logger.info('Prefix = ' + prefix)

torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print (torch.__version__)
args.device = device

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.manual_seed(args.seed)

steps = 0

if args.dataset in ['CIFAR-10', 'MNIST-10', 'FMNIST','TUH']:

    train_loader, test_loader, seq_length, input_channels, n_classes = data_generator(args.dataset,
                                                                                      batch_size=args.batch_size,
                                                                                      dataroot=args.dataroot,
                                                                                      shuffle=(not args.per_ex_stats))
    permute = torch.Tensor(
        np.random.permutation(seq_length).astype(np.float64)).long()  # Use only if args.permute is True

    estimate_class_distribution = torch.zeros(n_classes, args.parts, n_classes, dtype=torch.float)

    estimatedDistribution = None

    if args.per_ex_stats:
        estimatedDistribution = torch.zeros(len(train_loader) * args.batch_size, args.parts, n_classes,
                                            dtype=torch.float)
else:
    logger.info('Unknown dataset.. customize the routines to include the train/test loop.')
    exit(1)

optimizer = None
lr = args.lr

model = SeqModel(ninp=seq_length,
                 nhid=args.nhid,
                 nout=n_classes,
                 dropout=args.dropout,
                 dropouti=args.dropouti,
                 dropouth=args.dropouth,
                 wdrop=args.wdrop,
                 temporalwdrop=args.temporalwdrop,
                 wnorm=args.wnorm,
                 n_timesteps=seq_length,
                 parts=args.parts)

print (model)
total_params = count_parameters(model)
model.print_layer_shapes()

if args.cuda:
    permute = permute.cuda()

if len(args.load) > 0:
    logger.info("Loaded model\n")
    model_ckp = torch.load(args.load)
    model.load_state_dict(model_ckp['state_dict'])
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr, weight_decay=args.wdecay)
    optimizer.load_state_dict(model_ckp['optimizer'])
    print('best auroc of loaded model: ', model_ckp['AUROC'])

    args.dataset = "RPA"
    if args.dataset == 'TUH':
        test_loss, auroc, recall = test(model, test_loader, logger)

    if args.dataset == "RPA":
        test_loss, auroc, recall = test_RPA(model, test_loader, logger)

    if args.dataset == "Epilepsiae":
        test_loss, auroc, recall = test_Epilepsiae(model, test_loader, logger)

    sys.exit()

if args.cuda:
    model.cuda()

if optimizer is None:

    optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr, weight_decay=args.wdecay)
    if args.optim == 'SGD':
        optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr, momentum=0.9, weight_decay=args.wdecay)

# add cos scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

logger.info('Optimizer = ' + str(optimizer))
logger.info('Model total parameters: {}'.format(total_params))

all_test_losses = []
epochs = args.epochs
best_auroc = 0.0
best_val_loss = None
first_update = False
named_params = get_stats_named_params(model)

for epoch in range(1, epochs + 1):

    start = time.time()

    if args.dataset in ['CIFAR-10', 'MNIST-10', 'FMNIST','TUH']:

        train(epoch, args, train_loader, permute, n_classes, model, named_params, logger)
        reset_named_params(named_params, args)
        results_dict = {}

        test_loss, auroc, recall = test(model, test_loader, logger)

        if args.dataset == "TUH":

            results_dict["epoch"] = epoch + 1
            results_dict["test_loss"]= test_loss
            results_dict["AUROC"] = auroc

            logger.info('time taken = ' + str(time.time() - start))

            scheduler.step()
            print('Learning rate: ', scheduler.get_lr())

            print (auroc)
            print (best_auroc)

            is_best = auroc > best_auroc
            best_auroc = max(auroc, best_auroc)

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'AUROC': best_auroc,
                'optimizer': optimizer.state_dict(),
            }, is_best, prefix=prefix)

            file_path = 'results/' + exp_name + args.filepath + '.npy'

            with open(file_path, "a") as f:
                f.write(f"{results_dict}")
                f.write("\n")

        all_test_losses.append(test_loss)



