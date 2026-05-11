# This implementation is based on https://github.com/KarolisMart/DropGNN/blob/main/gin-graph_classification.py which was basied on https://github.com/weihua916/powerful-gnns and https://github.com/chrsmrrs/k-gnn/tree/master/examples
import os
import os.path as osp
import numpy as np
import time
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.loader.dataloader import Collater
from sklearn.model_selection import StratifiedKFold

from util import cos_anneal, get_cosine_schedule_with_warmup, LinkPredictionDataset, set_seed
from model import LinkPredictionAgentNet, add_model_args

from sklearn.metrics import roc_auc_score, f1_score
from torch.utils.data import Subset

import pickle

torch.set_printoptions(profile="full")

def main(args, cluster=None):
    print(args, flush=True)

    BATCH = args.batch_size

    if 'KarateLink' in args.dataset:
        path = os.path.join(args.data_dir, 'KarateLink.pkl')
    else:
        path = os.path.join(args.data_dir, 'GePhil.pkl')
    
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, LinkPredictionDataset):
        dataset = obj
    else:
        dataset = LinkPredictionDataset(obj, k_hop=args.k_hop)
        
    print(dataset)

    def separate_data(dataset_len, seed=42):
        labels = [data.y.item() for data in dataset]  # extract labels (0 or 1)

        # Use same splitting/10-fold as GIN paper
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle = True, random_state = seed)
        idx_list = []
        for idx in skf.split(np.zeros(dataset_len), labels):
            idx_list.append(idx)
        return idx_list

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model = LinkPredictionAgentNet(num_features=dataset.num_features, hidden_units=args.hidden_units, num_out_classes=1, dropout=args.dropout, num_steps=args.num_steps,
                    num_agents=args.num_agents, reduce=args.reduce, node_readout=args.node_readout, use_step_readout_lin=args.use_step_readout_lin,
                    num_pos_attention_heads=args.num_pos_attention_heads, readout_mlp=args.readout_mlp, self_loops=args.self_loops, post_ln=args.post_ln,
                    attn_dropout=args.attn_dropout, no_time_cond=args.no_time_cond, mlp_width_mult=args.mlp_width_mult, activation_function=args.activation_function,
                    negative_slope=args.negative_slope, input_mlp=args.input_mlp, attn_width_mult=args.attn_width_mult,
                    random_agent=args.random_agent, test_argmax=args.test_argmax, global_agent_pool=args.global_agent_pool, agent_global_extra=args.agent_global_extra,
                    basic_global_agent=args.basic_global_agent, basic_agent=args.basic_agent, bias_attention=args.bias_attention, visited_decay=args.visited_decay,
                    sparse_conv=args.sparse_conv, mean_pool_only=args.mean_pool_only, edge_negative_slope=args.edge_negative_slope,
                    final_readout_only=args.final_readout_only)

    model = model.to(device)

    def train(epoch, loader, optimizer):
        model.train()
        loss_all = 0
        n = 0
        correct = 0

        for data in loader:
            data = data.to(device)
            x = data.x
            edge_index = data.edge_index
            edge_attr = getattr(data, 'edge_attr', None)
            if edge_attr is not None:
                edge_attr = edge_attr.to(device)

            batch = data.batch
            # node_pair has shape [batch_size, 2] if collated properly
            node_pair = data.node_pair
            labels = data.y

            optimizer.zero_grad()
            logits, _ = model(x=x, edge_index=edge_index, batch=batch, node_pair=node_pair, edge_feat=edge_attr)
            logits = logits.view(-1)
            labels = labels.float()  # BCE expects float labels
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()

            optimizer.step()
            loss_all += loss.item() * labels.size(0)
            
            pred = (torch.sigmoid(logits) > args.threshold).long()
            correct += pred.eq(labels.long()).sum().item()
            n += labels.size(0)

        return loss_all / n, correct / n

    def val(loader):
        model.eval()
        with torch.no_grad():
            loss_all = 0
            for data in loader:
                data = data.to(device)
                x = data.x
                edge_index = data.edge_index
                edge_attr = getattr(data, 'edge_attr', None)
                if edge_attr is not None:
                    edge_attr = edge_attr.to(device)

                batch = data.batch
                # node_pair has shape [batch_size, 2] if collated properly
                node_pair = data.node_pair
                labels = data.y

                logits, _ = model(x=x, edge_index=edge_index, batch=batch, node_pair=node_pair, edge_feat=edge_attr)
                loss_all += F.binary_cross_entropy_with_logits(logits, labels).item()

        return loss_all / len(loader.dataset)

    def test(loader):
        model.eval()
        with torch.no_grad():
            correct = 0
            all_paths = list()
            labels_all = []
            probs_all = []
            preds_all = []

            for data in loader:
       
                data = data.to(device)
                x = data.x
                edge_index = data.edge_index
                edge_attr = getattr(data, 'edge_attr', None)
                if edge_attr is not None:
                    edge_attr = edge_attr.to(device)

                batch = data.batch
                # node_pair has shape [batch_size, 2] if collated properly
                node_pair = data.node_pair
                labels = data.y
                logits, paths = model(x=x, edge_index=edge_index, batch=batch, node_pair=node_pair, edge_feat=edge_attr, return_paths=True)
                logits = logits.view(-1)
                labels = labels.float()  # BCE expects float labels

                probs = torch.sigmoid(logits)
                preds = (probs > args.threshold).long()
                correct += preds.eq(labels.long()).sum().item()
                
                all_paths.extend(paths)
                probs_all.extend(probs)
                preds_all.extend(preds)
                labels_all.extend(labels)
            
            auc = roc_auc_score(labels_all, probs_all)
            f1 = f1_score(labels_all, preds_all)
            print(f"AUC: {auc:.4f}, F1: {f1:.4f}")

        return correct / len(loader.dataset), all_paths, auc, f1

    acc_all = []
    auc_all = []
    f1_all = []
    splits = separate_data(len(dataset), seed=0)
    print(model.__class__.__name__)
    for i, (train_idx, test_idx) in enumerate(splits):
        model.reset_parameters()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup, args.epochs, min_lr_mult=args.min_lr_mult)

        test_dataset = Subset(dataset, test_idx.tolist())
        train_dataset = Subset(dataset, train_idx.tolist())

        test_loader = DataLoader(test_dataset, batch_size=BATCH)
        train_loader = DataLoader(train_dataset, sampler=torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=int(len(train_dataset)*args.iters_per_epoch/(len(train_dataset)/BATCH))), batch_size=BATCH, drop_last=False, collate_fn=Collater(train_dataset, follow_batch=[],exclude_keys=[]))	# GIN like epochs/batches - they do 50 radom batches per epoch

        print('---------------- Split {} ----------------'.format(i), flush=True)
        
        test_acc = 0
        acc_temp = []
        auc_temp = []
        f1_temp = []
        for epoch in range(1, args.epochs+1):
            if args.verbose or epoch == 350:
                start = time.time()
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(0)
            lr = scheduler.optimizer.param_groups[0]['lr']
            if args.gumbel_warmup < 0:
                gumbel_warmup = args.warmup
            else:
                gumbel_warmup = args.gumbel_warmup
            model.temp = cos_anneal(gumbel_warmup, gumbel_warmup + args.gumbel_decay_epochs, args.gumbel_temp, args.gumbel_min_temp, epoch)
            train_loss, train_acc = train(epoch, train_loader, optimizer)
            scheduler.step()
            test_acc, paths, auc, f1 = test(test_loader)
            if args.verbose or epoch == 350:
                print('Epoch: {:03d}, LR: {:.7f}, Gumbel Temp: {:.4f}, Train Loss: {:.7f}, Train Acc: {:.4f}, Test Acc: {:.4f}, Time: {:.4f}, Mem: {:.3f}, Cached: {:.3f}, Steps: {:02d}'.format(epoch, lr, model.temp, train_loss, train_acc, test_acc, time.time() - start, torch.cuda.max_memory_allocated()/1024.0**3, torch.cuda.max_memory_reserved()/1024.0**3, len(train_loader)), flush=True)
            acc_temp.append(test_acc)
            auc_temp.append(auc)
            f1_temp.append(f1)
        acc_all.append(torch.tensor(acc_temp))
        auc_all.append(torch.tensor(auc_temp))
        f1_all.append(torch.tensor(f1_temp))
    acc_all = torch.stack(acc_all)
    auc_all = torch.stack(auc_all)
    f1_all = torch.stack(f1_all)

    # acc_mean = acc_all.mean(dim=0)
    # best_epoch = acc_mean.argmax().item()
    best_epoch = f1_all.mean(dim=0).argmax().item() 
    print('---------------- Final Epoch Result ----------------')
    print('Accuracy at last epoch - Mean: {:7f}, Std: {:7f}'.format(acc_all[:,-1].mean(), acc_all[:,-1].std()))
    print('---------------- Best Epoch (by F1): {} ----------------'.format(best_epoch))
    print('F1 at best epoch - Mean: {:7f}, Std: {:7f}'.format(f1_all[:,best_epoch].mean(), f1_all[:,best_epoch].std()), flush=True)
    
    return {
    "f1": f1_all[:, best_epoch].mean().item(),
    "auc": auc_all[:, best_epoch].mean().item(),
    "acc": acc_all[:, best_epoch].mean().item()
    }

# We use Optuna for best parameter search
import optuna
from optuna.trial import Trial

def objective(trial: Trial):
    parser = add_model_args(None, hyper=False)

    # Suggest hyperparameters
    args = parser.parse_args([])  # Empty because we'll override manually
    args.threshold = trial.suggest_float('threshold', 0.1, 0.9)
    args.num_agents = trial.suggest_categorical('num_agents', [2, 4, 6])
    args.k_hop = trial.suggest_categorical('k_hop', [3, 4])
    args.batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    args.hidden_units = trial.suggest_categorical('hidden_units', [4, 8, 16, 32])
    args.num_steps = trial.suggest_categorical('num_steps', [3, 6, 9])
    args.lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    args.dropout = trial.suggest_float('dropout', 0.0, 0.5)

    # Optional fixed settings
    args.epochs = 200
    args.verbose = True
    args.dataset = 'KarateLink'
    args.n_splits = 5
    _here = os.path.dirname(os.path.abspath(__file__))
    args.data_dir = os.path.join(_here, '..', 'out', 'datasets_splits')

    # Run the model and return best validation f1 score
    return main(args)['f1']  

# --- Use Optuna to run the search ---
def run_optuna():
    study = optuna.create_study(direction='maximize', study_name='AgentNet-KarateClub-tuning')
    study.optimize(objective, n_trials=3, timeout=36000)
    
    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

if __name__ == '__main__':
    _here = os.path.dirname(os.path.abspath(__file__))

    parser = add_model_args(None, hyper=False)
    parser.add_argument('--hyper', action='store_true', default=False,
                        help="Run Optuna hyperparameter search instead of training")
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(_here, '..', 'out', 'datasets_splits'),
                        help="Directory containing KarateLink.pkl and GePhil.pkl")

    args = parser.parse_args()

    if args.hyper:
        run_optuna()
    else:
        main(args)

    print('Finished', flush=True)