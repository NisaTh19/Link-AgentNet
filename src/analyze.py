#!/usr/bin/env python3
"""
Explainability analysis: learned agent policy vs. random-walk baseline.

Loads a trained LinkPredictionAgentNet from checkpoint, re-runs inference
with both the learned policy and a random-walk policy, then computes per-node
visitation energy, node ablation importance (logit drop on masking), and
weighted Jaccard similarity between the two walk sets.

Results are saved to <output-dir>/learn_vs_rw.csv. Summary statistics and
Wilcoxon signed-rank tests are printed to stdout.

Usage
-----
  cd src/

  # KarateLink (uses pre-computed checkpoints from out/):
  python analyze.py --dataset KarateLink

  # GePhil (dataset defaults are handled automatically):
  python analyze.py --dataset GePhil

Prerequisites
-------------
  Unzip the pre-computed checkpoints and outputs before running:
    cd out/
    unzip KarateLink_checkpoint.zip
    unzip KarateLink_output.zip
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from model import LinkPredictionAgentNet, add_model_args
from util import LinkPredictionDataset, set_seed


# ---------------------------------------------------------------------------
# Path / visit helpers
# ---------------------------------------------------------------------------

def freq_from_paths(paths_dict, data):
    """Normalized per-node visitation frequency over all steps and agents."""
    step_tensors = [pos for step, pos in paths_dict.items() if step > 0]
    positions = torch.cat(step_tensors)
    T = len(step_tensors)
    num_agents = len(paths_dict[0])
    num_nodes = int(data.edge_index.max()) + 1
    visits = torch.bincount(positions.cpu(), minlength=num_nodes)
    return visits.float() / (num_agents * T)


def degrees_within_subgraph(data):
    edge_index = data.edge_index
    num_nodes = int(edge_index.max()) + 1
    in_deg = torch.bincount(edge_index[1], minlength=num_nodes)
    out_deg = torch.bincount(edge_index[0], minlength=num_nodes)
    tot = in_deg + out_deg
    return {int(i): (int(i_d), int(o_d), int(t_d))
            for i, (i_d, o_d, t_d) in enumerate(zip(in_deg, out_deg, tot))}



# ---------------------------------------------------------------------------
# Model builder (DRY helper so we don't repeat 20 kwargs three times)
# ---------------------------------------------------------------------------

def _build_model(args, num_features, random_agent):
    return LinkPredictionAgentNet(
        num_features=num_features,
        hidden_units=args.hidden_units,
        num_out_classes=1,
        dropout=args.dropout,
        num_steps=args.num_steps,
        num_agents=args.num_agents,
        reduce=args.reduce,
        node_readout=args.node_readout,
        use_step_readout_lin=args.use_step_readout_lin,
        num_pos_attention_heads=args.num_pos_attention_heads,
        readout_mlp=args.readout_mlp,
        self_loops=args.self_loops,
        post_ln=args.post_ln,
        attn_dropout=args.attn_dropout,
        no_time_cond=args.no_time_cond,
        mlp_width_mult=args.mlp_width_mult,
        activation_function=args.activation_function,
        negative_slope=args.negative_slope,
        input_mlp=args.input_mlp,
        attn_width_mult=args.attn_width_mult,
        random_agent=random_agent,
        test_argmax=args.test_argmax,
        global_agent_pool=args.global_agent_pool,
        agent_global_extra=args.agent_global_extra,
        basic_global_agent=args.basic_global_agent,
        basic_agent=args.basic_agent,
        bias_attention=args.bias_attention,
        visited_decay=args.visited_decay,
        sparse_conv=args.sparse_conv,
        mean_pool_only=args.mean_pool_only,
        edge_negative_slope=args.edge_negative_slope,
        final_readout_only=args.final_readout_only,
        num_edge_features=args.num_edge_features,
    )


# ---------------------------------------------------------------------------
# Core XAI loop
# ---------------------------------------------------------------------------

def run_xai(args, dataset, splits, ckpt_file, only_split):
    """
    ckpt_file  : path to a single .pt checkpoint file.
    only_split : index of the split whose test set to analyse.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    readout_type = getattr(args, 'readout_type', 'all_steps')
    if readout_type == 'final_only':
        args.final_readout_only = True
        args.use_step_readout_lin = False
        args.readout_mlp = False
    elif readout_type == 'all_steps_linear':
        args.final_readout_only = False
        args.use_step_readout_lin = True
        args.readout_mlp = False
    elif readout_type == 'all_steps_mlp':
        args.final_readout_only = False
        args.use_step_readout_lin = False
        args.readout_mlp = True
    else:
        args.final_readout_only = False
        args.use_step_readout_lin = False
        args.readout_mlp = False

    rows = []

    for split_id, (_, test_idx) in enumerate(splits):
        if split_id != only_split:
            continue

        print(f"\n=== Split {split_id}  (checkpoint: {os.path.basename(ckpt_file)}) ===")
        base_model = _build_model(args, dataset.num_features, random_agent=False)
        base_model = base_model.to(device)
        ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
        base_model.load_state_dict(ckpt['model_state_dict'], strict=True)

        model_learn = _build_model(args, dataset.num_features, random_agent=False)
        model_learn.load_state_dict(base_model.state_dict())
        model_learn = model_learn.to(device).eval()

        model_rw = _build_model(args, dataset.num_features, random_agent=True)
        model_rw.load_state_dict(base_model.state_dict())
        model_rw = model_rw.to(device).eval()

        loader = DataLoader(Subset(dataset, test_idx.tolist()), batch_size=1, shuffle=False)

        probs_all_learn, probs_all_rw = [], []
        preds_all_learn, preds_all_rw = [], []
        labels_all = []
        n = correct_learn = correct_rw = 0

        for local_i, data in enumerate(loader):
            data = data.to(device)
            node_pair = data.node_pair
            edge_attr = getattr(data, "edge_attr", None)

            with torch.no_grad():
                logit_learn, paths_learn = model_learn(
                    x=data.x, edge_index=data.edge_index, batch=data.batch,
                    node_pair=node_pair, edge_feat=edge_attr, return_paths=True)
                logit_rw, paths_rw = model_rw(
                    x=data.x, edge_index=data.edge_index, batch=data.batch,
                    node_pair=node_pair, edge_feat=edge_attr, return_paths=True)

            p_learn = freq_from_paths(paths_learn, data)
            p_rw = freq_from_paths(paths_rw, data)
            energy_learn = p_learn.pow(2)
            energy_rw = p_rw.pow(2)

            numer = torch.minimum(p_learn, p_rw).sum()
            denom = torch.maximum(p_learn, p_rw).sum()
            jaccard_w = (numer / denom).item()

            labels = data.y.float()
            n += labels.numel()

            probs_learn = torch.sigmoid(logit_learn)
            preds_learn = (probs_learn > args.threshold).long()
            correct_learn += preds_learn.eq(labels.long()).sum().item()
            probs_all_learn.extend(probs_learn.unsqueeze(0))
            preds_all_learn.extend(preds_learn.unsqueeze(0))
            labels_all.append(labels.unsqueeze(0))

            probs_rw = torch.sigmoid(logit_rw)
            preds_rw = (probs_rw > args.threshold).long()
            correct_rw += preds_rw.eq(labels.long()).sum().item()
            probs_all_rw.extend(probs_rw.unsqueeze(0))
            preds_all_rw.extend(preds_rw.unsqueeze(0))

            degree_dict = degrees_within_subgraph(data)

            # Ablation importance: |logit change| when masking each visited node
            delta_learn = []
            for k in range(len(p_learn)):
                if p_learn[k] == 0:
                    delta_learn.append(0.0)
                    continue
                x_mask = data.x.clone()
                x_mask[k] = 0
                l_masked, _ = model_learn(x=x_mask, edge_index=data.edge_index,
                                          batch=data.batch, node_pair=node_pair,
                                          edge_feat=edge_attr, return_paths=False)
                delta_learn.append(float(torch.abs(logit_learn - l_masked)))

            delta_rw = []
            for k in range(len(p_learn)):
                if p_rw[k] == 0:
                    delta_rw.append(0.0)
                    continue
                x_mask = data.x.clone()
                x_mask[k] = 0
                l_masked, _ = model_rw(x=x_mask, edge_index=data.edge_index,
                                       batch=data.batch, node_pair=node_pair,
                                       edge_feat=edge_attr, return_paths=False)
                delta_rw.append(float(torch.abs(logit_rw - l_masked)))

            mask_vis_learn = energy_learn > 0
            ev_l = energy_learn[mask_vis_learn].cpu().numpy()
            dv_l = np.array(delta_learn)[mask_vis_learn.cpu().numpy()]
            rho_learn = float('nan') if len(np.unique(ev_l)) < 2 or len(np.unique(dv_l)) < 2 \
                else spearmanr(ev_l, dv_l).correlation

            mask_vis_rw = energy_rw > 0
            ev_r = energy_rw[mask_vis_rw].cpu().numpy()
            dv_r = np.array(delta_rw)[mask_vis_rw.cpu().numpy()]
            rho_rw = float('nan') if len(np.unique(ev_r)) < 2 or len(np.unique(dv_r)) < 2 \
                else spearmanr(ev_r, dv_r).correlation

            for n_id in range(len(p_learn)):
                in_d, out_d, tot_d = degree_dict.get(n_id, (0, 0, 0))
                rows.append({
                    "split": split_id,
                    "sample_idx": int(test_idx[local_i]),
                    "node_id": n_id,
                    "freq_learn": float(p_learn[n_id]),
                    "freq_rw": float(p_rw[n_id]),
                    "in_degree": in_d,
                    "out_degree": out_d,
                    "tot_degree": tot_d,
                    "energy_learn": float(energy_learn[n_id]),
                    "energy_rw": float(energy_rw[n_id]),
                    "jaccard_w": jaccard_w,
                    "rho_energy_delta_learn": rho_learn,
                    "rho_energy_delta_rw": rho_rw,
                    "global_f1_learn": None,
                    "global_auc_learn": None,
                    "global_acc_learn": None,
                    "global_f1_rw": None,
                    "global_auc_rw": None,
                    "global_acc_rw": None,
                })

        labels_np = torch.cat(labels_all, dim=0).cpu().numpy()
        probs_l_np = torch.cat(probs_all_learn, dim=0).cpu().numpy()
        preds_l_np = torch.cat(preds_all_learn, dim=0).cpu().numpy()
        probs_r_np = torch.cat(probs_all_rw, dim=0).cpu().numpy()
        preds_r_np = torch.cat(preds_all_rw, dim=0).cpu().numpy()

        f1_l = f1_score(labels_np, preds_l_np)
        auc_l = roc_auc_score(labels_np, probs_l_np)
        acc_l = correct_learn / n
        f1_r = f1_score(labels_np, preds_r_np)
        auc_r = roc_auc_score(labels_np, probs_r_np)
        acc_r = correct_rw / n

        for row in rows:
            if row["split"] == split_id:
                row.update({"global_f1_learn": f1_l, "global_auc_learn": auc_l,
                             "global_acc_learn": acc_l, "global_f1_rw": f1_r,
                             "global_auc_rw": auc_r, "global_acc_rw": acc_r})

    df = pd.DataFrame(rows)
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, "learn_vs_rw.csv")
    df.to_csv(out_file, index=False)
    print(f"\nSaved {len(df)} rows → {out_file}")
    return df


def print_summary(df):
    sample_summary = (
        df.groupby(["split", "sample_idx"])
          .agg(
              jaccard_w=("jaccard_w", "first"),
              rho_learn=("rho_energy_delta_learn", "first"),
              rho_rw=("rho_energy_delta_rw", "first"),
              acc_learn=("global_acc_learn", "first"),
              acc_rw=("global_acc_rw", "first"),
              auc_learn=("global_auc_learn", "first"),
              auc_rw=("global_auc_rw", "first"),
              f1_learn=("global_f1_learn", "first"),
              f1_rw=("global_f1_rw", "first"),
          )
    )
    sample_summary["dF1"] = sample_summary.f1_learn - sample_summary.f1_rw
    sample_summary["dAuc"] = sample_summary.auc_learn - sample_summary.auc_rw
    sample_summary["dAcc"] = sample_summary.acc_learn - sample_summary.acc_rw

    print("\n=== Sample-level XAI summary ===")
    print(sample_summary.describe())

    print("\n=== Wilcoxon signed-rank tests (learned vs. random walk) ===")
    for metric in ["f1", "auc", "acc"]:
        p = wilcoxon(sample_summary[f"{metric}_learn"], sample_summary[f"{metric}_rw"]).pvalue
        print(f"  Δ{metric.upper():3s}  p = {p:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SRC_DIR)

# Dataset-specific defaults (GePhil study was named with a _2 suffix during tuning)
DATASET_DEFAULTS = {
    'KarateLink': dict(
        optuna_db  = os.path.join(_REPO_ROOT, 'out', 'optuna_db', 'optuna_shared_KarateLink.db'),
        study_name = 'AgentNet-Tuning-KarateLink',
        num_edge_features=0, self_loops=False,
    ),
    'GePhil': dict(
        optuna_db  = os.path.join(_REPO_ROOT, 'out', 'optuna_db', 'optuna_shared_GePhil_2.db'),
        study_name = 'AgentNet-Tuning-GePhil_2',
        num_edge_features=2, self_loops=True,  # fin_freq + shared_board_weight
    ),
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--dataset', default='KarateLink', choices=['KarateLink', 'GePhil'])
    parser.add_argument('--data-dir', default=os.path.join(_REPO_ROOT, 'out', 'datasets_splits'),
                        help='Dir containing <dataset>.pkl and <dataset>_splits.pkl')
    parser.add_argument('--checkpoint', default=None,
                        help='Path to a .pt checkpoint file '
                             '(default: out/checkpoints/<dataset>_best.pt)')
    parser.add_argument('--output-dir', default=None,
                        help='Dir for learn_vs_rw.csv output (default: out/<dataset>_output)')
    parser.add_argument('--optuna-db', default=None,
                        help='Optuna SQLite file (overrides dataset default)')
    parser.add_argument('--study-name', default=None,
                        help='Optuna study name (overrides dataset default)')
    parser.add_argument('--num-edge-features', type=int, default=None,
                        help='Number of edge features (overrides dataset default)')
    parser.add_argument('--self-loops', action='store_true', default=None)
    cli = parser.parse_args()

    ds  = cli.dataset
    cfg = DATASET_DEFAULTS[ds]

    if cli.output_dir is None:
        cli.output_dir = os.path.join(_REPO_ROOT, 'out', f'{ds}_output')
    if cli.optuna_db is None:
        cli.optuna_db = cfg['optuna_db']
    if cli.study_name is None:
        cli.study_name = cfg['study_name']
    if cli.num_edge_features is None:
        cli.num_edge_features = cfg['num_edge_features']
    if cli.self_loops is None:
        cli.self_loops = cfg['self_loops']

    # Resolve checkpoint file
    if cli.checkpoint is None:
        cli.checkpoint = os.path.join(_REPO_ROOT, 'out', 'checkpoints', f'{ds}_best.pt')
    if not os.path.isfile(cli.checkpoint):
        print(f"Checkpoint not found: {cli.checkpoint}")
        print("Train the model first with: python link_prediction.py --dataset " + ds)
        exit(1)

    # Determine which split this checkpoint came from (for the correct test set)
    import pandas as pd
    metrics_csv = os.path.join(cli.output_dir, 'metrics_per_split_epoch.csv')
    df_m = pd.read_csv(metrics_csv)
    best_split = int(df_m.groupby('split')['test_f1'].max().idxmax())
    print(f"Using checkpoint: {cli.checkpoint}  (split {best_split} test set)")

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name=cli.study_name,
                              storage=f"sqlite:///{cli.optuna_db}")
    best_params = study.best_trial.params
    print(f"Loaded best params from '{cli.study_name}':")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    model_parser = add_model_args(None, hyper=False)
    args = model_parser.parse_args([])
    args.threshold = best_params.get('threshold', 0.5)
    args.num_agents = best_params.get('num_agents', 4)
    args.batch_size = best_params.get('batch_size', 32)
    args.hidden_units = best_params.get('hidden_units', 64)
    args.num_steps = best_params.get('num_steps', 8)
    args.lr = best_params.get('lr', 0.001)
    args.dropout = best_params.get('dropout', 0.0)
    args.num_pos_attention_heads = best_params.get('num_pos_attention_heads', 1)
    args.gumbel_temp = best_params.get('gumbel_temp', 1.0)
    args.readout_type = best_params.get('readout_type', 'all_steps')

    args.reduce = 'sum'
    args.global_agent_pool = True
    args.agent_global_extra = False
    args.basic_global_agent = True
    args.bias_attention = True
    args.self_loops = cli.self_loops
    args.num_edge_features = cli.num_edge_features
    args.output_dir = cli.output_dir

    set_seed(42)

    with open(os.path.join(cli.data_dir, f'{ds}.pkl'), 'rb') as f:
        obj = pickle.load(f)
    with open(os.path.join(cli.data_dir, f'{ds}_splits.pkl'), 'rb') as f:
        splits = pickle.load(f)

    dataset = obj if isinstance(obj, LinkPredictionDataset) else LinkPredictionDataset(obj, k_hop=args.k_hop)

    # GePhil was trained with edge features (fin_freq, shared_board_weight).
    # The pre-saved dataset has no edge_attr, so we inject them by swapping
    # the dataset's class for a subclass that overrides __getitem__.
    if args.num_edge_features > 0 and hasattr(dataset, 'G_nx'):
        from torch_geometric.utils import k_hop_subgraph as _khop
        from torch_geometric.data import Data as _Data

        G_nx = dataset.G_nx
        ei = dataset.edge_index
        _attrs = []
        for _k in range(ei.size(1)):
            _u, _v = int(ei[0, _k]), int(ei[1, _k])
            _d = G_nx[_u][_v] if G_nx.has_edge(_u, _v) else (G_nx[_v][_u] if G_nx.has_edge(_v, _u) else {})
            _attrs.append([float(_d.get('fin_freq', 0.0)), float(_d.get('shared_board_weight', 0.0))])
        dataset._edge_attr_full = torch.tensor(_attrs, dtype=torch.float)

        class _WithEdgeAttr(type(dataset)):
            def __getitem__(self, idx):
                if idx < len(self.positive_edges):
                    node_pair = torch.tensor(self.positive_edges[idx])
                    label = torch.tensor(1, dtype=torch.long)
                else:
                    node_pair = torch.tensor(self.negative_edges[idx - len(self.positive_edges)])
                    label = torch.tensor(0, dtype=torch.long)
                nodes, sub_ei, mapping, edge_mask = _khop(
                    node_idx=node_pair, num_hops=self.num_hops,
                    edge_index=self.edge_index, relabel_nodes=True,
                    num_nodes=self.num_nodes, flow='source_to_target')
                sub_ea = self._edge_attr_full[edge_mask]
                if label.item() == 1:
                    i, j = mapping[0].item(), mapping[1].item()
                    mask = ~(((sub_ei[0] == i) & (sub_ei[1] == j)) |
                             ((sub_ei[0] == j) & (sub_ei[1] == i)))
                    sub_ei = sub_ei[:, mask]
                    sub_ea = sub_ea[mask]
                    if not ((sub_ei[0] == i).any() or (sub_ei[1] == i).any()):
                        return self.__getitem__((idx + 1) % len(self))
                    if not ((sub_ei[0] == j).any() or (sub_ei[1] == j).any()):
                        return self.__getitem__((idx + 1) % len(self))
                x = self.x[nodes]
                return _Data(x=x, edge_index=sub_ei, edge_attr=sub_ea, y=label,
                             node_pair=mapping.view(1, 2),
                             batch=torch.zeros(x.size(0), dtype=torch.long))

        dataset.__class__ = _WithEdgeAttr

    df = run_xai(args, dataset, splits,
                 ckpt_file=cli.checkpoint,
                 only_split=best_split)
    print_summary(df)
