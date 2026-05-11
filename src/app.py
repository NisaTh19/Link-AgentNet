#!/usr/bin/env python3
"""
Link-AgentNet — Interactive Streamlit Explorer

Launch from the repo root or src/ directory:
    streamlit run src/app.py

Requires the pre-computed checkpoints to be unzipped first:
    cd out/
    unzip KarateLink_checkpoint.zip && unzip KarateLink_output.zip
    unzip GePhil_checkpoint.zip    && unzip GePhil_output.zip
"""

import os, sys, pickle, torch
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import streamlit as st
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import add_model_args
from util import LinkPredictionDataset, set_seed
from analyze import freq_from_paths, _build_model

# ── Constants ─────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Fixed colour per agent — consistent across all-agents and single-agent views
AGENT_COLORS = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45',
    '#469990', '#dcbeff', '#9a6324', '#800000',
]

DATASET_CFG = {
    'KarateLink': dict(
        optuna_db   = os.path.join(_ROOT, 'out', 'optuna_db', 'optuna_shared_KarateLink.db'),
        study_name  = 'AgentNet-Tuning-KarateLink',
        best_ckpt   = os.path.join(_ROOT, 'out', 'checkpoints', 'KarateLink_best.pt'),
        data_dir    = os.path.join(_ROOT, 'out', 'datasets_splits'),
        num_edge_features=0, self_loops=False,
    ),
    'GePhil': dict(
        optuna_db   = os.path.join(_ROOT, 'out', 'optuna_db', 'optuna_shared_GePhil_2.db'),
        study_name  = 'AgentNet-Tuning-GePhil_2',
        best_ckpt   = os.path.join(_ROOT, 'out', 'checkpoints', 'GePhil_best.pt'),
        data_dir    = os.path.join(_ROOT, 'out', 'datasets_splits'),
        num_edge_features=2, self_loops=True,
    ),
}

# ── Cached loaders ───────────────────────────────────────────────────────────

@st.cache_resource
def load_graph_and_dataset(ds_name):
    cfg = DATASET_CFG[ds_name]
    set_seed(42)
    with open(os.path.join(cfg['data_dir'], f'{ds_name}.pkl'), 'rb') as f:
        obj = pickle.load(f)
    dataset = obj if isinstance(obj, LinkPredictionDataset) else LinkPredictionDataset(obj, k_hop=3)
    return dataset.G_nx, dataset


@st.cache_resource
def load_args_from_optuna(ds_name):
    import optuna
    cfg = DATASET_CFG[ds_name]
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name=cfg['study_name'],
                              storage=f"sqlite:///{cfg['optuna_db']}")
    bp = study.best_trial.params

    parser = add_model_args(None, hyper=False)
    args = parser.parse_args([])
    args.threshold            = bp.get('threshold', 0.5)
    args.num_agents           = bp.get('num_agents', 4)
    args.hidden_units         = bp.get('hidden_units', 64)
    args.num_steps            = bp.get('num_steps', 8)
    args.dropout              = bp.get('dropout', 0.0)
    args.num_pos_attention_heads = bp.get('num_pos_attention_heads', 1)
    args.gumbel_temp          = bp.get('gumbel_temp', 1.0)
    args.readout_type         = bp.get('readout_type', 'all_steps')
    args.lr                   = bp.get('lr', 0.001)
    args.batch_size           = bp.get('batch_size', 32)
    args.reduce = 'sum'; args.global_agent_pool = True; args.agent_global_extra = False
    args.basic_global_agent = True; args.bias_attention = True
    args.self_loops        = cfg['self_loops']
    args.num_edge_features = cfg['num_edge_features']
    rt = args.readout_type
    args.final_readout_only   = rt == 'final_only'
    args.use_step_readout_lin = rt == 'all_steps_linear'
    args.readout_mlp          = rt == 'all_steps_mlp'
    return args


@st.cache_resource
def load_model(ds_name):
    cfg    = DATASET_CFG[ds_name]
    args   = load_args_from_optuna(ds_name)
    _, dataset = load_graph_and_dataset(ds_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = _build_model(args, dataset.num_features, random_agent=False).to(device)
    ckpt   = torch.load(cfg['best_ckpt'], map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()
    return model, args, device

# ── Data helpers ─────────────────────────────────────────────────────────────

def _build_edge_attr(G, edge_index_orig, edge_mask):
    orig = edge_index_orig[:, edge_mask]
    attrs = []
    for k in range(orig.size(1)):
        u, v = int(orig[0, k]), int(orig[1, k])
        d = G[u][v] if G.has_edge(u, v) else (G[v][u] if G.has_edge(v, u) else {})
        attrs.append([float(d.get('fin_freq', 0.0)), float(d.get('shared_board_weight', 0.0))])
    return torch.tensor(attrs, dtype=torch.float)


def pair_to_data(src, tgt, dataset, device, G=None):
    node_pair = torch.tensor([src, tgt])
    nodes, sub_ei, mapping, edge_mask = k_hop_subgraph(
        node_idx=node_pair, num_hops=dataset.num_hops,
        edge_index=dataset.edge_index, relabel_nodes=True,
        num_nodes=dataset.num_nodes, flow='source_to_target',
    )
    i, j = mapping[0].item(), mapping[1].item()
    direct = ~(((sub_ei[0] == i) & (sub_ei[1] == j)) |
               ((sub_ei[0] == j) & (sub_ei[1] == i)))
    sub_ea = None
    if G is not None and any('fin_freq' in d for _, _, d in G.edges(data=True)):
        sub_ea = _build_edge_attr(G, dataset.edge_index, edge_mask)[direct]
    sub_ei = sub_ei[:, direct]
    x = dataset.x[nodes]
    data = Data(x=x, edge_index=sub_ei, edge_attr=sub_ea, y=torch.tensor(0),
                node_pair=mapping.view(1, 2),
                batch=torch.zeros(x.size(0), dtype=torch.long))
    return data.to(device), nodes

# ── Plotly helpers ───────────────────────────────────────────────────────────

@st.cache_data
def compute_layout(_G, ds_name):
    return nx.spring_layout(_G, seed=42)


def graph_figure(G, pos, src_node=None, tgt_node=None, title=''):
    node_list = list(G.nodes())
    ex, ey = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        ex += [x0, x1, None]; ey += [y0, y1, None]
    colors = ['#e84545' if n == src_node else '#f5a623' if n == tgt_node else '#5b8dee'
              for n in node_list]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ex, y=ey, mode='lines',
                             line=dict(width=1, color='#ccc'), hoverinfo='none'))
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in node_list], y=[pos[n][1] for n in node_list],
        mode='markers+text', text=[str(n) for n in node_list],
        textposition='top center', hoverinfo='text',
        marker=dict(size=12, color=colors, line=dict(width=1, color='#fff')),
    ))
    fig.update_layout(title=title, showlegend=False, hovermode='closest',
                      margin=dict(b=5, l=5, r=5, t=35),
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      height=380)
    return fig


def _hex_to_rgba(hex_color, alpha):
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f'rgba({r},{g},{b},{alpha:.2f})'


def walk_figure(data, paths, step, node_labels, src_local, tgt_local, pos_sub, title,
                agent_id=None, group_size=None):
    """
    agent_id=None  → show all agents, one coloured path per agent.
    agent_id=k     → show only agent k, with alpha-faded path in its fixed colour.
    group_size     → first group_size agents start at source; the rest at target.
    """
    num_agents = len(paths[0])
    gs = group_size if group_size is not None else num_agents // 2
    clamped = min(step, max(paths.keys()))

    agents_to_draw = [agent_id] if agent_id is not None else list(range(num_agents))

    # Visited nodes for selected agents up to this step
    visited = set()
    for a in agents_to_draw:
        for s in range(clamped + 1):
            visited.add(int(paths[s][a]))

    show_nodes = visited | {src_local, tgt_local}

    # Cumulative visit frequency (for node shading)
    freq = torch.zeros(data.x.size(0))
    if clamped > 0:
        for a in agents_to_draw:
            for s in range(1, clamped + 1):
                freq[int(paths[s][a])] += 1
        freq = freq / (clamped * len(agents_to_draw))

    # Current positions
    current_nodes = {int(paths[clamped][a]) for a in agents_to_draw} if clamped > 0 else set()

    # Background edges (only between visible nodes)
    ei = data.edge_index.cpu()
    ex, ey = [], []
    for k in range(ei.size(1)):
        u, v = int(ei[0, k]), int(ei[1, k])
        if u in show_nodes and v in show_nodes:
            ex += [pos_sub[u][0], pos_sub[v][0], None]
            ey += [pos_sub[u][1], pos_sub[v][1], None]

    fig = go.Figure()
    if ex:
        fig.add_trace(go.Scatter(x=ex, y=ey, mode='lines',
                                 line=dict(width=0.8, color='#ddd'),
                                 hoverinfo='none', showlegend=False))

    # Draw each agent's path
    for a in agents_to_draw:
        color = AGENT_COLORS[a % len(AGENT_COLORS)]
        role  = 'src' if a < gs else 'tgt'
        agent_path = [int(paths[s][a]) for s in range(clamped + 1)]

        if agent_id is not None:
            # Single agent: alpha-faded segments (light → dark)
            for s in range(len(agent_path) - 1):
                n1, n2 = agent_path[s], agent_path[s + 1]
                alpha = 0.2 + 0.8 * (s / max(len(agent_path) - 1, 1))
                fig.add_trace(go.Scatter(
                    x=[pos_sub[n1][0], pos_sub[n2][0]],
                    y=[pos_sub[n1][1], pos_sub[n2][1]],
                    mode='lines',
                    line=dict(width=4, color=_hex_to_rgba(color, alpha)),
                    hoverinfo='none', showlegend=False,
                ))
        else:
            # All agents: one solid-colour trace per agent (legend-visible)
            path_x = [pos_sub[n][0] for n in agent_path]
            path_y = [pos_sub[n][1] for n in agent_path]
            fig.add_trace(go.Scatter(
                x=path_x, y=path_y,
                mode='lines',
                line=dict(width=2.5, color=color),
                name=f'Agent {a} ({role})',
                showlegend=True,
                hoverinfo='none',
            ))

    # Nodes (only visible ones)
    node_list = sorted(show_nodes)
    max_f = max(float(freq.max()), 1e-6)
    cmap = plt.get_cmap('Blues')
    colors_n, sizes = [], []
    for n in node_list:
        if n == src_local:
            colors_n.append('#e84545')
        elif n == tgt_local:
            colors_n.append('#f5a623')
        else:
            colors_n.append(mc.to_hex(cmap(0.15 + 0.85 * float(freq[n]) / max_f)))
        base = 18 if n in {src_local, tgt_local} else 13
        sizes.append(26 if n in current_nodes else base)

    fig.add_trace(go.Scatter(
        x=[pos_sub[n][0] for n in node_list],
        y=[pos_sub[n][1] for n in node_list],
        mode='markers+text',
        text=[str(node_labels[n]) for n in node_list],
        textposition='top center',
        hovertext=[f'Node {node_labels[n]}  freq={float(freq[n]):.3f}' for n in node_list],
        hoverinfo='text',
        marker=dict(size=sizes, color=colors_n, line=dict(width=1.5, color='#555')),
        showlegend=False,
    ))

    legend_cfg = dict(
        orientation='v', yanchor='top', y=1.0, xanchor='left', x=1.01,
        font=dict(size=10), bgcolor='rgba(255,255,255,0.8)',
    ) if agent_id is None else dict(visible=False)

    fig.update_layout(
        title=title,
        hovermode='closest',
        showlegend=(agent_id is None),
        legend=legend_cfg,
        margin=dict(b=5, l=5, r=80, t=40),  # r=80 leaves room for legend
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=390,
    )
    return fig


def bar_chart(x, y1, y2, name1, name2, title, yaxis_title):
    fig = go.Figure()
    fig.add_trace(go.Bar(name=name1, x=x, y=y1, marker_color='#5b8dee'))
    fig.add_trace(go.Bar(name=name2, x=x, y=y2, marker_color='#f5a623'))
    fig.update_layout(
        title=title, barmode='group',
        xaxis=dict(title=dict(text='Node ID', standoff=12)),
        yaxis_title=yaxis_title,
        height=320,
        margin=dict(t=30, b=55, l=50, r=10),
        legend=dict(orientation='h', yanchor='bottom', y=1.04, xanchor='right', x=1),
    )
    return fig

# ── App ───────────────────────────────────────────────────────────────────────

st.set_page_config(page_title='Link-AgentNet Explorer', layout='wide')
st.title('Link-AgentNet — Interactive Explorer')
st.caption('Visualize agent walks and explainability for link prediction on graphs.')

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header('Settings')
    ds_name  = st.selectbox('Dataset', ['KarateLink', 'GePhil'])
    show_xai = st.checkbox('Show XAI panel (slower)', value=False)
    st.markdown('---')
    st.markdown(
        '**Node colours**\n\n'
        '🔴 Source node.\n\n'
        '🟠 Target node.\n\n'
        '🔵 Visited node — shade = visit frequency.\n\n'
        '---\n\n'
        '**Agent colours** are fixed per agent ID and consistent across all views.\n\n'
        '**src** agents start at the source node.\n\n'
        '**tgt** agents start at the target node.'
    )

cfg = DATASET_CFG[ds_name]

if not os.path.isfile(cfg['best_ckpt']):
    st.error(f"Checkpoint not found: `{cfg['best_ckpt']}`")
    st.stop()

with st.spinner('Loading model and graph…'):
    G, dataset = load_graph_and_dataset(ds_name)
    model, args, device = load_model(ds_name)

group_size = args.num_agents // 2

# ── Network overview ──────────────────────────────────────────────────────────
st.subheader('Network overview')
oc1, oc2 = st.columns([3, 1])
pos_full = compute_layout(G, ds_name)

with oc2:
    st.metric('Nodes', G.number_of_nodes())
    st.metric('Edges', G.number_of_edges())
    st.metric('Avg degree', f"{2 * G.number_of_edges() / G.number_of_nodes():.1f}")
    st.metric('Density', f"{nx.density(G):.4f}")
with oc1:
    st.plotly_chart(graph_figure(G, pos_full, title=f'{ds_name} — full graph'),
                    use_container_width=True)

# ── Node pair selector ────────────────────────────────────────────────────────
st.subheader('Select a node pair to predict')
all_nodes = sorted(G.nodes())
pc1, pc2, pc3 = st.columns([2, 2, 1])
with pc1:
    src = st.selectbox('Source node', all_nodes, index=0)
with pc2:
    tgt = st.selectbox('Target node', [n for n in all_nodes if n != src], index=1)
with pc3:
    st.write(''); st.write('')
    run = st.button('Predict', type='primary', use_container_width=True)

actual = int(G.has_edge(src, tgt))

# ── Run inference → store in session_state so widgets persist across reruns ───
if run:
    with st.spinner('Running inference…'):
        data, sub_nodes = pair_to_data(src, tgt, dataset, device, G=G)
        src_local = int(data.node_pair[0, 0])
        tgt_local = int(data.node_pair[0, 1])
        node_labels = {i: int(sub_nodes[i]) for i in range(data.x.size(0))}
        edge_attr = data.edge_attr
        pos_sub = nx.spring_layout(
            nx.Graph([(int(data.edge_index[0, k]), int(data.edge_index[1, k]))
                      for k in range(data.edge_index.size(1))]),
            seed=42,
        )
        with torch.no_grad():
            logit_l, paths_l = model(x=data.x, edge_index=data.edge_index,
                                     batch=data.batch, node_pair=data.node_pair,
                                     edge_feat=edge_attr, return_paths=True)
            prob_l = float(torch.sigmoid(logit_l)); pred_l = int(prob_l > args.threshold)

            model_rw = _build_model(args, dataset.num_features, random_agent=True)
            model_rw.load_state_dict(model.state_dict())
            model_rw = model_rw.to(device).eval()
            logit_r, paths_r = model_rw(x=data.x, edge_index=data.edge_index,
                                        batch=data.batch, node_pair=data.node_pair,
                                        edge_feat=edge_attr, return_paths=True)
            prob_r = float(torch.sigmoid(logit_r)); pred_r = int(prob_r > args.threshold)

        st.session_state['pred'] = dict(
            ds_name=ds_name, src=src, tgt=tgt, actual=actual,
            data=data.cpu(),
            sub_nodes=sub_nodes.cpu(),
            edge_attr=edge_attr.cpu() if edge_attr is not None else None,
            paths_l={k: v.cpu() for k, v in paths_l.items()},
            paths_r={k: v.cpu() for k, v in paths_r.items()},
            logit_l=logit_l.cpu(), logit_r=logit_r.cpu(),
            prob_l=prob_l, prob_r=prob_r, pred_l=pred_l, pred_r=pred_r,
            node_labels=node_labels, src_local=src_local, tgt_local=tgt_local,
            pos_sub=pos_sub,
        )

# ── Show results (persists while slider / agent selector change) ───────────────
pred = st.session_state.get('pred')
if (pred is None
        or pred['ds_name'] != ds_name
        or pred['src'] != src
        or pred['tgt'] != tgt):
    st.info('Select a node pair above and click **Predict**.')
    st.stop()

p = pred

# Summary metrics
st.markdown('---')
mc1, mc2, mc3 = st.columns(3)
mc1.metric('Actual label', 'LINK ✓' if p['actual'] else 'NO LINK')
mc2.metric(f'Learned policy  {"✓" if p["pred_l"] == p["actual"] else "✗"}',
           f'{"LINK" if p["pred_l"] else "NO LINK"}  ({p["prob_l"]:.1%})')
mc3.metric(f'Random walk  {"✓" if p["pred_r"] == p["actual"] else "✗"}',
           f'{"LINK" if p["pred_r"] else "NO LINK"}  ({p["prob_r"]:.1%})')

# ── Walk controls ─────────────────────────────────────────────────────────────
st.subheader('Agent walk — subgraph view')

num_agents = args.num_agents
# Dropdown: label each agent with its role and colour swatch
agent_options = ['All agents'] + [
    f'Agent {i}  ({"src" if i < group_size else "tgt"})'
    for i in range(num_agents)
]

wc1, wc2 = st.columns([3, 2])
with wc1:
    step = st.slider('Walk step', 0, args.num_steps, args.num_steps)
with wc2:
    agent_choice = st.selectbox('Filter by agent', agent_options)
    agent_id = None if agent_choice == 'All agents' else int(agent_choice.split()[1])

fc1, fc2 = st.columns(2)
with fc1:
    st.plotly_chart(
        walk_figure(p['data'], p['paths_l'], step, p['node_labels'],
                    p['src_local'], p['tgt_local'], p['pos_sub'],
                    f'Learned policy — step {step}',
                    agent_id=agent_id, group_size=group_size),
        use_container_width=True,
    )
with fc2:
    st.plotly_chart(
        walk_figure(p['data'], p['paths_r'], step, p['node_labels'],
                    p['src_local'], p['tgt_local'], p['pos_sub'],
                    f'Random walk — step {step}',
                    agent_id=agent_id, group_size=group_size),
        use_container_width=True,
    )

if agent_id is None:
    st.caption(
        'Each agent has a fixed colour (shown in the legend). '
        '**src** agents start at the source node; **tgt** agents at the target. '
        'Blue node shade = cumulative visit frequency. Larger node = agents present at this step.'
    )
else:
    role = 'source' if agent_id < group_size else 'target'
    st.caption(
        f'Agent {agent_id} starts at the **{role}** node. '
        'Path colour fades from light (early) to dark (late). '
        'The larger node is the current position.'
    )

# ── XAI panel ─────────────────────────────────────────────────────────────────
if show_xai:
    st.subheader('XAI analysis')
    from scipy.stats import spearmanr

    data_cpu  = p['data']
    ea_cpu    = p['edge_attr']
    logit_l_c = p['logit_l']
    logit_r_c = p['logit_r']
    ea_dev    = ea_cpu.to(device) if ea_cpu is not None else None
    data_dev  = data_cpu.to(device)

    p_l = freq_from_paths(p['paths_l'], data_cpu).numpy()
    p_r = freq_from_paths(p['paths_r'], data_cpu).numpy()
    energy_l, energy_r = p_l ** 2, p_r ** 2

    model_rw2 = _build_model(args, dataset.num_features, random_agent=True)
    model_rw2.load_state_dict(model.state_dict())
    model_rw2 = model_rw2.to(device).eval()

    prog = st.progress(0, text='Computing node ablation importance…')
    delta_l, delta_r = [], []
    n_nodes = data_dev.x.size(0)
    for k in range(n_nodes):
        prog.progress((k + 1) / n_nodes, text=f'Ablating node {k + 1}/{n_nodes}…')
        x_mask = data_dev.x.clone(); x_mask[k] = 0
        with torch.no_grad():
            ll, _ = model(x=x_mask, edge_index=data_dev.edge_index,
                          batch=data_dev.batch, node_pair=data_dev.node_pair, edge_feat=ea_dev)
            lr, _ = model_rw2(x=x_mask, edge_index=data_dev.edge_index,
                               batch=data_dev.batch, node_pair=data_dev.node_pair, edge_feat=ea_dev)
        delta_l.append(float(torch.abs(logit_l_c.to(device) - ll)) if p_l[k] > 0 else 0.0)
        delta_r.append(float(torch.abs(logit_r_c.to(device) - lr)) if p_r[k] > 0 else 0.0)
    prog.empty()

    orig_ids = [p['node_labels'][i] for i in range(n_nodes)]
    numer = np.minimum(p_l, p_r).sum(); denom = np.maximum(p_l, p_r).sum()
    jaccard = float(numer / denom) if denom > 0 else 0.0

    xc1, xc2 = st.columns(2)
    with xc1:
        st.plotly_chart(bar_chart(orig_ids, energy_l, energy_r, 'Learned', 'Random walk',
                                  'Visitation energy', 'freq²'), use_container_width=True)
    with xc2:
        st.plotly_chart(bar_chart(orig_ids, delta_l, delta_r, 'Learned', 'Random walk',
                                  'Node ablation importance', '|Δlogit|'), use_container_width=True)

    mask_l = energy_l > 0; mask_r = energy_r > 0
    rho_l = spearmanr(energy_l[mask_l], np.array(delta_l)[mask_l]).correlation \
        if mask_l.sum() >= 2 else float('nan')
    rho_r = spearmanr(energy_r[mask_r], np.array(delta_r)[mask_r]).correlation \
        if mask_r.sum() >= 2 else float('nan')

    sc1, sc2, sc3 = st.columns(3)
    sc1.metric('Weighted Jaccard', f'{jaccard:.3f}')
    sc2.metric('Spearman ρ — learned', f'{rho_l:.3f}' if not np.isnan(rho_l) else 'n/a')
    sc3.metric('Spearman ρ — random walk', f'{rho_r:.3f}' if not np.isnan(rho_r) else 'n/a')
    st.caption('Spearman ρ: correlation between visitation energy and ablation importance. '
               'Higher → most-visited nodes are also most decision-critical.')
