import torch
import numpy as np
from tqdm import tqdm
import pickle
import random
import os
import torch.nn.functional as F
from src.models import GraphSAGE
from .dataloader import NodeNegativeLoader
from .loss.contrastive_loss import ContrastiveLoss, GraceLoss
from .get_pc import get_pc
from torch_geometric.data import Data

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def train(args, dataset, model, optimizer, criterion, device, fans_out, all_principal_component, train_id):
    model.train()
    total_loss = 0
    total_ins_loss = 0
    total_con_loss = 0
    total_nodes = 0

    # Wrap single graph into a list for consistency
    graph_list = dataset if isinstance(dataset, list) else [dataset]

    for graph in graph_list:
        train_loader = NodeNegativeLoader(
            graph,
            batch_size=args.batch_size,
            shuffle=True,
            neg_ratio=args.num_negs,
            num_neighbors=fans_out,
            mask_feat_ratio_1=args.drop_feature_rate_1,
            mask_feat_ratio_2=args.drop_feature_rate_2,
            drop_edge_ratio_1=args.drop_edge_rate_1,
            drop_edge_ratio_2=args.drop_edge_rate_2,
        )
        pbar = tqdm(total=len(train_loader))

        for step, (ori_graph, view_1, view_2) in enumerate(train_loader):
            ori_graph, view_1, view_2 = ori_graph.to(device), view_1.to(device), view_2.to(device)

            optimizer.zero_grad()
            z1 = model(view_1.x, view_1.edge_index)[view_1.node_label_index]
            z2 = model(view_2.x, view_2.edge_index)[view_2.node_label_index]

            proj_z1 = model.projection(z1)
            proj_z2 = model.projection(z2)

            principal_component = all_principal_component[ori_graph.raw_nodes] if args.self_tp else all_principal_component

            if args.use_tp:
                loss, ins_loss, contrast_loss = criterion(z1, z2, proj_z1, proj_z2, principal_component)
                total_ins_loss += ins_loss * proj_z1.shape[0]
                total_con_loss += contrast_loss * proj_z1.shape[0]
                total_loss += (ins_loss + contrast_loss) * proj_z1.shape[0]
            else:
                loss = criterion(proj_z1, proj_z2)
                total_loss += loss.item() * proj_z1.shape[0]

            total_nodes += proj_z1.shape[0]
            loss.backward()
            optimizer.step()

            if step % args.log_every == 0:
                if args.use_tp:
                    print(f'Step {step:05d} | Loss {loss.item():.4f} | Instance {ins_loss:.4f} | Contrast {contrast_loss:.4f}')
                else:
                    print(f'Step {step:05d} | Loss {loss.item():.4f}')
            pbar.update()
        pbar.close()

    return total_loss / total_nodes, total_ins_loss / total_nodes, total_con_loss / total_nodes

@torch.no_grad()
def test(args, dataset, model, criterion, device, all_principal_component):
    model.eval()
    total_loss = 0
    total_ins_loss = 0
    total_con_loss = 0
    total_nodes = 0

    graph_list = dataset if isinstance(dataset, list) else [dataset]

    for graph in graph_list:
        test_loader = NodeNegativeLoader(
            graph,
            batch_size=512,
            shuffle=False,
            neg_ratio=0,
            num_neighbors=[-1],
            mask_feat_ratio_1=args.drop_feature_rate_1,
            mask_feat_ratio_2=args.drop_feature_rate_2,
            drop_edge_ratio_1=args.drop_edge_rate_1,
            drop_edge_ratio_2=args.drop_edge_rate_2,
        )
        pbar = tqdm(total=len(test_loader))

        for ori_graph, view_1, view_2 in test_loader:
            ori_graph, view_1, view_2 = ori_graph.to(device), view_1.to(device), view_2.to(device)

            z1 = model(view_1.x, view_1.edge_index)[view_1.node_label_index]
            z2 = model(view_2.x, view_2.edge_index)[view_2.node_label_index]

            proj_z1 = model.projection(z1)
            proj_z2 = model.projection(z2)

            principal_component = all_principal_component[ori_graph.raw_nodes] if args.self_tp else all_principal_component

            if args.use_tp:
                loss, ins_loss, contrast_loss = criterion(z1, z2, proj_z1, proj_z2, principal_component)
                total_ins_loss += ins_loss * proj_z1.shape[0]
                total_con_loss += contrast_loss * proj_z1.shape[0]
                total_loss += (ins_loss + contrast_loss) * proj_z1.shape[0]
            else:
                loss = criterion(proj_z1, proj_z2)
                total_loss += loss.item() * proj_z1.shape[0]

            total_nodes += proj_z1.shape[0]
            pbar.update()
        pbar.close()

    print(f"Mean Test Loss: {total_loss / total_nodes}")
    print(f"Mean Instance Loss: {total_ins_loss / total_nodes}")
    print(f"Mean Contrastive Loss: {total_con_loss / total_nodes}")

    return total_loss / total_nodes, total_ins_loss / total_nodes, total_con_loss / total_nodes

def run(seed, data, dataset_name, llm_name,
        gpu=0, num_epochs=70, num_layers=2, num_negs=0, patience=10,
        fan_out='25,10', batch_size=512, log_every=20, eval_every=50,
        lr=0.002, dropout=0.5, num_workers=0, lazy_load=True,
        use_tp=True, self_tp=False, drop_edge_rate_1=0.3, drop_edge_rate_2=0.4,
        drop_feature_rate_1=0.0, drop_feature_rate_2=0.1, tau=0.4,
        gnn_type='sage', models_root='../../../models/', datasets_root='../../../datasets/'):

    fans_out = [int(i) for i in fan_out.split(',')]
    assert len(fans_out) == num_layers

    device = torch.device(f'cuda:{gpu}' if gpu >= 0 else 'cpu')

    model_save_path = os.path.join(models_root, 'teaglm/')
    os.makedirs(model_save_path, exist_ok=True)

    print("Node Unsupervised")

    if isinstance(data, list):
        for g in data:
            g.x = g.x.type(torch.float)
        train_id = torch.cat([g.x for g in data])
    else:
        data.x = data.x.type(torch.float)
        train_id = data.x

    print(f"Seed {seed}")
    seed_everything(seed)

    if not isinstance(data, list):
        data = data.to(device, 'x', 'edge_index')

    llm_str, all_principal_component = get_pc(llm_name, datasets_root=datasets_root)
    all_principal_component = all_principal_component.to(device, dtype=torch.float)

    num_node_features = data[0].x.shape[1] if isinstance(data, list) else data.x.shape[1]

    model = GraphSAGE(
        num_node_features,
        hidden_channels=all_principal_component.size(1) // 2,
        out_channels=all_principal_component.size(1),
        n_layers=num_layers,
        num_proj_hidden=all_principal_component.size(1),
        activation=F.relu,
        dropout=dropout,
        edge_dim=None,
        gnn_type=gnn_type
    ).to(device)
    model = model.to(dtype=torch.float)

    print(model)

    criterion = ContrastiveLoss(tau, self_tp=self_tp).to(device) if use_tp else GraceLoss(tau).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    args = type('Args', (), {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'log_every': log_every,
        'eval_every': eval_every,
        'drop_edge_rate_1': drop_edge_rate_1,
        'drop_edge_rate_2': drop_edge_rate_2,
        'drop_feature_rate_1': drop_feature_rate_1,
        'drop_feature_rate_2': drop_feature_rate_2,
        'num_negs': num_negs,
        'lazy_load': lazy_load,
        'use_tp': use_tp,
        'self_tp': self_tp
    })

    no_increase = 0
    best_loss = float('inf')
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        total_mean_loss, total_mean_instance_loss, total_mean_contrastive_loss = train(
            args, data, model, optimizer, criterion, device, fans_out, all_principal_component, train_id
        )
        if total_mean_loss < best_loss:
            best_loss = total_mean_loss
            no_increase = 0
            torch.save(model.state_dict(), os.path.join(model_save_path, f'GraphSAGE_{dataset_name}_{llm_str}_seed_{seed}.pth'))
        else:
            no_increase += 1
            if no_increase > patience:
                break

    with open(os.path.join(model_save_path, f'GraphSAGE_{dataset_name}_{llm_str}_seed_{seed}_model_params.pkl'), "wb") as f:
        params = {
            "in_channels": num_node_features,
            "hidden_channels": all_principal_component.size(1) // 2,
            "out_channels": all_principal_component.size(1),
            "n_layers": num_layers,
            "num_proj_hidden": all_principal_component.size(1),
            "activation": F.relu,
            "dropout": dropout,
            "edge_dim": None,
            "gnn_type": gnn_type
        }
        pickle.dump(params, f)

if __name__ == '__main__':
    pass
