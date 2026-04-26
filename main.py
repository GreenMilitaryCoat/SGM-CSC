import os
os.environ["OMP_NUM_THREADS"] = "4"
import argparse
import os.path
from myutil import *
from model_GraphSAGE import *
from post_clustering import *
from data import *
from tqdm import tqdm

setup_seed(100)
parser = argparse.ArgumentParser(description="use pretraining net work for feature extract")

parser.add_argument("--dataset",
                        dest='dataset',
                        choices=('fashion_mnist',
                                 'cifar10',
                                 'cifar100',
                                 'stl10',
                                 'imagenet10',     # 新增
                                 'imagenet_dogs'), # 新增
                        help="Dataset to train",
                        default='stl10')
parser.add_argument("--beta", type=float, default=0)
parser.add_argument("--lambda1", type=float, default=1)
parser.add_argument("--lambda2", type=float, default=1)
parser.add_argument("--lambda3", type=float, default=1)
parser.add_argument("--lambda4", type=float, default=1)
parser.add_argument("--lambda5", type=float, default=0.5)
parser.add_argument("--lambda6", type=float, default=1)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--epochs", type=int, default=300)

args = parser.parse_args()


dim_subspace = 12
ro = 8
alpha = 0.04
num_cluster = 10
if args.dataset == 'cifar100':
    num_cluster = 100
    args.epochs = 500
elif args.dataset == 'imagenet_dogs':
    num_cluster = 15
elif args.dataset == 'imagenet10':  # 新增 imagenet10 分支
    num_cluster = 10

# load data
saved_features = torch.load(os.path.join(features_save_dir, args.dataset + features_suffix))
features = saved_features['data']
label = saved_features['label']

# 修改后代码
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
datas = []
sigma = 0.5 # 高斯核宽度（可根据需求调整）
threshold = 0.1 # 相似度阈值（可根据需求调整）

for feat in features:
    # 特征转为numpy数组（适配cdist）
    feat_np = feat.cpu().numpy()
    # 构建高斯图
    edge_index, adj_matrix = build_gaussian_graph(feat_np, sigma=sigma, threshold=threshold)
    # 构建PyG数据对象
    pyg_graph = torchgeo.data.Data()
    pyg_graph.x = feat.to(device)  # 特征保持原格式
    pyg_graph.edge_index = edge_index.long().to(device)  # 转换为长整型并移至设备
    pyg_graph.edge_index = torchgeo.utils.to_undirected(pyg_graph.edge_index)  # 转为无向图
    pyg_graph.num_nodes = feat.shape[0]
    datas.append(pyg_graph)

model = GraphSAGECluster(features=features, hidden_channels=16, num_sample=features[0].shape[0] )
model.to(device)

# loss and optimizer
optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)

# training loop
pbar = tqdm(range(args.epochs))
for epoch in pbar:
    fusion_expression, content_features, structure_features, content_expression, structure_expression = model(datas)

    # self-expression loss
    attribute_express_loss = F.mse_loss(content_features, torch.mm(content_expression, content_features))
    graph_express_loss = F.mse_loss(structure_features, torch.mm(structure_expression, structure_features))

    # self-expression coefficient loss
    attribute_express_coefficient_loss = torch.linalg.matrix_norm(content_expression, 1)
    graph_express_coefficient_loss = torch.linalg.matrix_norm(structure_expression, 1)

    # C_F loss
    fusion_expression_coefficient_loss = torch.linalg.matrix_norm(fusion_expression, 1)

    consistency_loss = F.smooth_l1_loss(content_expression, structure_expression, beta=1.0)

    total_loss = args.lambda1 * attribute_express_loss + \
                 args.lambda2 * graph_express_loss + \
                 args.lambda3 * attribute_express_coefficient_loss + \
                 args.lambda4 * graph_express_coefficient_loss + \
                 args.lambda5 * consistency_loss + \
                 args.lambda6 * fusion_expression_coefficient_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    pbar.set_postfix({"loss": total_loss.item()})

# results
print("content self-expression clustering results:")
Ca = content_expression.detach().to('cpu').numpy()
y_pred = sklearn_spectral_clustering(Ca, num_cluster)
print(f"ACC = {cluster_accuracy(label, y_pred):.4f}, NMI = {nmi(label, y_pred):.4f}, ARI = {ari(label, y_pred):.4f}")

print("fusion self-expression clustering results:")
C = fusion_expression.detach().to('cpu').numpy()
y_pred = sklearn_spectral_clustering(C, num_cluster)
print(f"ACC = {cluster_accuracy(label, y_pred):.4f}, NMI = {nmi(label, y_pred):.4f}, ARI = {ari(label, y_pred):.4f}")









