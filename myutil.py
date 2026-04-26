import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import torch_geometric.utils as pyg_utils
from sklearn import metrics
from sklearn.neighbors import kneighbors_graph
import torch_geometric as torchgeo
from model_GraphSAGE import *
from sklearn.metrics import normalized_mutual_info_score
import random
import torchvision.models as models
nmi = normalized_mutual_info_score


def cluster_accuracy(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)

    # Find optimal one-to-one mapping between cluster labels and true labels
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)

    # Return cluster accuracy
    return contingency_matrix[row_ind, col_ind].sum() / np.sum(contingency_matrix)



# myutil.py中修正load_pretrain_model函数
def load_pretrain_model(model_name):
    if model_name == 'convnext_tiny':
        model = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.DEFAULT
        )
    elif model_name == 'convnext_small':
        model = models.convnext_small(
            weights=models.ConvNeXt_Small_Weights.DEFAULT
        )

    elif model_name == 'convnext_base':
        model = models.convnext_base(
            weights=models.ConvNeXt_Base_Weights.DEFAULT
        )

    elif model_name == 'convnext_large':
        model = models.convnext_large(
            weights=models.ConvNeXt_Large_Weights.DEFAULT
        )

    elif model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def build_gaussian_graph(features, sigma=1.0, threshold=0.1):
    """
    features: 样本特征矩阵 [n_samples, feat_dim] 你的ConvNeXt提取的特征
    sigma: 高斯核宽度，经验值0.5-2.0即可，无需精细调参
    threshold: 相似度阈值，过滤极小的相似度（减少边数，降低计算量）
    return: edge_index (PyG格式) + 邻接矩阵
    """
    # 计算两两样本的欧式距离
    dist_matrix = cdist(features, features, metric='euclidean')
    # 计算高斯相似度矩阵
    sim_matrix = np.exp(- dist_matrix ** 2 / (2 * sigma ** 2))
    # 过滤低相似度的边，生成邻接矩阵
    adj_matrix = (sim_matrix >= threshold).astype(np.float32)
    # 转为PyG的edge_index格式，适配你的GraphSAGE
    edge_index = pyg_utils.dense_to_sparse(torch.from_numpy(adj_matrix))[0]
    return edge_index, adj_matrix

def bulid_pyg_data(features, sparse_adj):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datas = []
    for i in range(len(features)):
        pyg_graph = torchgeo.data.Data()
        pyg_graph.x = features[i].to(device)
        edge_index = torch.from_numpy(np.transpose(np.stack(sparse_adj[i].nonzero(), axis=1))).long().to(device)
        pyg_graph.edge_index = edge_index
        pyg_graph.edge_index = torchgeo.utils.to_undirected(pyg_graph.edge_index)
        pyg_graph.num_nodes = features[i].shape[0]
        datas.append(pyg_graph)
    return datas