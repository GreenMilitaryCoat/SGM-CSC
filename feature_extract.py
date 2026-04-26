import argparse
from myutil import *
import copy
from data import *
import types
import os



parser = argparse.ArgumentParser(description="use pretraining net work for feature extract")
parser.add_argument("--dataset",
                        # required=True,
                        dest='dataset',
                    choices=('fashion_mnist',
                             'cifar10',
                             'stl10',
                             'cifar100',
                             'imagenet10',  # 新增
                             'imagenet_dogs',  # 新增
                             'tiny_imagenet'),  # 新增
                    help="Dataset to train")

parser.add_argument("--model",
                   choices=('convnext_tiny', 'convnext_small', 'convnext_base',
                            'convnext_large', 'resnet50'),  # 新增resnet50
                   default='convnext_base')
args = parser.parse_args()

if not os.path.exists(features_save_dir):
    os.mkdir(features_save_dir)

setup_seed(100)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.dataset == 'cifar10':
    all_img = False
    num_clusters = 10
    img_size =150
elif args.dataset == 'stl10':
    all_img = False
    num_clusters = 10
    img_size = 240
elif args.dataset == 'cifar100':
    all_img = False
    num_clusters = 100
    img_size =150
elif args.dataset == 'fashion_mnist':
    all_img = False
    num_clusters = 10
    img_size = 150
elif args.dataset == 'imagenet_dogs':
    all_img = False
    num_clusters = 15
    img_size = 224
elif args.dataset == 'imagenet10':
    all_img = False
    num_clusters = 10  # imagenet10 是10类
    img_size = 224

# model choose
model = load_pretrain_model(args.model)
model.to(device)

# add method get_middle_features to model

def get_middle_feature_convnext(self, x):
    """
    Extract 4-stage ConvNeXt features.
    torchvision ConvNeXt.features structure (typical):
    [0] stem
    [1] stage1
    [2] downsample
    [3] stage2
    [4] downsample
    [5] stage3
    [6] downsample
    [7] stage4
    """
    feats = []
    out = x
    stage_indices = [1, 3, 5, 7]

    for i, m in enumerate(self.features):
        out = m(out)
        if i in stage_indices:
            feats.append(
                torch.flatten(
                    F.adaptive_avg_pool2d(out, 1), 1
                )
            )

    return feats


def get_middle_feature_resnet(self, x):
    """
    Extract ResNet layer1-layer4 features (对应conv1后+4个stage)
    ResNet结构：conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2 -> layer3 -> layer4
    """
    feats = []
    # 前处理（conv1 + bn1 + relu + maxpool）
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    # 提取4个stage的输出特征
    x = self.layer1(x)
    feats.append(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))  # layer1特征
    x = self.layer2(x)
    feats.append(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))  # layer2特征
    x = self.layer3(x)
    feats.append(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))  # layer3特征
    x = self.layer4(x)
    feats.append(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))  # layer4特征
    return feats


# 修改模型特征提取方法的绑定逻辑
# 原代码中绑定convnext特征提取的部分替换为：
if args.model.startswith('convnext'):
    model.get_middle_feature = types.MethodType(get_middle_feature_convnext, model)
elif args.model == 'resnet50':
    model.get_middle_feature = types.MethodType(get_middle_feature_resnet, model)

# =========================
# Load data
# =========================
dataset = load_raw_image(args.dataset, img_size)
dl = DataLoader(dataset, batch_size=100, shuffle=False)


# =========================
# Feature extraction
# =========================
model.eval()
features = []
y_true = torch.empty((0,), dtype=torch.long)

for i, (X, y) in enumerate(dl):

    if args.dataset != 'cifar100' and not all_img and i == 10:
        break
    if args.dataset == 'cifar100' and not all_img and i == 30:
        break

    X = X.to(device)
    y_true = torch.cat((y_true, y), dim=0)

    with torch.no_grad():
        feat_list = model.get_middle_feature(X)

    if len(features) == 0:
        features = copy.deepcopy(feat_list)
    else:
        for j in range(len(features)):
            features[j] = torch.cat(
                (features[j], feat_list[j]), dim=0
            )

torch.save({'data':features, 'label':y_true}, os.path.join(features_save_dir, args.dataset + features_suffix))
