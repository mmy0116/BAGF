import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import pickle
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import os
import warnings
import argparse

warnings.filterwarnings('ignore', category=UserWarning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument('--hidden_channels', type=int, default=100)
parser.add_argument('--in_channels', type=int, default=0)
parser.add_argument('--out_channels', type=int, default=1)
parser.add_argument('--dataset_file', type=str, default="BioPlex")
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--lr', type=float, default=0.0006)
parser.add_argument('--w_decay', type=float, default=0.0002)
parser.add_argument('--times', type=int, default=100, help='Number of times to repeat training.')
args = parser.parse_args()


# 加载数据
def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


# 加载数据集
if args.dataset_file == "CPDB":
    ppiAdj = load_obj("./data/CPDB/CPDB_ppi_3.pkl")
    pathAdj = load_obj("./data/CPDB/CPDB_path_3.pkl")
    omicsfeature = load_obj("./data/CPDB/CPDB_12850_5CV.pkl")
elif args.dataset_file == "BioPlex":
    ppiAdj = load_obj("./data/BioPlex/BioPlex_ppi_3.pkl")
    pathAdj = load_obj("./data/BioPlex/BioPlex_path_3.pkl")
    omicsfeature = load_obj("./data/BioPlex/BioPlex_8304_5CV.pkl")
elif args.dataset_file == "irefindex":
    ppiAdj = load_obj("./data/irefindex/irefindex_ppi_3.pkl")
    pathAdj = load_obj("./data/irefindex/irefindex_path_3.pkl")
    omicsfeature = load_obj("./data/irefindex/irefindex_11777_5CV.pkl")

# 准备图数据
graphlist = []

# 处理 PPI 网络
std_ppi = StandardScaler()
features_ppi = std_ppi.fit_transform(np.abs(omicsfeature['feature'].detach().numpy()))
features_ppi = torch.FloatTensor(features_ppi)

data_ppi = Data(
    x=features_ppi,
    y=omicsfeature['label'],
    edge_index=ppiAdj["edge_index"],
    mask=omicsfeature['mask'],  # 修改：从数据文件加载预定义的mask
    node_names=omicsfeature['node_name']
)
graphlist.append(data_ppi)

# 处理 Pathway 网络
std_path = StandardScaler()
features_path = std_path.fit_transform(np.abs(pathAdj['feature']))
features_path = torch.FloatTensor(features_path)

data_path = Data(
    x=features_path,
    y=omicsfeature['label'],
    edge_index=pathAdj["edge_index"],
    mask=omicsfeature['mask'],  # 修改：从数据文件加载预定义的mask
    node_names=omicsfeature['node_name']
)
graphlist.append(data_path)

# 将图数据传到设备
graphdata = [graph.to(device) for graph in graphlist]
data = graphdata[0]

# 更新输入通道数
args.in_channels = graphdata[0].x.shape[1]

# 导入模型
from my_model import my


# 训练函数
def train(data, graphdata, model, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(graphdata)
    loss = F.binary_cross_entropy_with_logits(out[data.mask], data.y[data.mask].view(-1, 1))
    loss.backward()
    optimizer.step()
    return loss.item()


# 准备保存结果的路径
file_save_path = './'
if not os.path.exists(file_save_path):
    os.makedirs(file_save_path)

# 用于累积所有节点的预测得分
pred_all_nodes = np.zeros((data.num_nodes, 1))

# 重复训练 args.times 次
for i in range(args.times):
    model = my(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)

    # 训练模型
    for epoch in range(1, args.epochs + 1):
        loss = train(data, graphdata, model, optimizer)

    # 预测并累积
    model.eval()
    with torch.no_grad():
        out = model(graphdata)
        pred_all_nodes += torch.sigmoid(out).cpu().detach().numpy()
    
    print(f'训练进度: {i+1}/{args.times} 完成')

# 计算平均预测得分
pred_all_nodes = pred_all_nodes / args.times

# 创建并保存结果
pre_res = pd.DataFrame(pred_all_nodes, columns=['score'], index=data.node_names)
pre_res.sort_values(by=['score'], inplace=True, ascending=False)

output_file = os.path.join(file_save_path, 'predicted_scores_BioPlex.txt')
pre_res.to_csv(path_or_buf=output_file, sep='\t', index=True, header=True)

print(f'\n预测得分文件已保存至: {output_file}')
print(f'文件包含 {len(pre_res)} 个基因的预测得分\n')
print('前10个预测得分最高的基因:')
print(pre_res.head(10))
