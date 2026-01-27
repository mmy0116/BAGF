import numpy as np
import torch
from sklearn import metrics
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
parser.add_argument('--in_channels', type=int, default=0)  # 将在数据加载后更新
parser.add_argument('--out_channels', type=int, default=1)  # 二分类任务
parser.add_argument('--dataset_file', type=str, default="CPDB")
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--lr', type=float, default=0.0006)
parser.add_argument('--w_decay', type=float, default=0.0002)
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

# 处理 PPI 网络（使用 omicsfeature 的特征）
std_ppi = StandardScaler()
features_ppi = std_ppi.fit_transform(np.abs(omicsfeature['feature'].detach().numpy()))
features_ppi = torch.FloatTensor(features_ppi)

data_ppi = Data(
    x=features_ppi,
    y=omicsfeature['label'],
    edge_index=ppiAdj["edge_index"],
    mask=omicsfeature['split_set'],
    node_names=omicsfeature['node_name']
)
graphlist.append(data_ppi)

# 处理 Pathway 网络
std_path = StandardScaler()
features_path = std_path.fit_transform(np.abs(pathAdj['feature']))  # 假设 pathAdj 包含特征
features_path = torch.FloatTensor(features_path)

data_path = Data(
    x=features_path,
    y=omicsfeature['label'],
    edge_index=pathAdj["edge_index"],
    mask=omicsfeature['split_set'],
    node_names=omicsfeature['node_name']
)
graphlist.append(data_path)

# 将图数据传到设备（GPU/CPU）
graphdata = [graph.to(device) for graph in graphlist]
data = graphdata[0]

# 更新输入通道数
args.in_channels = graphdata[0].x.shape[1]
# 打印 in_channels 的值
print(f"输入特征维度 (in_channels): {args.in_channels}")

# 导入修改后的模型
from my_model import my


# 测试函数
def test(data, mask, graphdata):
    model.eval()
    out = model(graphdata)

    pred = torch.sigmoid(out[mask])
    pred_prob = pred.cpu().detach().numpy()
    true_labels = data.y[mask].cpu().numpy()
    
    # AUC和AUPR计算
    precision, recall, _thresholds = metrics.precision_recall_curve(true_labels, pred_prob)
    aupr = metrics.auc(recall, precision)
    auc = metrics.roc_auc_score(true_labels, pred_prob)
    
    return auc, aupr, pred_prob


# 训练函数
def train(tr_mask, data, graphdata):
    # 训练模型
    model.train()
    optimizer.zero_grad()
    out = model(graphdata)

    # 只使用一个损失函数，因为模型架构已经改变
    loss = F.binary_cross_entropy_with_logits(out[tr_mask], data.y[tr_mask].view(-1, 1))

    loss.backward()
    optimizer.step()
    
    return loss.item()


# 准备保存结果的路径
file_save_path = './data'
if not os.path.exists(file_save_path):
    os.makedirs(file_save_path)

# 十次五折交叉验证
AUC = np.zeros(shape=(10, 5))
AUPR = np.zeros(shape=(10, 5))
pred_all = []
label_all = []

for i in range(10):
    for cv_run in range(5):
        tr_mask, te_mask = data.mask[i][cv_run]
        model = my(args).to(device)

        # 使用单一优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)

        for epoch in range(1, args.epochs + 1):
            loss = train(tr_mask, data, graphdata)
            if epoch % 50 == 0:  # 每50个epoch打印一次
                # 计算当前的所有指标
                train_auc, train_aupr, _ = test(data, tr_mask, graphdata)
                test_auc, test_aupr, _ = test(data, te_mask, graphdata)
                print(f'Round-{i}, CV-{cv_run + 1}, Training epoch: {epoch:03d}, Loss: {loss:.6f}, '
                      f'Train AUC: {train_auc:.5f}, Train AUPR: {train_aupr:.5f}, '
                      f'Test AUC: {test_auc:.5f}, Test AUPR: {test_aupr:.5f}')

        # 最终测试
        AUC[i][cv_run], AUPR[i][cv_run], pred = test(data, te_mask, graphdata)
        pred_all.append(pred)
        label_all.append(data.y[te_mask].cpu().numpy())
        print('Round--%d CV--%d  Final AUC: %.5f, Final AUPR: %.5f' % 
              (i, cv_run + 1, AUC[i][cv_run], AUPR[i][cv_run]))
    
    print('Round--%d Mean AUC: %.5f, Mean AUPR: %.5f' % 
          (i, np.mean(AUC[i, :]), np.mean(AUPR[i, :])))

# 最终结果输出
print('BAGF模型 10轮5折交叉验证-- 平均 AUC: %.4f, 平均 AUPR: %.4f' %
      (AUC.mean(), AUPR.mean()))

# 保存结果
torch.save(pred_all, os.path.join(file_save_path, 'CPDB_pred_all.pkl'))
torch.save(label_all, os.path.join(file_save_path, 'CPDB_label_all.pkl'))

# 保存指标的详细结果
results_dict = {
    'AUC': AUC,
    'AUPR': AUPR
}
torch.save(results_dict, os.path.join(file_save_path, 'CPDB_all_metrics.pkl'))
