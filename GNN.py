import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch_geometric.utils import remove_self_loops
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

# Step 1: 载入大规模基因表达数据
df = pd.read_csv('LAML_mean.csv')

# Step 2: 随机选择部分样本，删除基因名和基因类型列
gene_expression = df.drop(['gene_name', 'gene_type'], axis=1).values

# Step 3: 基于方差筛选高方差基因，减少维度
gene_variances = np.var(gene_expression, axis=1)
top_genes_idx = np.argsort(gene_variances)[-2000:]  # 选择方差最高的1200基因
gene_expression_reduced = gene_expression[top_genes_idx, :]

# Step 4: 标准化基因表达数据
scaler = StandardScaler()
gene_expression_reduced = scaler.fit_transform(gene_expression_reduced)

# Step 5: 计算基因表达的余弦相似度矩阵
similarity_matrix = cosine_similarity(gene_expression_reduced)

# Step 6: 构建稀疏的相似度矩阵，只保留相似度超过某个阈值的基因对
threshold = 0.5  # 设置相似度阈值
edge_index = []
k = 10  # 每个节点连接的前20个最相似的节点
for i in range(len(similarity_matrix)):
    top_k_indices = np.argsort(similarity_matrix[i])[-k:]
    for j in top_k_indices:
        if similarity_matrix[i, j] > threshold:
            edge_index.append([i, j])
            edge_index.append([j, i])

# 转换为 PyTorch 张量
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# 去除自环
edge_index, _ = remove_self_loops(edge_index)

# Step 7: 将基因表达数据作为节点特征，创建 PyG 数据对象
x = torch.tensor(gene_expression_reduced, dtype=torch.float)
data = Data(x=x, edge_index=edge_index)

# Step 8: 对基因类别进行标签编码
label_encoder = LabelEncoder()
gene_labels = df['gene_type'].values[top_genes_idx]  # 只保留筛选后的基因标签
gene_labels = label_encoder.fit_transform(gene_labels)
gene_labels = torch.tensor(gene_labels, dtype=torch.long)

# Step 9: 按照基因进行训练集和测试集的划分
num_genes = gene_expression_reduced.shape[0]  # 基因数量为行数
train_idx, test_idx = train_test_split(np.arange(num_genes), test_size=0.2, random_state=42)

# 对应的基因类别标签
train_labels = gene_labels[train_idx]  # 训练集基因标签
test_labels = gene_labels[test_idx]    # 测试集基因标签

# Step 10: 计算类别权重，处理类别不平衡问题
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(gene_labels), y=gene_labels.numpy())
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Step 11: 定义 GCN 模型和优化器
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(data.num_node_features, 128)
        self.conv2 = GCNConv(128, 256)
        self.conv3 = GCNConv(256, 128)
        self.conv4 = GCNConv(128, len(np.unique(gene_labels)))
        self.dropout_rate = 0.5

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        x = self.conv4(x, edge_index)
        return F.log_softmax(x, dim=1)


# 初始化模型
device = torch.device('cpu')
model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

# Step 12: 训练模型
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)[train_idx]   # 仅在训练集上进行前向传播
    loss = criterion(out, train_labels)  # 计算损失
    loss.backward()
    optimizer.step()
    return loss.item()

# 训练过程
for epoch in range(60000):
    loss = train()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

# Step 13: 测试模型
def test():
    model.eval()
    with torch.no_grad():
        out = model(data)[test_idx]  # 在测试集上进行预测
        pred = out.argmax(dim=1)  # 取最大值对应的类别作为预测结果
        acc = accuracy_score(test_labels.cpu(), pred.cpu())  # 准确率
        f1 = f1_score(test_labels.cpu(), pred.cpu(), average='macro')  # F1 score
        return acc, f1

# 测试模型性能
acc, f1 = test()
print(f'Test Accuracy: {acc:.4f}, Test F1 Score: {f1:.4f}')
