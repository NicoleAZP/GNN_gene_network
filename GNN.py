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

# Step 1: Load large-scale gene expression data
df = pd.read_csv('LAML_mean.csv')

# Step 2: Select random samples and remove gene name and gene type columns
gene_expression = df.drop(['gene_name', 'gene_type'], axis=1).values

# Step 3: Select high-variance genes based on variance to reduce dimensionality
gene_variances = np.var(gene_expression, axis=1)
top_genes_idx = np.argsort(gene_variances)[-2000:]  # Select the top 2000 genes with highest variance
gene_expression_reduced = gene_expression[top_genes_idx, :]

# Step 4: Standardize gene expression data
scaler = StandardScaler()
gene_expression_reduced = scaler.fit_transform(gene_expression_reduced)

# Step 5: Compute cosine similarity matrix of the gene expression data
similarity_matrix = cosine_similarity(gene_expression_reduced)

# Step 6: Construct a sparse similarity matrix, keeping only gene pairs with similarity above a threshold
threshold = 0.5  # Set similarity threshold
edge_index = []
k = 10  # Each node connects to its 10 most similar neighbors
for i in range(len(similarity_matrix)):
    top_k_indices = np.argsort(similarity_matrix[i])[-k:]
    for j in top_k_indices:
        if similarity_matrix[i, j] > threshold:
            edge_index.append([i, j])
            edge_index.append([j, i])

# Convert to PyTorch tensor
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# Remove self-loops
edge_index, _ = remove_self_loops(edge_index)

# Step 7: Use gene expression data as node features and create PyTorch Geometric Data object
x = torch.tensor(gene_expression_reduced, dtype=torch.float)
data = Data(x=x, edge_index=edge_index)

# Step 8: Encode gene categories into numerical labels
label_encoder = LabelEncoder()
gene_labels = df['gene_type'].values[top_genes_idx]  # Only keep labels for the selected genes
gene_labels = label_encoder.fit_transform(gene_labels)
gene_labels = torch.tensor(gene_labels, dtype=torch.long)

# Step 9: Split the dataset into training and test sets based on genes
num_genes = gene_expression_reduced.shape[0]  # Number of genes is the number of rows
train_idx, test_idx = train_test_split(np.arange(num_genes), test_size=0.2, random_state=42)

# Corresponding gene labels
train_labels = gene_labels[train_idx]  # Training set gene labels
test_labels = gene_labels[test_idx]    # Test set gene labels

# Step 10: Compute class weights to handle class imbalance
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(gene_labels), y=gene_labels.numpy())
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Step 11: Define GCN model and optimizer
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


# Initialize model
device = torch.device('cpu')
model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

# Step 12: Train the model
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)[train_idx]   # Forward propagation on the training set
    loss = criterion(out, train_labels)  # Compute loss
    loss.backward()
    optimizer.step()
    return loss.item()

# Training process
for epoch in range(60000):
    loss = train()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

# Step 13: Test the model
def test():
    model.eval()
    with torch.no_grad():
        out = model(data)[test_idx]  # Make predictions on the test set
        pred = out.argmax(dim=1)  # Choose the class with the highest score as the prediction
        acc = accuracy_score(test_labels.cpu(), pred.cpu())  # Accuracy
        f1 = f1_score(test_labels.cpu(), pred.cpu(), average='macro')  # F1 score
        return acc, f1

# Test model performance
acc, f1 = test()
print(f'Test Accuracy: {acc:.4f}, Test F1 Score: {f1:.4f}')
