# Graph Neural Network (GNN) Project
## Author: Ziqi Zhou
## Introduction

This project implements a Graph Neural Network (GNN) for working with graph-structured data. GNNs are powerful tools for various graph-based tasks such as node classification, graph classification, and link prediction. They work by leveraging message-passing mechanisms, allowing nodes in a graph to exchange information with their neighbors. This process allows the network to capture the structural relationships between nodes and generate useful representations for downstream tasks.

This specific implementation includes several GNN layers, and can be adapted for different graph learning tasks.

## Features

- **Modular Design**: The implementation allows for easy integration of different GNN layers such as Graph Convolutional Networks (GCN), Graph Attention Networks (GAT), and others.
- **Customizable Model**: Users can modify the number of layers, the hidden units per layer, and activation functions.
- **Flexible Input**: The implementation can work with various types of graph data, making it adaptable to different datasets.
- **Optimization Ready**: The code includes standard optimization techniques like weight regularization, dropout, and learning rate scheduling.

## File Structure

- **GNN.py**: Contains the GNN model implementation, including the data loading and training pipeline.
- **dataset/graph_data.pkl**: A placeholder for the dataset file. In this project, the dataset is expected to be in a pickled format and should consist of graphs where each node has associated features and labels (if applicable).
- **utils.py**: Contains utility functions for loading datasets, processing graph data, and splitting the data into train/validation/test sets.

## Model Architecture

The GNN model consists of the following components:

1. **Input Layer**: Accepts node features and edge information from the input graph.
2. **Hidden Layers**: Several GNN layers (such as GCN or GAT) are stacked, each followed by an activation function (e.g., ReLU).
3. **Output Layer**: Generates final node/graph-level embeddings used for the specific task at hand.

## Dependencies

To run this project, ensure you have the following dependencies installed:

- `torch` (PyTorch)
- `torch-geometric` (PyTorch Geometric)
- `networkx`
- `numpy`
- `scikit-learn`

You can install these dependencies using the following command:

```bash
pip install torch torch-geometric networkx numpy scikit-learn
```

## Usage

To train the model, run the following command:

```bash
python GNN.py
```

The script automatically handles data loading and model training. To modify hyperparameters (e.g., learning rate, number of epochs, GNN layer types), you can edit the relevant sections of `GNN.py`.

## Dataset

The dataset should be stored in a directory named `dataset`. The expected format for the dataset is a pickled file containing a graph object with node features and edge lists. Example:

```python
import pickle
with open('dataset/graph_data.pkl', 'rb') as f:
    graph_data = pickle.load(f)
```

The graph object should include:
- **Node features**: A matrix where each row corresponds to the features of a node.
- **Edge list**: A list of tuples representing edges between nodes.
- **Labels (optional)**: If performing node classification, the graph should also contain labels for each node.

## Model Customization

The model can be customized by modifying the following parameters in `GNN.py`:

- **Number of layers**: Adjust how many GNN layers to stack.
- **Hidden units**: Set the number of hidden units in each layer.
- **Activation functions**: Change the non-linearity applied after each GNN layer.
- **Dropout**: Configure dropout to prevent overfitting.
- **Learning rate**: Modify the learning rate for optimization.

## Results

The training process will print the loss and accuracy on the training and validation sets. After training, the model will output final metrics and save the model weights.

## Contact

For any questions or issues, feel free to contact the project maintainer.
