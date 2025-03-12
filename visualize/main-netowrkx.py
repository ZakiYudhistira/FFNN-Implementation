import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

# Define the architecture
input_layer_size = 3
hidden_layer_size = 4
output_layer_size = 2

# Randomly initialize weights for demonstration
weights_input_hidden = np.random.randn(input_layer_size, hidden_layer_size).round(4)
weights_hidden_output = np.random.randn(hidden_layer_size, output_layer_size).round(4)

print("Weights Input -> Hidden:\n", weights_input_hidden)
print("Weights Hidden -> Output:\n", weights_hidden_output)

def visualize_ffnn(weights_input_hidden, weights_hidden_output):
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes for each layer
    input_nodes = [f"Input_{i}" for i in range(weights_input_hidden.shape[0])]
    hidden_nodes = [f"Hidden_{i}" for i in range(weights_input_hidden.shape[1])]
    output_nodes = [f"Output_{i}" for i in range(weights_hidden_output.shape[1])]

    # Add edges with weights for Input -> Hidden
    for i, input_node in enumerate(input_nodes):
        for j, hidden_node in enumerate(hidden_nodes):
            weight = weights_input_hidden[i, j]
            G.add_edge(input_node, hidden_node, weight=weight)

    # Add edges with weights for Hidden -> Output
    for i, hidden_node in enumerate(hidden_nodes):
        for j, output_node in enumerate(output_nodes):
            weight = weights_hidden_output[i, j]
            G.add_edge(hidden_node, output_node, weight=weight)

    # Define positions for the nodes (layered layout)
    pos = {}
    layer_spacing = 1.0
    for i, node in enumerate(input_nodes):
        pos[node] = (0, i * layer_spacing)
    for i, node in enumerate(hidden_nodes):
        pos[node] = (1, i * layer_spacing)
    for i, node in enumerate(output_nodes):
        pos[node] = (2, i * layer_spacing)

    # Draw the graph
    plt.figure(figsize=(10, 6))
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=800)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

    # Draw edges with weights as labels
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color='gray')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.7, font_color='red')

    plt.title("Feed-Forward Neural Network Visualization")
    plt.axis('off')
    plt.show()

# Call the visualization function
visualize_ffnn(weights_input_hidden, weights_hidden_output)