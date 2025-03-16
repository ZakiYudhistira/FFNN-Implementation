import numpy as np
from graphviz import Digraph

# Define the architecture
input_layer_size = 3
hidden_layer_size = 4
output_layer_size = 2

# Randomly initialize weights for demonstration
np.random.seed(42)
weights_input_hidden = np.random.randn(input_layer_size, hidden_layer_size)
weights_hidden_output = np.random.randn(hidden_layer_size, output_layer_size)

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Example input
input_data = np.array([0.5, 0.8, 0.2])

# Step 1: Input -> Hidden Layer
hidden_weighted_sum = np.dot(input_data, weights_input_hidden)
hidden_activations = sigmoid(hidden_weighted_sum)

# Step 2: Hidden -> Output Layer
output_weighted_sum = np.dot(hidden_activations, weights_hidden_output)
output_activations = sigmoid(output_weighted_sum)

def visualize_step_graphviz(input_data, hidden_weighted_sum=None, hidden_activations=None,
                            output_weighted_sum=None, output_activations=None, step=1):
    # Create a Graphviz graph
    dot = Digraph(comment=f"Step {step}: FFNN Visualization")
    dot.attr(rankdir='LR', size='8,5')  # Left-to-right layout

    # Add input nodes
    for i in range(len(input_data)):
        dot.node(f"Input_{i}", f"Input\n{input_data[i]:.2f}", shape="circle", style="filled", fillcolor="lightblue")

    # Add hidden nodes
    for j in range(weights_input_hidden.shape[1]):
        if hidden_activations is not None:
            label = f"Hidden\n{hidden_activations[j]:.2f}"
        elif hidden_weighted_sum is not None:
            label = f"Hidden\n{hidden_weighted_sum[j]:.2f}"
        else:
            label = "Hidden"
        dot.node(f"Hidden_{j}", label, shape="circle", style="filled", fillcolor="lightgreen")

    # Add output nodes
    for k in range(weights_hidden_output.shape[1]):
        if output_activations is not None:
            label = f"Output\n{output_activations[k]:.2f}"
        elif output_weighted_sum is not None:
            label = f"Output\n{output_weighted_sum[k]:.2f}"
        else:
            label = "Output"
        dot.node(f"Output_{k}", label, shape="circle", style="filled", fillcolor="orange")

    # Add edges for Input -> Hidden
    for i in range(weights_input_hidden.shape[0]):
        for j in range(weights_input_hidden.shape[1]):
            weight = weights_input_hidden[i, j]
            dot.edge(f"Input_{i}", f"Hidden_{j}", label=f"{weight:.2f}", fontsize="10")

    # Add edges for Hidden -> Output
    for j in range(weights_hidden_output.shape[0]):
        for k in range(weights_hidden_output.shape[1]):
            weight = weights_hidden_output[j, k]
            dot.edge(f"Hidden_{j}", f"Output_{k}", label=f"{weight:.2f}", fontsize="10")

    # Render the graph
    dot.render(f"ffnn_step_{step}", format="png", cleanup=True)
    print(f"Step {step} visualization saved as 'ffnn_step_{step}.png'")
    
# Step 1: Input Layer
visualize_step_graphviz(input_data, step=1)

# Step 2: Hidden Layer Weighted Sum
visualize_step_graphviz(input_data, hidden_weighted_sum=hidden_weighted_sum, step=2)

# Step 3: Hidden Layer Activations
visualize_step_graphviz(input_data, hidden_activations=hidden_activations, step=3)

# Step 4: Output Layer Weighted Sum
visualize_step_graphviz(input_data, hidden_activations=hidden_activations,
                        output_weighted_sum=output_weighted_sum, step=4)

# Step 5: Output Layer Activations
visualize_step_graphviz(input_data, hidden_activations=hidden_activations,
                        output_activations=output_activations, step=5)