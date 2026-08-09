import math

# Activation
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Inputs and target
x1, x2, x3 = 1.0, 0.0, 1.0
target = 1.0  
lr = 0.5

# Weights (inputs → hidden1)
w_x1_h1, w_x2_h1, w_x3_h1 = 0.2, 0.4, -0.5
# Weights (inputs → hidden2)
w_x1_h2, w_x2_h2, w_x3_h2 = -0.3, 0.1, 0.2

# Biases for hidden neurons
b_h1, b_h2 = -0.4, 0.2

# Weights (hidden → output)
w_h1_out, w_h2_out = -0.3, -0.2
# Bias for output
b_out = 0.1

# Train for 3 iterations
epochs = 3
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}")

    # Forward pass - hidden layer
    h1 = sigmoid(x1*w_x1_h1 + x2*w_x2_h1 + x3*w_x3_h1 + b_h1)
    h2 = sigmoid(x1*w_x1_h2 + x2*w_x2_h2 + x3*w_x3_h2 + b_h2)

    # Forward pass - output
    y_pred = sigmoid(h1*w_h1_out + h2*w_h2_out + b_out)

    print(f"  h1={h1:.6f}, h2={h2:.6f}, y_pred={y_pred:.6f}")

    # Backpropagation
    d_out = (target - y_pred) * sigmoid_derivative(y_pred)
    d_h1 = sigmoid_derivative(h1) * (d_out * w_h1_out)
    d_h2 = sigmoid_derivative(h2) * (d_out * w_h2_out)

    # Update weights hidden → output
    w_h1_out += lr * d_out * h1
    w_h2_out += lr * d_out * h2
    b_out    += lr * d_out

    # Update weights input → h1
    w_x1_h1 += lr * d_h1 * x1
    w_x2_h1 += lr * d_h1 * x2
    w_x3_h1 += lr * d_h1 * x3
    b_h1    += lr * d_h1

    # Update weights input → h2
    w_x1_h2 += lr * d_h2 * x1
    w_x2_h2 += lr * d_h2 * x2
    w_x3_h2 += lr * d_h2 * x3
    b_h2    += lr * d_h2

    print("  Updated Weights:")
    print("   input→h1:", w_x1_h1, w_x2_h1, w_x3_h1)
    print("   input→h2:", w_x1_h2, w_x2_h2, w_x3_h2)
    print("   hidden→out:", w_h1_out, w_h2_out)
    print("   biases:", b_h1, b_h2, b_out)

# Final results
predicted_class = 1 if y_pred >= 0.5 else 0
closeness = (1 - abs(target - y_pred)) * 100.0

print("\n===== Final Result =====")
print(f"Final predicted value: {y_pred:.6f}")
print(f"Predicted class: {predicted_class}")
print(f"Closeness: {closeness:.2f}%")
