import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def linearRegression(X: np.array, Y: np.array, lr: float, lambda_: float):
    """
    Parameters:
    - X: Input feature matrix (NumPy array)
    - Y: Target vector (NumPy array)
    - lr: Learning rate (float)
    - lambda_: L1 regularization coefficient (float)
    
    Returns:
    - weights: Learned model parameters (including bias at index 0)
    """
    # 1. Add a column of 1s to X to handle the bias (intercept) term
    n_samples = X.shape[0]
    X_b = np.c_[np.ones((n_samples, 1)), X] 
    
    # 2. Initialize weights to zeros (weights[0] will be our bias)
    n_features = X_b.shape[1]
    weights = np.zeros(n_features)
    
    epochs = 1000 # Number of iterations for Gradient Descent
    
    # 3. Gradient Descent Loop
    for _ in range(epochs):
        # Calculate current predictions: y = X * w
        y_predicted = np.dot(X_b, weights)
        
        # Calculate the gradient of the Loss function (Mean Squared Error)
        dw = (1 / n_samples) * np.dot(X_b.T, (y_predicted - Y))
        
        # Add L1 Regularization (Lasso) penalty to the gradient
        # The derivative of |w| is sign(w). We do NOT regularize the bias term (weights[0]).
        l1_penalty = lambda_ * np.sign(weights)
        l1_penalty[0] = 0 
        
        dw += l1_penalty
        
        # Update weights
        weights -= lr * dw
        
    return weights

# --- Quick Test / Demonstration ---
if __name__ == "__main__":
    # Generate some dummy linear data
    np.random.seed(42)
    X_test = 2 * np.random.rand(100, 1)
    Y_test = 4 + 3 * X_test.flatten() + np.random.randn(100) # y = 3x + 4 + noise

    # Train the model
    learned_weights = linearRegression(X_test, Y_test, lr=0.1, lambda_=0.01)
    
    print(f"Learned Bias: {learned_weights[0]:.4f} (Expected ~4)")
    print(f"Learned Weight: {learned_weights[1]:.4f} (Expected ~3)")

    # Plotting the result
    plt.scatter(X_test, Y_test, color="blue", label="Data")
    plt.plot(X_test, learned_weights[0] + learned_weights[1]*X_test, color="red", label="Regression Line")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Linear Regression from Scratch")
    plt.legend()
    plt.show()
    
    plt.plot(X_test, learned_weights[0] + learned_weights[1]*X_test, color="red", label="Regression Line")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Linear Regression from Scratch")
    plt.legend()
    
    # Save the plot as an image instead of trying to show it interactively
    plt.savefig("linear_regression_plot.png")
    print("✅ Plot saved successfully as 'linear_regression_plot.png'")
