import numpy as np

def train_linear_svr(X, y, epsilon=0.05, lam=1e-4, lr=0.01, epochs=1000):
    n_samples, n_features = X.shape
    
    # Initialize weights and bias to zeros
    w = np.zeros(n_features)
    b = 0.0
    loss_history = []
    
    for t in range(epochs):
        # 1. Predict current values
        predictions = np.matmul(X, w) + b
        
        # 2. Calculate error (Actual - Predicted)
        errors = y - predictions
        
        # 3. Find points outside the epsilon-tube (|error| > epsilon)
        outside_mask = np.abs(errors) > epsilon
        
        # 4. Get direction (+1 or -1) for points outside, 0 for points inside
        error_signs = np.sign(errors) * outside_mask
        
        # 5. Calculate slopes (gradients)
        grad_w = (lam * w) - (np.matmul(X.T, error_signs) / n_samples)
        grad_b = -np.mean(error_signs)
        
        # 6. Update weights and bias
        w -= lr * grad_w
        b -= lr * grad_b

        # 7. Track loss (Regularization Loss + Epsilon-Insensitive Loss)
        margin_loss = np.maximum(0, np.abs(errors) - epsilon).mean()
        reg_loss = 0.5 * lam * np.sum(w ** 2)
        loss_history.append(reg_loss + margin_loss)
        
    return {'weights': w, 'bias': b, 'loss_history': loss_history}

def predict_linear_svr(model, X):
    return np.matmul(X, model['weights']) + model['bias']