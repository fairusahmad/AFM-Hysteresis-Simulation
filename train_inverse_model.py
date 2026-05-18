"""
train_inverse_model.py - Train AI inverse model for hysteresis compensation
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def load_data(filename="preprocessed_data_inverse.npz"):
    """Load preprocessed data"""
    if not os.path.exists(filename):
        print(f"File {filename} does not exist, please run data preprocessing first")
        return None
    
    data = np.load(filename)
    return {
        'X_train': data['X_train'],
        'X_test': data['X_test'],
        'y_train': data['y_train'],
        'y_test': data['y_test']
    }

def train_mlp_model(data, hidden_layers=(50, 25, 10), max_iter=2000):
    """
    Train MLP neural network
    
    hidden_layers: number of neurons in hidden layers, e.g., (50, 25) for two layers
    max_iter: maximum number of iterations
    """
    print("\n" + "="*50)
    print("Training MLP inverse model")
    print("="*50)
    print(f"Hidden layer structure: {hidden_layers}")
    print(f"Max iterations: {max_iter}")
    
    # Create MLP regressor
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation='relu',
        solver='adam',
        max_iter=max_iter,
        random_state=42,
        verbose=True
    )
    
    # Train model
    print("\nStarting training...")
    model.fit(data['X_train'], data['y_train'])
    
    # Predict
    y_train_pred = model.predict(data['X_train'])
    y_test_pred = model.predict(data['X_test'])
    
    # Evaluate
    train_mse = mean_squared_error(data['y_train'], y_train_pred)
    test_mse = mean_squared_error(data['y_test'], y_test_pred)
    train_r2 = r2_score(data['y_train'], y_train_pred)
    test_r2 = r2_score(data['y_test'], y_test_pred)
    
    print(f"\nTraining MSE: {train_mse:.6f}")
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    
    return model, {'train_mse': train_mse, 'test_mse': test_mse, 
                   'train_r2': train_r2, 'test_r2': test_r2}

def plot_training_results(data, model):
    """Plot training results comparison"""
    y_train_pred = model.predict(data['X_train'])
    y_test_pred = model.predict(data['X_test'])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Training set predicted vs actual
    axes[0, 0].scatter(data['y_train'], y_train_pred, alpha=0.5, s=1)
    axes[0, 0].plot([data['y_train'].min(), data['y_train'].max()], 
                    [data['y_train'].min(), data['y_train'].max()], 'r--')
    axes[0, 0].set_xlabel('Actual')
    axes[0, 0].set_ylabel('Predicted')
    axes[0, 0].set_title('Training set: Predicted vs Actual')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Test set predicted vs actual
    axes[0, 1].scatter(data['y_test'], y_test_pred, alpha=0.5, s=1)
    axes[0, 1].plot([data['y_test'].min(), data['y_test'].max()], 
                    [data['y_test'].min(), data['y_test'].max()], 'r--')
    axes[0, 1].set_xlabel('Actual')
    axes[0, 1].set_ylabel('Predicted')
    axes[0, 1].set_title('Test set: Predicted vs Actual')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Error distribution
    train_error = y_train_pred - data['y_train']
    test_error = y_test_pred - data['y_test']
    axes[1, 0].hist(train_error, bins=50, alpha=0.5, label='Training', color='blue')
    axes[1, 0].hist(test_error, bins=50, alpha=0.5, label='Test', color='orange')
    axes[1, 0].set_xlabel('Prediction error')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Error distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Model performance summary
    axes[1, 1].text(0.1, 0.5, f'Training R² = {r2_score(data["y_train"], y_train_pred):.4f}\nTest R² = {r2_score(data["y_test"], y_test_pred):.4f}', 
                    fontsize=14, transform=axes[1, 1].transAxes)
    axes[1, 1].set_title('Model performance')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

def test_compensation(model, scaler_X, scaler_y):
    """
    Test the compensation effect of the inverse model
    Simulation: given desired position → model predicts command → simulate hysteresis → check actual position
    """
    print("\n" + "="*50)
    print("Testing inverse model compensation effect")
    print("="*50)
    
    # Import hysteresis model
    from hysteresis import NanoPositioner
    
    # Create positioner
    stage = NanoPositioner(r=30)
    stage.reset(0, 0)
    
    # Test points: sequence of desired positions
    desired_positions = np.linspace(200, 1400, 50)
    
    # Without compensation: command directly
    cmd_no_comp = desired_positions
    actual_no_comp = []
    for cmd in cmd_no_comp:
        actual, _ = stage.move_to(cmd, 0)
        actual_no_comp.append(actual)
    
    # With compensation: use AI model to predict command
    stage.reset(0, 0)
    cmd_with_comp = []
    actual_with_comp = []
    
    # Initialize historical features
    prev_actual = 0
    prev_prev_actual = 0
    
    for i, desired in enumerate(desired_positions):
        # Build features (need historical data)
        if i == 0:
            actual_prev_1 = desired
            actual_prev_2 = desired
            velocity = 0
        elif i == 1:
            actual_prev_1 = desired_positions[i-1]
            actual_prev_2 = desired_positions[i-1]
            velocity = desired - desired_positions[i-1]
        else:
            actual_prev_1 = desired_positions[i-1]
            actual_prev_2 = desired_positions[i-2]
            velocity = desired - desired_positions[i-1]
        
        # Direction
        direction = 1 if velocity >= 0 else -1
        
        # Build feature vector
        features = np.array([[desired, actual_prev_1, actual_prev_2, direction, velocity]])
        features_scaled = scaler_X.transform(features)
        
        # Predict command
        cmd_pred = scaler_y.inverse_transform(model.predict(features_scaled).reshape(-1, 1)).ravel()[0]
        cmd_with_comp.append(cmd_pred)
        
        # Apply hysteresis
        actual, _ = stage.move_to(cmd_pred, 0)
        actual_with_comp.append(actual)
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Without vs with compensation
    axes[0].plot(desired_positions, desired_positions, 'k--', label='Ideal (1:1)')
    axes[0].plot(desired_positions, actual_no_comp, 'b-', label='Without compensation', alpha=0.7)
    axes[0].plot(desired_positions, actual_with_comp, 'r-', label='With AI compensation', alpha=0.7)
    axes[0].set_xlabel('Desired position (um)')
    axes[0].set_ylabel('Actual position (um)')
    axes[0].set_title('Compensation effect comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')
    
    # Plot 2: Error comparison
    error_no_comp = np.array(actual_no_comp) - desired_positions
    error_with_comp = np.array(actual_with_comp) - desired_positions
    axes[1].plot(desired_positions, error_no_comp, 'b-', label='Without compensation', alpha=0.7)
    axes[1].plot(desired_positions, error_with_comp, 'r-', label='With AI compensation', alpha=0.7)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Desired position (um)')
    axes[1].set_ylabel('Error (um)')
    axes[1].set_title('Error comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Statistical indicators
    rmse_no_comp = np.sqrt(np.mean(error_no_comp**2))
    rmse_with_comp = np.sqrt(np.mean(error_with_comp**2))
    max_error_no_comp = np.max(np.abs(error_no_comp))
    max_error_with_comp = np.max(np.abs(error_with_comp))
    
    axes[2].bar(['Without compensation', 'With AI compensation'], [rmse_no_comp, rmse_with_comp], color=['blue', 'red'])
    axes[2].set_ylabel('RMSE (um)')
    axes[2].set_title(f'RMSE: {rmse_no_comp:.2f} → {rmse_with_comp:.2f}')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nRMSE without compensation: {rmse_no_comp:.2f} um")
    print(f"RMSE with AI compensation: {rmse_with_comp:.2f} um")
    print(f"Error reduction: {(1 - rmse_with_comp/rmse_no_comp)*100:.1f}%")

def save_model(model, scaler_X, scaler_y, filename="inverse_model.pkl"):
    """Save model and scalers"""
    model_data = {
        'model': model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y
    }
    joblib.dump(model_data, filename)
    print(f"\nModel saved to: {filename}")

if __name__ == "__main__":
    print("="*60)
    print("Training AI inverse model")
    print("="*60)
    
    # 1. Load data
    print("\nLoading preprocessed data...")
    data = load_data("preprocessed_data_inverse.npz")
    
    if data is None:
        print("Please run preprocess_data.py first to preprocess data")
        exit()
    
    print(f"Training set: {len(data['X_train'])} samples")
    print(f"Test set: {len(data['X_test'])} samples")
    
    # 2. Train model
    model, metrics = train_mlp_model(data, hidden_layers=(100, 50, 25), max_iter=2000)
    
    # 3. Plot training results
    plot_training_results(data, model)
    
    # 4. Note about scalers
    print("\n" + "="*50)
    print("Note: Full testing requires scaler objects")
    print("Please modify preprocessing script to save scalers, or proceed to next integration step")
    print("="*50)
    
    # 5. Save model
    # Since scalers are not available, we only save the model for now
    # save_model(model, None, None, "inverse_model.pkl")
    print("\nModel training complete!")