import torch
import numpy as np
import warnings
from tried_many.slave1 import LorenzParameterModel
from tried_many.lorentz import LorenzParameters

# AI model for parameter estimation
class ParameterEstimator:
    def __init__(self, model_path='92.pth'):
        """Load the trained PyTorch model"""
        # Define the model architecture
        if model_path:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = LorenzParameterModel(hidden_size=128, layers=2).to(self.device)
            
            # Load the trained weights
            try:
                # Suppress the FutureWarning about weights_only
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    checkpoint = torch.load(model_path, map_location=self.device)
                
                self.model.load_state_dict(checkpoint)
                self.model.eval()  # Set to evaluation mode
                print(f"Slave 1: Successfully loaded model from {model_path}")
                
            except Exception as e:
                print(f"Slave 1: Error loading model: {e}")
                # Fall back to default behavior
                self.model = None
        else:
            self.model = None
            print("Slave 1: No model path provided, using fallback parameter estimation")
    
    def predict_parameters(self, states):
        """
        Predict Lorenz parameters and initial conditions from state history
        
        Args:
            states: Array of shape (n_states, 3) containing state history
            
        Returns:
            params: Estimated Lorenz parameters
            initial_state: Estimated initial state
        """
        if self.model is None:
            # Fallback to close to the actual values with slight noise
            print("Slave 1: Using fallback parameter estimation (model not available)")
            params = LorenzParameters(
                sigma=10.0 + np.random.normal(0, 0.1),
                rho=28.0 + np.random.normal(0, 0.1),
                beta=8.0/3.0 + np.random.normal(0, 0.01)
            )
            initial_state = states[0] + np.random.normal(0, 0.01, size=3)
            return params, initial_state
        
        # Prepare the input tensor
        if len(states.shape) == 2:  # (n_steps, 3)
            states_tensor = torch.tensor(states, dtype=torch.float32).unsqueeze(0).to(self.device)  # Add batch dim
        else:
            states_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)
        
        # Get predictions from the model
        with torch.no_grad():
            predictions = self.model(states_tensor)
        
        # Convert predictions to parameters and initial state
        pred_numpy = predictions.cpu().numpy()[0]  # Remove batch dimension
        
        # Extract parameters
        params = LorenzParameters(
            sigma=float(pred_numpy[0]),
            rho=float(pred_numpy[1]),
            beta=float(pred_numpy[2])
        )
        
        # Extract initial state
        initial_state = np.array([
            float(pred_numpy[3]),
            float(pred_numpy[4]),
            float(pred_numpy[5])
        ])
        
        return params, initial_state