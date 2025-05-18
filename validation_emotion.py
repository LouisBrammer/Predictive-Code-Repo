import tensorflow as tf
import os

def load_model(model_path='model.keras'):
    try:
        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Load the model
        model = tf.keras.models.load_model(model_path)
        print(f"Successfully loaded model from {model_path}")
        return model
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

if __name__ == "__main__":
    # Load the model
    model = load_model()


