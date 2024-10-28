import numpy as np
import joblib

class DDoSDetector:
    def __init__(self, model_path):
        # Load the saved model and preprocessors
        saved_model = joblib.load(model_path)
        self.model = saved_model['model']
        self.scaler = saved_model['scaler']
        self.label_encoder = saved_model['label_encoder']
        
    def predict(self, network_data):
        # Preprocess the input data
        scaled_data = self.scaler.transform(network_data)
        
        # Make prediction
        prediction = self.model.predict(scaled_data)
        probability = self.model.predict_proba(scaled_data)
        
        # Decode the prediction
        attack_type = self.label_encoder.inverse_transform(prediction)
        
        return {
            'prediction': attack_type[0],
            'confidence': np.max(probability[0]),
            'is_attack': attack_type[0] != 'BENIGN'
        }