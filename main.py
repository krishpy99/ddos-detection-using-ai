from train_model import DDoSDetectionModel
from infer_model import DDoSDetector

def train_ddos_model(dataset_path, model_save_path):
    try:
        # Initialize model
        detector = DDoSDetectionModel()
        
        # Prepare data
        print("Loading and preparing data...")
        X_train, X_test, y_train, y_test = detector.prepare_data(dataset_path)
        
        # Train model
        print("Training model...")
        detector.train(X_train, y_train)
        
        # Evaluate model
        print("Evaluating model...")
        metrics = detector.evaluate(X_test, y_test)
        
        # Save model
        print("Saving model...")
        detector.save_model(model_save_path)
        
        return metrics
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return None

def detect_ddos(model_path, network_data):
    detector = DDoSDetector(model_path)
    result = detector.predict(network_data)
    return result

if __name__ == "__main__":
    dataset_path = 'data.csv'
    model_save_path = 'ddos_detection_model.joblib'
    
    metrics = train_ddos_model(dataset_path, model_save_path)
    if metrics:
        print("\nTraining Results:")
        print("\nClassification Report:")
        print(metrics['classification_report'])
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        print(f"\nCross-validation Score: {metrics['cross_val_score']:.4f}")