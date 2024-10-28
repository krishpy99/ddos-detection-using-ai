import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
from sklearn.model_selection import cross_val_score
import joblib

class DDoSDetectionModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
    def prepare_data(self, dataset_path):
        # Load and preprocess data
        data = pd.read_csv(dataset_path)
        
        # Remove any infinite or null values
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Encode labels
        if 'Label' in data.columns:
            y = self.label_encoder.fit_transform(data['Label'])
            X = data.drop(['Label'], axis=1)
        else:
            raise ValueError('Label column not found in dataset')
            
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
    def train(self, X_train, y_train):
        # Train the model
        self.model.fit(
            X_train, 
            y_train,
            eval_set=[(X_train, y_train)],
            verbose=True
        )
        
    def evaluate(self, X_test, y_test):
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate and return metrics
        return {
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'cross_val_score': np.mean(cross_val_score(self.model, X_test, y_test, cv=5))
        }
        
    def save_model(self, path):
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder
        }, path)