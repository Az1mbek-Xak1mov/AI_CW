import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

class RiskPredictor:
    def __init__(self):
        """
        Initializes the RiskPredictor with a Random Forest model.
        Features are set based on the defined student dataset.
        """
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_cols = [
            'workload_hours',
            'available_hours',
            'stress_level',
            'confidence_level',
            'missed_sessions'
        ]
        self.is_trained = False

    def _generate_synthetic_data(self, n_samples=500) -> pd.DataFrame:
        """
        Generates a synthetic dataset for student risk evaluation.
        The risk level is determined by compounding logic on simulated attributes.
        """
        np.random.seed(42)
        
        # Synthetic generation ranges
        workload_hours = np.random.randint(10, 60, n_samples)
        available_hours = np.random.randint(10, 60, n_samples)
        stress_level = np.random.randint(1, 11, n_samples)
        confidence_level = np.random.randint(1, 11, n_samples)
        missed_sessions = np.random.randint(0, 10, n_samples)
        
        # Risk logic formulation
        # High workload relative to availability significantly increases risk
        workload_ratio = workload_hours / (available_hours + 1)
        
        # Compute a raw risk score for the synthetic student
        # adding normal noise to prevent perfect deterministic splits during training
        raw_risk_score = (
            workload_ratio * 3.5 +
            stress_level * 1.2 -
            confidence_level * 1.5 +
            missed_sessions * 2.0 +
            np.random.normal(0, 2, n_samples) 
        )
        
        # Discretize the continuous risk score into 0 (Low), 1 (Medium), 2 (High) risk quantiles
        p33 = np.percentile(raw_risk_score, 33)
        p66 = np.percentile(raw_risk_score, 66)
        
        risk_level = np.zeros(n_samples, dtype=int)
        risk_level[raw_risk_score > p33] = 1
        risk_level[raw_risk_score > p66] = 2
        
        df = pd.DataFrame({
            'workload_hours': workload_hours,
            'available_hours': available_hours,
            'stress_level': stress_level,
            'confidence_level': confidence_level,
            'missed_sessions': missed_sessions,
            'risk_level': risk_level
        })
        return df

    def train_and_evaluate(self):
        """
        Generates data, splits it appropriately, trains the model, and prints performance metrics.
        """
        df = self._generate_synthetic_data()
        
        X = df[self.feature_cols]
        y = df['risk_level']
        
        # Split logic
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        
        print(f"Total Dataset Shape: {df.shape}")
        print("--- Evaluation Metrics ---")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=["Low Risk (0)", "Medium Risk (1)", "High Risk (2)"]))
        
        # Explainability: Feature importances extraction
        print("\n--- Feature Importances ---")
        importances = self.model.feature_importances_
        feature_imp = pd.Series(importances, index=self.feature_cols).sort_values(ascending=False)
        for feature, imp in feature_imp.items():
            print(f"{feature}: {imp:.4f}")

    def predict_student(self, student_data: dict) -> int:
        """
        Predicts the risk level for a single student profile.
        
        Args:
            student_data: Dictionary containing keys corresponding to all `feature_cols`.
            
        Returns:
            An integer (0, 1, or 2) denoting the predicted risk level.
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call train_and_evaluate() first.")
            
        # Standardize using a single row dataframe
        df = pd.DataFrame([student_data])
        
        # Ensure correct column order is passed to sklearn
        df = df[self.feature_cols]
        return self.model.predict(df)[0]

if __name__ == "__main__":
    predictor = RiskPredictor()
    predictor.train_and_evaluate()
    
    sample_student = {
        'workload_hours': 55,         
        'available_hours': 15,        
        'stress_level': 9,            
        'confidence_level': 2,        
        'missed_sessions': 6
    }
    
    prediction = predictor.predict_student(sample_student)
    risk_map = {0: "Low", 1: "Medium", 2: "High"}
    
    print("\n--- Testing Single Inference ---")
    print(f"Sample Student Predicts: Risk Level {prediction} ({risk_map.get(prediction)})")
