import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

class TitanicSurvivalModel:
    """
    A class to handle the machine learning process for the Titanic dataset using Random Forest classifiers.

    This class manages the training of multiple Random Forest models and combines
    their predictions to create a final output.
    """



    def __init__(self, param_train_data: pd.DataFrame, param_test_data: pd.DataFrame, param_features: list, target: str, modelcount: int):
        """
    Initialize the TitanicSurvivalModel instance.

    Args:
        param_train_data (pd.DataFrame): The training data.
        param_test_data (pd.DataFrame): The test data.
        param_features (list): List of feature column names.
        target (str): The target variable name.
        modelcount (int): The number of Random Forest models to create and use.

    Attributes:
        param_train_data (pd.DataFrame): Stored training data.
        param_test_data (pd.DataFrame): Stored test data.
        param_features (list): Stored list of feature column names.
        target (str): Stored target variable name, stripped of brackets and quotes.
        modelcount (int): Number of Random Forest models to use.
        MAXMODELS (int): Maximum allowed number of models (5).
        models (list): List of RandomForestClassifier instances.
        predictions (list): List to store predictions from each model.

    Raises:
        ValueError: If modelcount is greater than MAXMODELS.

    Note:
        The number of models is capped at MAXMODELS (5) to prevent excessive computation.
        If modelcount exceeds MAXMODELS, it will be set to MAXMODELS.
    """
        self.MAXMODELS = 5 # Change with caution!
        self.param_train_data = param_train_data
        self.param_test_data = param_test_data
        self.param_features = param_features
        self.target = target.strip('[]"')
        self.modelcount = min(modelcount, self.MAXMODELS)
        self.models = [RandomForestClassifier(n_estimators=100, max_depth=5, random_state=i) for i in range(self.modelcount)]
        self.predictions = []

        if modelcount > self.MAXMODELS:
            print(f"Warning: modelcount exceeds MAXMODELS. Using {self.MAXMODELS} models instead.")


    def LEARNING_MODULAR(self, model_index):
        """
        Train a single Random Forest model and make predictions.

        Args:
            model_index (int): The index of the model being trained.

        Returns:
            numpy.ndarray: The predictions made by the model, or None if an error occurred.
        """
        try:
            print(f"Starting the learning process for model {model_index}.")
            
            X = pd.get_dummies(self.param_train_data[self.param_features])
            X_test = pd.get_dummies(self.param_test_data[self.param_features])
            
            if self.target not in self.param_train_data.columns:
                raise ValueError(f"Target column '{self.target}' not found in the training data.")
            
            y = self.param_train_data[self.target]
            
            # Ensure X_test has the same columns as X, in the same order
            X_test = X_test.reindex(columns=X.columns, fill_value=0)
            
            self.models[model_index].fit(X, y)
            print(f"Model {model_index} training completed.")
            
            predictions = self.models[model_index].predict(X_test)
            self.predictions.append(predictions)
            print(f"Predictions for model {model_index} completed. Predictions: {len(predictions)}")
            
            return predictions
        except Exception as e:
            print(f"An error occurred in model {model_index}: {str(e)}")
            return None

    def run_parallel(self):
        """
        Run the machine learning process for all models.

        This method trains all models sequentially (using a single thread) and combines their predictions
        to create a final output, which is saved to a CSV file.
        """
        with ThreadPoolExecutor(max_workers=self.MAXMODELS) as executor:
            futures = [executor.submit(self.LEARNING_MODULAR, i) for i in range(self.modelcount)] 
            for future in as_completed(futures):
                future.result()
        
        if not self.predictions:
            print("No valid predictions were made. Check your input data and try again.")
            return

        combined_predictions = np.mean(self.predictions, axis=0)
        final_predictions = (combined_predictions > 0.5).astype(int)
        
        # Ensure we're using all rows from the test data
        passenger_ids = self.param_test_data['PassengerId'].values
        
        if len(passenger_ids) != len(final_predictions):
            print(f"Warning: Mismatch in number of predictions ({len(final_predictions)}) and PassengerIds ({len(passenger_ids)})")
        
        # Create DataFrame ensuring all PassengerIds are included
        output = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': final_predictions[:len(passenger_ids)]})
        
        # Sort by PassengerId to ensure correct order
        output = output.sort_values('PassengerId')
        
        output.to_csv('prediction.csv', index=False)
        print(f"Final predictions saved successfully to 'prediction.csv'. Predictions: {len(output)}")

def main():
    """
    Main function to run the Titanic survival prediction model.

    This function loads the training and test data, initializes the TitanicSurvivalModel class,
    and runs the prediction process.
    """
    # Load data
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    features = ['Pclass', 'Sex', 'SibSp', 'Parch']
    target = 'Survived'

    model = TitanicSurvivalModel(train_data, test_data, features, target,modelcount=1)
    model.run_parallel()

if __name__ == "__main__":
    main()
