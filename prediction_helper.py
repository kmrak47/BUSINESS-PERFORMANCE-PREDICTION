import numpy as np

def predict_business_performance(model, input_data):
    """
    Helper function to predict business performance using the trained model.

    Args:
    model: Trained model loaded using joblib.
    input_data: Array of input features for prediction.

    Returns:
    Predicted business performance category.
    """
    try:
        prediction = model.predict(input_data)
        performance_labels = ['Very Poor', 'Poor', 'Average', 'Good', 'Best']
        return performance_labels[prediction[0]]
    except AttributeError:
        return "Model is not valid. Please check the model file."
