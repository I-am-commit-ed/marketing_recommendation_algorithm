import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model
model_path = "/Users/manuel/Documents/GitHub/JeanPierreWeill/ML/product_recommendation_model.h5"
model = load_model(model_path)

# Define the recommendation function
def recommend_next_products(user_purchase_history, encoder_title, max_sequence_length, top_k=5):
    """
    Generate product recommendations based on user purchase history.

    Args:
        user_purchase_history (list): A list of product title indices from the user's purchase history.
        encoder_title (LabelEncoder): Encoder used to encode product titles.
        max_sequence_length (int): Maximum sequence length for input padding.
        top_k (int): Number of top recommendations to return.

    Returns:
        recommendations (list): List of recommended product titles.
    """
    # Encode and pad the input sequence
    encoded_sequence = encoder_title.transform(user_purchase_history)
    padded_sequence = pad_sequences([encoded_sequence], maxlen=max_sequence_length, padding='pre')

    # Predict the probabilities for the next product
    predicted_probabilities = model.predict(padded_sequence)
    top_indices = np.argsort(predicted_probabilities[0])[-top_k:][::-1]

    # Decode the recommended product indices
    recommendations = encoder_title.inverse_transform(top_indices)
    return recommendations
