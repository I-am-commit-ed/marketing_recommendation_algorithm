from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.regularizers import l2

app = Flask(__name__)
CORS(app)

class ProductRecommender:
    def __init__(self, model_path=None):
        self.model = None
        self.encoder = None
        self.max_sequence_length = None
        if model_path:
            self.load_model(model_path)
        else:
            self.build_model()
            
    def build_model(self):
        """Build an improved recommendation model"""
        num_classes = self._get_num_classes()  # You'll need to implement this based on your data
        max_sequence_length = 10  # Adjust based on your needs
        embedding_dim = 256
        
        model = Sequential([
            Embedding(input_dim=num_classes, 
                     output_dim=embedding_dim, 
                     input_length=max_sequence_length,
                     embeddings_regularizer=l2(1e-4)),
            Dropout(0.3),
            Bidirectional(LSTM(256, return_sequences=True)),
            Bidirectional(LSTM(128, return_sequences=False)),
            Dropout(0.3),
            Dense(512, activation='relu', kernel_regularizer=l2(1e-4)),
            Dropout(0.3),
            Dense(256, activation='relu', kernel_regularizer=l2(1e-4)),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model

    def _initialize_encoder(self):
        """Initialize the label encoder with your product data"""
        try:
            sales_data = pd.read_csv("/path/to/your/preprocessed_sales_data.csv")
            self.encoder = LabelEncoder()
            self.encoder.fit(sales_data['product_title'].unique())
            self.max_sequence_length = 10  # Adjust based on your needs
        except Exception as e:
            print(f"Error initializing encoder: {e}")
            raise

    def preprocess_input(self, purchase_history):
        """Preprocess the input purchase history."""
        try:
            # Convert product names to encoded indices
            encoded_products = self.encoder.transform(purchase_history)
            # Pad sequence
            padded_sequence = pad_sequences([encoded_products], 
                                         maxlen=self.max_sequence_length,
                                         padding='pre')
            return padded_sequence
        except Exception as e:
            print(f"Error preprocessing input: {e}")
            raise

    def get_recommendations(self, purchase_history, top_k=5):
        """Generate product recommendations based on purchase history."""
        try:
            # Preprocess input
            padded_sequence = self.preprocess_input(purchase_history)
            
            # Get predictions
            predictions = self.model.predict(padded_sequence)
            
            # Get top k recommendations
            top_indices = np.argsort(predictions[0])[-top_k:][::-1]
            recommendations = self.encoder.inverse_transform(top_indices)
            
            return recommendations.tolist()
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            raise

    def load_model(self, model_path):
        """Load a pre-trained model"""
        try:
            self.model = load_model(model_path)
            self._initialize_encoder()
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def _get_num_classes(self):
        """Get the number of unique products in your dataset"""
        try:
            sales_data = pd.read_csv("/path/to/your/preprocessed_sales_data.csv")
            return len(sales_data['product_title'].unique())
        except Exception as e:
            print(f"Error getting number of classes: {e}")
            raise

# Initialize recommender
try:
    recommender = ProductRecommender("/path/to/your/product_recommendation_model.h5")
except Exception as e:
    print(f"Failed to initialize recommender: {e}")

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        purchase_history = data.get('purchase_history', '').split(',')
        purchase_history = [p.strip() for p in purchase_history if p.strip()]
        
        if not purchase_history:
            return jsonify({
                'error': 'Please provide a valid purchase history'
            }), 400
            
        recommendations = recommender.get_recommendations(purchase_history)
        return jsonify({
            'recommendations': recommendations
        })
    except Exception as e:
        return jsonify({
            'error': f'Error generating recommendations: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)