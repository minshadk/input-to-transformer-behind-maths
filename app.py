import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Set a fixed seed for reproducible math
np.random.seed(42)

# Define simplified dimensions for visualization
D_MODEL = 4
D_K = 4

# Initialize random learned weight matrices (4x4)
W_Q = np.random.randn(D_MODEL, D_K)
W_K = np.random.randn(D_MODEL, D_K)
W_V = np.random.randn(D_MODEL, D_K)

vocab = {}

def get_embedding(token):
    if token not in vocab:
        vocab[token] = np.random.randn(D_MODEL)
    return vocab[token]

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.json or {}
    text = data.get('text', '')
    if not text or not text.strip():
        return jsonify({'error': 'Please enter at least one word to visualize.'}), 400
        
    # Step 1: Tokenization
    tokens = text.strip().split()
    if len(tokens) > 20:
        return jsonify({'error': 'Please enter 20 words or fewer for clear visualization.'}), 400
    
    # Step 2: Input Embeddings
    embeddings = np.array([get_embedding(t) for t in tokens])
    
    # Step 3: Query, Key, Value Projections
    Q = np.dot(embeddings, W_Q)
    K = np.dot(embeddings, W_K)
    V = np.dot(embeddings, W_V)
    
    # Step 4: Attention Scores (Dot Product of Q and transposed K)
    scores = np.dot(Q, K.T)
    
    # Step 5: Scaling
    scaled_scores = scores / np.sqrt(D_K)
    
    # Step 6: Softmax (Attention Weights)
    attention_weights = softmax(scaled_scores)
    
    # Step 7: Final Output (Context Vectors)
    output = np.dot(attention_weights, V)
    
    def format_matrix(matrix):
        return np.round(matrix, 3).tolist()
        
    return jsonify({
        'tokens': tokens,
        'd_model': D_MODEL,
        'd_k': D_K,
        'W_Q': format_matrix(W_Q),
        'W_K': format_matrix(W_K),
        'W_V': format_matrix(W_V),
        'embeddings': format_matrix(embeddings),
        'Q': format_matrix(Q),
        'K': format_matrix(K),
        'V': format_matrix(V),
        'scores': format_matrix(scores),
        'scaled_scores': format_matrix(scaled_scores),
        'attention_weights': format_matrix(attention_weights),
        'output': format_matrix(output)
    })

if __name__ == '__main__':
    app.run(debug=True)