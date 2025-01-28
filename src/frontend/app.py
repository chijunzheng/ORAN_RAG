# src/frontend/app.py

from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
import os
import sys
import logging

# Add the project root to sys.path to ensure modules can be imported
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.authentication.auth_manager import AuthManager
from src.config import load_config, validate_config
from src.chatbot.chatbot import Chatbot
from src.vector_search.searcher import VectorSearcher
from src.vector_search.reranker import Reranker

app = Flask(__name__)

# Configure secret key and session type
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY')
if not app.config['SECRET_KEY']:
    raise ValueError("No SECRET_KEY set for Flask application. Set the FLASK_SECRET_KEY environment variable.")

app.config['SESSION_TYPE'] = 'filesystem'  # Store sessions in the filesystem

# Initialize Session
Session(app)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config_path = os.path.join(project_root, 'configs', 'config.yaml')  # Adjust the path if necessary
config = load_config(config_path)
validate_config(config)

# Initialize Authentication
try:
    auth_manager = AuthManager(config=config)
    auth_manager.authenticate_user()
    logger.info("Authentication successful.")
except Exception as e:
    logger.error(f"Authentication failed: {e}")
    raise

# Initialize VectorSearcher
try:
    vector_searcher = VectorSearcher(
        project_id=config['gcp']['project_id'],
        location=config['gcp']['location'],
        bucket_name=config['gcp']['bucket_name'],
        embeddings_path=config['gcp']['embeddings_path'],
        bucket_uri=config['gcp']['bucket_uri'],
        credentials=auth_manager.get_credentials()
    )
    logger.info("VectorSearcher initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize VectorSearcher: {e}")
    raise

# Initialize Reranker
try:
    ranking_config = config['ranking']
    reranker = Reranker(
        project_id=config['gcp']['project_id'],
        location=config['gcp']['location'],
        ranking_config=ranking_config['ranking_config'],
        credentials=auth_manager.get_credentials(),
        model=ranking_config['model'],
        rerank_top_n=ranking_config['rerank_top_n']
    )
    logger.info("Reranker initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Reranker: {e}")
    raise

# Initialize Chatbot
try:
    chatbot = Chatbot(
        project_id=config['gcp']['project_id'],
        location=config['gcp']['location'],
        bucket_name=config['gcp']['bucket_name'],
        embeddings_path=config['gcp']['embeddings_path'],
        bucket_uri=config['gcp']['bucket_uri'],
        index_endpoint_display_name=config['vector_search']['endpoint_display_name'],
        deployed_index_id=config['vector_search']['deployed_index_id'],
        generation_temperature=config['generation']['temperature'],
        generation_top_p=config['generation']['top_p'],
        generation_max_output_tokens=config['generation']['max_output_tokens'],
        vector_searcher=vector_searcher,
        credentials=auth_manager.get_credentials(),
        num_neighbors=config['vector_search']['num_neighbors'],
        reranker=reranker
    )
    logger.info("Chatbot initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Chatbot: {e}")
    raise

@app.route('/')
def home():
    # Clear conversation history when the home page is accessed
    session.pop('conversation_history', None)
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form.get('message')
    if not user_input:
        logger.warning("No message provided by the user.")
        return jsonify({'error': 'No message provided.'}), 400

    try:
        # Retrieve or initialize conversation history from session
        conversation_history = session.get('conversation_history', [])

        # Get response from chatbot
        response = chatbot.get_response(user_input, conversation_history)

        # Update conversation history
        conversation_history.append({'user': user_input, 'assistant': response})
        session['conversation_history'] = conversation_history

        logger.info(f"User: {user_input}")
        logger.info(f"Chatbot: {response}")

        return jsonify({'response': response})
    except Exception as e:
        logger.error(f"Error getting response from chatbot: {e}", exc_info=True)
        return jsonify({'error': 'Error processing your request.'}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='127.0.0.1', port=5001, debug=True)