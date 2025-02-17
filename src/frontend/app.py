# src/frontend/app.py

import os
import sys
import logging
from flask import Flask, render_template, request, jsonify, session
from flask_session import Session

# Add the project root to sys.path so that modules can be imported.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.authentication.auth_manager import AuthManager
from src.config import load_config, validate_config
from src.chatbot.chatbot import Chatbot
from src.vector_search.searcher import VectorSearcher
from src.vector_search.reranker import Reranker
from src.yang_pipeline import process_yang_dir
from src.embeddings.embedder import Embedder  # for triggering re-embedding (if needed)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY')
if not app.config['SECRET_KEY']:
    raise ValueError("No SECRET_KEY set for Flask application. Set the FLASK_SECRET_KEY environment variable.")
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config_path = os.path.join(project_root, 'configs', 'config.yaml')
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

# (Optional) If you want to reprocess YANG files at startup, call process_yang_dir
# (Typically this step is run in a separate pipeline; here we add an endpoint to trigger it.)
# For example:
# yang_chunks = process_yang_dir(config['paths']['yang_dir'])
# Optionally, run your embedder to rebuild embeddings for YANG files if needed.
# embedder = Embedder(
#     project_id=config['gcp']['project_id'],
#     location=config['gcp']['location'],
#     bucket_name=config['gcp']['bucket_name'],
#     embeddings_path=config['gcp']['embeddings_path'],
#     credentials=auth_manager.get_credentials()
# )
# embedder.generate_and_store_embeddings(yang_chunks)

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

# Initialize Chatbot (which now builds prompts using the complete metadata)
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
<<<<<<< HEAD
    # Clear conversation history when the home page is accessed
=======
    # Clear conversation history when the home page is accessed.
>>>>>>> gpl_generation
    session.pop('conversation_history', None)
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form.get('message')
    if not user_input:
        logger.warning("No message provided by the user.")
        return jsonify({'error': 'No message provided.'}), 400

    try:
        # Retrieve or initialize conversation history.
        conversation_history = session.get('conversation_history', [])

        # Get response from chatbot.
        response = chatbot.get_response(user_input, conversation_history)

        # Update conversation history.
        conversation_history.append({'user': user_input, 'assistant': response})
        session['conversation_history'] = conversation_history

        logger.info(f"User: {user_input}")
        logger.info(f"Chatbot: {response}")
        return jsonify({'response': response})
    except Exception as e:
        logger.error(f"Error getting response from chatbot: {e}", exc_info=True)
        return jsonify({'error': 'Error processing your request.'}), 500

# (Optional) Provide an endpoint to reprocess the YANG files and rebuild embeddings.
@app.route('/reprocess_yang', methods=['POST'])
def reprocess_yang():
    try:
        yang_dir = config['paths']['yang_dir']
        logger.info(f"Reprocessing YANG files in directory: {yang_dir}")
        # Process YANG files to produce new chunks (with the new metadata including chunk_index).
        yang_chunks = process_yang_dir(yang_dir)
        logger.info(f"Reprocessed {len(yang_chunks)} YANG chunks.")

        # Optionally, trigger re-embedding.
        embedder = Embedder(
            project_id=config['gcp']['project_id'],
            location=config['gcp']['location'],
            bucket_name=config['gcp']['bucket_name'],
            embeddings_path=config['gcp']['embeddings_path'],
            credentials=auth_manager.get_credentials()
        )
        embedder.generate_and_store_embeddings(yang_chunks)
        logger.info("Re-embedded YANG chunks successfully.")
        return jsonify({'message': 'YANG files reprocessed and embeddings updated.'})
    except Exception as e:
        logger.error(f"Error reprocessing YANG files: {e}", exc_info=True)
        return jsonify({'error': 'Failed to reprocess YANG files.'}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True)