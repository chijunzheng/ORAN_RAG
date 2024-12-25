# ORAN_RAG

## Table of Contents

- [ORAN\_RAG](#oran_rag)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Usage](#usage)
    - [Command-Line Arguments](#command-line-arguments)
      - [Available Flags:](#available-flags)
    - [Usage Examples](#usage-examples)
  - [Features](#features)
  - [Evaluation](#evaluation)
    - [Steps](#steps)

## Introduction

**ORAN RAG** is a comprehensive Retrieval-Augmented Generation (RAG) system tailored for O-RAN (Open Radio Access Network) documents. It leverages Google Cloud's Vertex AI and various other GCP services to process, index, and enable intelligent querying over a large corpus of O-RAN-related documents. The system includes functionalities for document preprocessing, chunking, embedding generation, vector indexing, chatbot interaction, and model evaluation.

## Prerequisites

Before setting up and running the ORAN RAG project, ensure that you have the following prerequisites in place:

1. **Google Cloud Platform (GCP) Setup:**
   - **Google Cloud Project:** Create a Google Cloud project.
   - **APIs Enabled:** Enable the following APIs in your GCP project:
     - Vertex AI API
     - Cloud Storage API
     - Firestore API
     - Any other necessary APIs as per project requirements.
   - **IAM Service Account Permissions:** Create a service account with the following roles:
     - **Vertex AI Administrator**
     - **Storage Admin**
     - **Cloud Datastore Owner**
     - **User**
     - **All Firestore Roles**
   - **Service Account Key:** Download the service account key in JSON format. This key will be used for project authentication.

2. **Local Environment Setup:**
   - **Python 3.8+** installed on your machine.
   - **pip** package manager.

## Installation

1. **Clone the Repository:**
```bash
git clone https://github.com/chijunzheng/ORAN_RAG.git
cd ORAN_RAG
```
2.	**Install dependencies**
```bash
pip install -r requirements.txt
```
## Configuration

**config.yaml**

The config.yaml file is the central configuration file that dictates the behavior of the ORAN RAG pipeline. It includes settings for Google Cloud Platform (GCP), file paths, vector search parameters, chunking configurations, generation parameters, logging, and evaluation settings.

**Configuration Structure:**
```yaml
gcp:
  project_id: "oranrag"
  location: "us-central1"
  bucket_name: "oran-rag-bucket-us-central"
  embeddings_path: "embeddings/"
  bucket_uri: "gs://oran-rag-bucket-us-central/embeddings/"
  qna_dataset_path: "dataset/Q&A_ds_H.json"
  credentials_path: "/path/to/credentials.json"

paths:
  documents: "/path/to/Documents/"
  embeddings_save_path: "/path/to/Embeddings/"

vector_search:
  index_display_name: "oran_rag_index"  # Name for the index creation
  endpoint_display_name: "oran_rag_index_endpoint"  # Name for the index deployment
  deployed_index_id: "oran_rag_index_deployed" # Name for the deployed index

  # Parameters for index creation
  index_description: "Index for O-RAN document embeddings"
  dimensions: 768
  approximate_neighbors_count: 5
  leaf_node_embedding_count: 500
  leaf_nodes_to_search_percent: 7
  distance_measure_type: "COSINE_DISTANCE"
  feature_norm_type: "UNIT_L2_NORM"
  shard_size: "SHARD_SIZE_SMALL"

  # Parameters for index deployment
  machine_type: "e2-standard-2"
  min_replica_count: 2
  max_replica_count: 2

  # Parameters for search
  num_neighbors: 10

chunking:
  chunk_size: 1536  # Maximum number of characters per chunk
  chunk_overlap: 256  # Number of overlapping characters between chunks
  separators: [". ", "? ", "! ", "\n\n"]
  min_char_count: 100

generation:
  temperature: 0.7
  top_p: 0.9
  max_output_tokens: 2000

logging:
  log_file: "/path/to/log/oran_rag.log"  # User-defined path for log files

evaluation:
  num_questions: 3243
  excel_file_path: "/path/to/Evaluation/evaluation_results.xlsx"
  max_workers: 1
  plot_save_path: "/path/to/Evaluation/accuracy_plots.png"
  ```

**Key Sections:**
- **gcp:** Contains GCP-related configurations such as project ID, location, bucket names, paths for embeddings, Q&A datasets, and service account credentials.
- **paths:** Specifies local paths for storing documents and embeddings.
- **vector_search:** Configures parameters related to vector indexing and searching, including index names, descriptions, dimensions, and deployment settings.
- **chunking:** Defines how documents are split into chunks, including size, overlap, separators, and minimum character count.
- **generation:** Sets parameters for text generation models like temperature, top-p, and maximum output tokens.
- **logging:** Specifies the path for log files to monitor and debug the system.
- **evaluation:** Configures settings for evaluating the performance of the RAG pipeline and Gemini LLM, including the number of questions, paths for saving results, and plot configurations.

**Customization:**
- Service Account Credentials: Ensure that the credentials_path points to the downloaded IAM service account key JSON file.
- File Paths: Update the paths under paths, logging, and evaluation sections to match your local or cloud storage directories.
- Vector Search Parameters: Adjust parameters like dimensions, num_neighbors, and deployment settings based on your project’s requirements and the scale of your data.
- Chunking and Generation: Modify chunk_size, chunk_overlap, and generation parameters to optimize for performance and accuracy.

## Usage
The primary entry point of the ORAN RAG system is main.py. This script orchestrates the entire pipeline, from preprocessing documents to deploying the vector index and optionally running evaluations.
**Running the Pipeline**
```bash
python src/main.py --config configs/config.yaml [OPTIONS]
```

### Command-Line Arguments

#### Available Flags:

- **Configuration:**
  
  - ```--config``` 
    - Path to the configuration file
    - **Type:** str
    - **Default:** configs/config.yaml
    - **Example:** ```python src/main.py --config configs/config.yaml```

   - **Pipeline Control**
     - ```--skip-preprocessing:```:
       - Skip all preprocessing stages including:
         - Document conversion
         - Loading PDFs
         - Text formatting
         - Chunking
         - Embedding
       - **Example:** ```python src/main.py --skip-preprocessing```
     - ```--skip-create-index```:
       - Skip the index creation stage. Useful when the index has already been created.
       - **Example:** ```--python main.py --skip-create-index
     - ```--skip-create-endpoint```
       - Skip the creation of a new index endpoint. Useful when the index endpoint has already been created.
     - ```--skip-deploy-index```:
       - Skip the index deployment stage. Useful when the index has already been deployed to the endpoint.


      - **Evaluation:** 
        - ```--evaluation```
          - Toggle evaluation on
            - "on": Perform evaluation using the Q&A dataset.
            - "off": Skip evaluation
          - **Default:** off
          - **Example:** ``` python src/main.py --evaluation on```
      - **Help:**
        - ```--help```
          - Display the help message with detailed usage instructions.
          - **Example:** ``` python src/main.py --help```

### Usage Examples

1. Run only the Chatbot:
```
python src/main.py --skip-preprocessing --skip-create-index --skip-create-endpoint --skip-deploy-index
```
- **Behaviour:**
  - Skips preprocessing, index creation, endpoint creation, and index deployment.
  - Initializes VectorSearcher and starts the chatbot.

2. Skip only proprocessing
```
python src/main.py --skip-preprocessing
```
- **Behaviour:**
  - Skips preprocessing
  - Creates the index, endpoint, and deploys the index.
  - Starts the chatbot.

3.	Skip Preprocessing and Enable Evaluation:
```
python src/main.py --skip-preprocessing --evaluation on
```
- Behavior:
   - Skips preprocessing.
	- Creates the index and endpoint, deploys the index.
	- Skips chatbot interaction.
	- Executes the evaluation stage.

4.	Run the Full Pipeline with Evaluation:
```
python src/main.py  --evaluation on
```
- Behavior:
   - Performs preprocessing, index creation, endpoint creation, and index deployment.
	- Skips chatbot interaction.
	- Executes the evaluation stage.

5.	Run the Full Pipeline without Evaluation:
```
python src/main.py 
```
6.	Display the Help Message:
```
python src/main.py --help
```
- Behavior:
   - Displays detailed usage instructions and available command-line flags.

## Features
- **Document Preprocessing:** Converts, cleans, and formats documents by removing multi-line headers, footers, and Table of Contents (TOC) pages.
- **Chunking:** Splits documents into context-rich chunks for efficient embedding and indexing.
- **Embedding Generation:** Generates embeddings for each chunk using state-of-the-art models.
- **Vector Indexing:** Creates and manages vector indexes for efficient similarity search.
- **Chatbot Interface:** Interactive chatbot that utilizes the indexed data to answer user queries.
- **Evaluation Module:** Compares the performance of the RAG pipeline against the raw Gemini LLM using a predefined Q&A dataset.
- **Logging:** Comprehensive logging for monitoring and debugging.

## Evaluation

The evaluation module compares the performance of the RAG pipeline against the raw a LLM using a predefined Q&A dataset.

### Steps

1. **Load Q&A Dataset:**
The dataset is from an open sourced O-RAN benchmark. Only the hard Q&A dataset was loaded into the Google Cloud Storage then loaded from the GCS path specified in qna_dataset_path within config.yaml. Each entry should include a question, multiple choices, and the correct answer.

2. **Run Evaluation:**
3. **Results:**
   - Excel Report:
     - An Excel file (evaluation_results.xlsx) is generated, detailing each question, the correct answer, the predictions from both RAG and Gemini LLM, and their correctness.
   - **Accuracy Plots:**
     - A bar chart comparing the accuracies of the RAG pipeline and Gemini LLM is saved as accuracy_plots.png.

  