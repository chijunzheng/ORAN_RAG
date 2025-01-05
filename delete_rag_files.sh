#!/bin/bash

# Configuration - Replace these with your actual values
PROJECT_ID="350296334322"  # Replace with your Project ID
LOCATION="us-central1"     # Replace with your Location
RAG_CORPUS_ID="9144559043375792128"  # Replace with your RAG Corpus ID

# Function to delete a single RAG file
delete_rag_file() {
  local rag_file_name="$1"
  echo "Deleting RAG file: $rag_file_name"
  curl -s -X DELETE \
    -H "Authorization: Bearer $(gcloud auth print-access-token)" \
    "https://$LOCATION-aiplatform.googleapis.com/v1beta1/$rag_file_name"

  if [ $? -eq 0 ]; then
    echo "Successfully deleted: $rag_file_name"
  else
    echo "Failed to delete: $rag_file_name"
  fi

  # Optional: Add a short delay to respect API rate limits
  sleep 0.1
}

# Fetch all RAG files
echo "Fetching RAG files from corpus..."
RESPONSE=$(curl -s -X GET \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json; charset=utf-8" \
  "https://$LOCATION-aiplatform.googleapis.com/v1beta1/projects/$PROJECT_ID/locations/$LOCATION/ragCorpora/$RAG_CORPUS_ID/ragFiles")

# Check if the response contains 'ragFiles'
if echo "$RESPONSE" | jq -e '.ragFiles' > /dev/null; then
  # Extract the full 'name' field for each RAG file
  echo "$RESPONSE" | jq -r '.ragFiles[].name' > rag_file_names.txt

  # Check if any RAG files were found
  if [ ! -s rag_file_names.txt ]; then
    echo "No RAG files found to delete."
    exit 0
  fi

  # Iterate and delete each RAG file
  echo "Deleting RAG files..."
  while read -r RAG_FILE_NAME; do
    delete_rag_file "$RAG_FILE_NAME"
  done < rag_file_names.txt

  echo "Deletion process completed."
else
  echo "No 'ragFiles' found in the response."
  echo "Response was:"
  echo "$RESPONSE"
  exit 1
fi

# Clean up
rm -f rag_file_names.txt
