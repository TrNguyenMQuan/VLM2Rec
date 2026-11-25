#!/bin/bash

# Define the URL and output file
URL="https://snap.stanford.edu/data/amazon/productGraph/metadata.json.gz"
OUTPUT_FILE="metadata.json.gz"

# Download the file
echo "Downloading file from $URL..."
curl -o $OUTPUT_FILE $URL

# Check if the download was successful
if [ $? -eq 0 ]; then
    echo "Download completed successfully."
else
    echo "Download failed."
    exit 1
fi

# Extract the file
echo "Extracting $OUTPUT_FILE..."
gunzip $OUTPUT_FILE

# Check if the extraction was successful
if [ $? -eq 0 ]; then
    echo "Extraction completed successfully."
else
    echo "Extraction failed."
    exit 1
fi