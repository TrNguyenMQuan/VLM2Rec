#!/bin/bash

DATASET_NAME="$1"
mkdir -p raw
cd raw

if [ "$DATASET_NAME" = "all" ]; then
    cd ..
    sh download_data.sh Beauty            
    sh download_data.sh Sports            
    sh download_data.sh Toys_and_Games
    sh download_data.sh Clothing
    # sh _download.sh Yelp           

elif [ "$DATASET_NAME" = "Beauty" ]; then
    wget https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz -O Beauty.json.gz
    gzip -d Beauty.json.gz

elif [ "$DATASET_NAME" = "Sports" ]; then
    wget https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Sports_and_Outdoors_5.json.gz -O Sports.json.gz
    gzip -d Sports.json.gz

elif [ "$DATASET_NAME" = "Toys_and_Games" ]; then
    wget https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Toys_and_Games_5.json.gz -O Toys_and_Games.json.gz
    gzip -d Toys_and_Games.json.gz

elif [ "$DATASET_NAME" = "Clothing" ]; then
    wget https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Clothing_Shoes_and_Jewelry_5.json.gz -O Clothing.json.gz
    gzip -d Clothing.json.gz


else   
    echo "Invalid dataset name"
    echo ""
fi