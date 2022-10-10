#!/bin/bash
# from https://github.com/amazon-research/tgl/blob/main/down.sh

mkdir -p ../data/MOOC/
aria2c -x 16 -d ../data/MOOC https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/MOOC/edges.csv
mkdir -p ../data/REDDIT/
aria2c -x 16 -d ../data/REDDIT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/REDDIT/edge_features.pt
aria2c -x 16 -d ../data/REDDIT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/REDDIT/edges.csv
aria2c -x 16 -d ../data/REDDIT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/REDDIT/labels.csv
mkdir -p ../data/WIKI
aria2c -x 16 -d ../data/WIKI https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/WIKI/edge_features.pt
aria2c -x 16 -d ../data/WIKI https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/WIKI/edges.csv
aria2c -x 16 -d ../data/WIKI https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/WIKI/labels.csv
mkdir -p ../data/LASTFM/
aria2c -x 16 -d ../data/LASTFM https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/LASTFM/edges.csv
mkdir -p ../data/GDELT/
aria2c -x 16 -d ../data/GDELT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/node_features.pt
aria2c -x 16 -d ../data/GDELT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/labels.csv
aria2c -x 16 -d ../data/GDELT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/edges.csv
aria2c -x 16 -d ../data/GDELT https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/GDELT/edge_features.pt
mkdir -p ../data/MAG/
aria2c -x 16 -d ../data/MAG https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/MAG/labels.csv
aria2c -x 16 -d ../data/MAG https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/MAG/edges.csv
aria2c -x 16 -d ../data/MAG https://s3.us-west-2.amazonaws.com/dgl-data/dataset/tgl/MAG/node_features.pt

