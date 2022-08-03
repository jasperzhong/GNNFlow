#!/bin/bash

./run.sh tgat uniform REDDIT LFUCache
./run.sh tgat uniform REDDIT LRUCache
./run.sh tgat uniform REDDIT FIFOCache
./run.sh tgat recent REDDIT LFUCache
./run.sh tgat recent REDDIT LRUCache
./run.sh tgat recent REDDIT FIFOCache
./run.sh tgat uniform WIKI LFUCache
./run.sh tgat uniform WIKI LRUCache
./run.sh tgat uniform WIKI FIFOCache
./run.sh tgat recent WIKI LFUCache
./run.sh tgat recent WIKI LRUCache
./run.sh tgat recent WIKI FIFOCache
./run.sh tgat uniform MOOC LRUCache
./run.sh tgat uniform MOOC LFUCache
./run.sh tgat uniform MOOC FIFOCache
./run.sh tgat recent MOOC LRUCache
./run.sh tgat recent MOOC LFUCache
./run.sh tgat recent MOOC FIFOCache
./run.sh tgat uniform LASTFM LRUCache
./run.sh tgat uniform LASTFM LFUCache
./run.sh tgat uniform LASTFM FIFOCache
./run.sh tgat recent LASTFM LRUCache
./run.sh tgat recent LASTFM LFUCache
./run.sh tgat recent LASTFM FIFOCache
# ./run.sh tgat uniform MAG LRUCache
# ./run.sh tgat uniform MAG LFUCache
# ./run.sh tgat uniform MAG FIFOCache
# ./run.sh tgat recent MAG LRUCache
# ./run.sh tgat recent MAG LFUCache
# ./run.sh tgat recent MAG FIFOCache
# ./run.sh tgat uniform GDELT LRUCache
# ./run.sh tgat uniform GDELT LFUCache
# ./run.sh tgat uniform GDELT FIFOCache
# ./run.sh tgat recent GDELT LRUCache
# ./run.sh tgat recent GDELT LFUCache
# ./run.sh tgat recent GDELT FIFOCache