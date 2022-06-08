# dynamic-graph-neural-network

## TODO:

- [ ] raise an exception when adding edges with timestmaps that are smaller than the current timestamps.
- [ ] addedges: set the max_size of a block so that when the size of the incoming edges is larger than the max_size of the block, it can be automatically split to many blocks rather than only using one block to contain all the edges.