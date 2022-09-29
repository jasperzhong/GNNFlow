import torch


class SamplingResultTorch:
    def __init__(self):
        self.row: torch.Tensor = None
        self.col: torch.Tensor = None
        self.num_src_nodes: int = None
        self.num_dst_nodes: int = None
        self.all_nodes: torch.Tensor = None
        self.all_timestamps: torch.Tensor = None
        self.delta_timestamps: torch.Tensor = None
        self.eids: torch.Tensor = None

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)


# let pickle know how to serialize the SamplingResultType
globals()['SamplingResultTorch'] = SamplingResultTorch
