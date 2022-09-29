from typing import NamedTuple

import torch

fields = [("row", torch.Tensor),
          ("col", torch.Tensor),
          ("all_nodes", torch.Tensor),
          ("all_timestamps",
           torch.Tensor),
          ("delta_timestamps",
           torch.Tensor),
          ("eids", torch.Tensor),
          ("num_src_nodes", int),
          ("num_dst_nodes", int)],

SamplingResultType = NamedTuple('SamplingResultType', fields,
                                defaults=(None,) * len(fields))
# let pickle know how to serialize the SamplingResultType
globals()['SamplingResultType'] = SamplingResultType
