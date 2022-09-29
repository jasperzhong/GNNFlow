from collections import namedtuple


fields = ("row", "col", "num_src_nodes", "num_dst_nodes",
          "all_nodes", "all_timestamps", "delta_timestamps", "eids")
SamplingResultType = namedtuple('SamplingResultType', fields,
                                defaults=(None,) * len(fields))

# let pickle know how to serialize the SamplingResultType
globals()['SamplingResultType'] = SamplingResultType
