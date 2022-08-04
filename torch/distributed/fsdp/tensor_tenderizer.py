import abc




try:
    from spmd.tensor import Tensor as DistributedTensor, DeviceMesh
    from spmd.tensor.placement_types import Placement

    def is_distributed_tensor(obj):
        return isinstance(obj, DistributedTensor)

except:
    def is_distributed_tensor(obj):
        return False

