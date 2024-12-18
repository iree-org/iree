from iree.turbine.aot import ParameterArchiveBuilder
import torch

archive = ParameterArchiveBuilder()
tensor = torch.tensor([1, 2, 3, 4], dtype=torch.uint8)
archive.add_tensor("the_torch_tensor", tensor)
archive.save("tensor_saved_with_iree_turbine.irpa")
