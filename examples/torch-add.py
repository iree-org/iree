#!/usr/bin/env python
# coding=utf-8
import torch                                                                                                                                     

class TinyModel(torch.nn.Module):

    def __init__(self):
        super(TinyModel, self).__init__()

    def forward(self, x, y):
        add = torch.add(x, y)
        return add


tinymodel = TinyModel()

print('The model:')
print(tinymodel.eval())

# print('\n\nModel params:')
# for param in tinymodel.parameters():
#     print(param)

import torch_mlir

model = tinymodel
model.eval()
# data = torch.randn(4,1)
arg0 = torch.eye(1, 4)
arg1 = torch.eye(1, 4)
out_stablehlo_mlir_path = "./tinymodel_stablehlo.mlir"                                                                                                             
module = torch_mlir.compile(model, (arg0, arg1), output_type=torch_mlir.OutputType.STABLEHLO, use_tracing=False)                                                                         
with open(out_stablehlo_mlir_path, "w", encoding="utf-8") as outf:
    outf.write(str(module))
print(f"StableHLO IR of resent18 successfully \
 written into {out_stablehlo_mlir_path}")


from magic import *

res = Engine(model, arg0, arg1)
print(res)

