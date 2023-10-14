

import torch                                                                                                                                     

class TinyModel(torch.nn.Module):

    def __init__(self):
        super(TinyModel, self).__init__()

        self.linear1 = torch.nn.Linear(4, 1)
        # self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        # x = self.activation(x)
        return x


tinymodel = TinyModel()

print('The model:')
print(tinymodel)

print('\n\nJust one layer:')
print(tinymodel.linear1)

# print('\n\nModel params:')
# for param in tinymodel.parameters():
#     print(param)

import torch_mlir

model = tinymodel
model.eval()
# data = torch.randn(4,1)
data = torch.eye(1, 4)
out_stablehlo_mlir_path = "./tinymodel_stablehlo.mlir"                                                                                                             
module = torch_mlir.compile(model, data, output_type=torch_mlir.OutputType.STABLEHLO, use_tracing=False)                                                                         
with open(out_stablehlo_mlir_path, "w", encoding="utf-8") as outf:
    outf.write(str(module))
print(f"StableHLO IR of resent18 successfully \
 written into {out_stablehlo_mlir_path}")


from magic import *

res = Engine(model, data)
print(res)
