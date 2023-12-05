import numpy as np

batch = 1
reductionSize = 33000 # tried 32000, 32, 10
label = 237
data = np.zeros([batch, reductionSize]).astype(np.float32)
data[0, label] = 53.0
np.save("input0.npy", data)
data_fp16 = data.astype(np.float16)
np.save("input0_f16.npy", data_fp16)