# IREE–JAX Frontend

## Requirements

A local JAX installation is necessary in addition to IREE's Python requirements:

```shell
python -m pip install jax jaxlib
```

## Just In Time Compilation with Runtime Bindings

A just-in-time compilation decorator similar to `jax.jit` is provided by
`iree.jax.jit`:

```python
import iree.jax

import jax
import jax.numpy as jnp

# 'backend' is one of 'vmla', 'llvmaot' and 'vulkan' and defaults to 'llvmaot'.
@iree.jax.jit(backend="llvmaot")
def linear_relu_layer(params, x):
  w, b = params
  return jnp.max(jnp.matmul(x, w) + b, 0)

w = jnp.zeros((784, 128))
b = jnp.zeros(128)
x = jnp.zeros((1, 784))

linear_relu_layer([w, b], x)
```

## Ahead of Time Compilation

An ahead-of-time compilation function provides a lower-level API for compiling a
function with a specific input signature without creating the runtime bindings
for execution within Python. This is primarily useful for targeting other
runtime environments like Android.

### Example: Compile a MLP and run it on Android

Install the Android NDK according to the
[Android Getting Started](https://google.github.io/iree/get-started/getting-started-android-cmake)
doc, and then ensure the following environment variable is set:

```shell
export ANDROID_NDK=# NDK install location
```

The code below assumes that you have `flax` installed.

```python
import iree.jax

import jax
import jax.numpy as jnp
import flax
from flax import linen as nn


class MLP(nn.Module):

  @nn.compact
  def __call__(self, x):
    x = x.reshape((x.shape[0], -1))  # Flatten.
    x = nn.Dense(128)(x)
    x = nn.relu(x)
    x = nn.Dense(10)(x)
    x = nn.log_softmax(x)
    return x


image = jnp.zeros((1, 28, 28, 1))
params = MLP().init(jax.random.PRNGKey(0), image)["params"]

apply_args = [{"params": params}, image]
options = dict(target_backends=["dylib-llvm-aot"],
               extra_args=["--iree-llvm-target-triple=aarch64-linux-android"])

compiled_binary = iree.jax.aot(MLP().apply, *apply_args, **options)

with open("/tmp/mlp_apply.vmfb", "wb") as f:
  f.write(compiled_binary)
```

IREE doesn't provide installable tools for Android at this time, so they'll need
to be built according to the
[Android Getting Started](https://google.github.io/iree/get-started/getting-started-android-cmake).
Afterward, the compiled `.vmfb` can be pushed to an Android device and executed
using `iree-run-module`:

```shell
adb push /tmp/mlp_apply.vmfb /data/local/tmp/
adb push ../iree-build-android/iree/tools/iree-run-module /data/local/tmp/
adb shell /data/local/tmp/iree-run-module \
  --driver=dylib \
  --module_file=/data/local/tmp/mlp_apply.vmfb \
  --entry_function=main \
  --function_input=128xf32 \
  --function_input=784x128xf32 \
  --function_input=10xf32 \
  --function_input=128x10xf32 \
  --function_input=1x28x28x1xf32
```
