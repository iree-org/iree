# Python packaging scripts.

Note that packages will be placed in `bindings/python/packaging/dist` with the
canonical instructions. However, the setup scripts can be run from anywhere and
will create `build` and `dist` directories where run. Wheels can be installed
with `pip3 install --user dist/*.whl`.

## Building core wheels

Most of IREE is built/packaged with CMake. Canonical instructions follow:

### Linux

```shell
export LDFLAGS=-fuse-ld=/usr/bin/ld.lld-10
export CMAKE_BUILD_ROOT=$HOME/build-iree-release
export IREE_SRC=$HOME/src/iree
rm -Rf $CMAKE_BUILD_ROOT; mkdir -p $CMAKE_BUILD_ROOT
cmake -GNinja -B$CMAKE_BUILD_ROOT -H$IREE_SRC \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DIREE_BUILD_PYTHON_BINDINGS=ON -DIREE_BUILD_SAMPLES=OFF
(cd $CMAKE_BUILD_ROOT && ninja)
(cd $IREE_SRC/bindings/python/packaging && (
rm -Rf build;
python3 setup_compiler.py bdist_wheel;
rm -Rf build;
python3 setup_rt.py bdist_wheel))
```

## Building IREE/TensorFlow wheels

TensorFlow integration must be built via Bazel. Canonical instructions follow:

### Linux

```shell
export IREE_SRC=$HOME/src/iree
cd $IREE_SRC
bazel build -c opt \
  //integrations/tensorflow/bindings/python/packaging:all_tf_packages
(cd $IREE_SRC/bindings/python/packaging && (
rm -Rf build;
python3 setup_tf.py bdist_wheel))
```
