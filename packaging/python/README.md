# Python packaging scripts.

Note that packages will be placed in `packaging/python/dist` with the canonical
instructions. However, the setup scripts can be run from anywhere and will
create `build` and `dist` directories where run. Wheels can be installed with
`pip3 install --user dist/*.whl`.

## Building core wheels with CMake

Most of IREE is built/packaged with CMake. For the parts that build with CMake,
this is preferred.

Canonical instructions follow:

### Linux

```shell
export LDFLAGS=-fuse-ld=/usr/bin/ld.lld
export PYIREE_CMAKE_BUILD_ROOT="${HOME?}/build-iree-release"
export IREE_SRC="${HOME?}/src/iree"
rm -Rf "${PYIREE_CMAKE_BUILD_ROOT?}"; mkdir -p "${PYIREE_CMAKE_BUILD_ROOT?}"
cmake -GNinja -B"${PYIREE_CMAKE_BUILD_ROOT?}" -H"${IREE_SRC}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DIREE_BUILD_PYTHON_BINDINGS=ON -DIREE_BUILD_SAMPLES=OFF
(cd "${PYIREE_CMAKE_BUILD_ROOT?}" && ninja)
(cd "${IREE_SRC?}/packaging/python" && (
  rm -Rf build;
  python3 setup_compiler.py bdist_wheel;
  rm -Rf build;
  python3 setup_rt.py bdist_wheel))
```

## Building IREE/TensorFlow wheels

If building TensorFlow integration wheels, then this must be done via Bazel. In
this case, it can be easiest to just package everything from a Bazel build to
avoid multiple steps.

Canonical instructions follow:

### Env Setup

```shell
IREE_SRC=$HOME/src/iree
export PYIREE_BAZEL_BUILD_ROOT="$IREE_SRC/bazel-bin"
if which cygpath; then
  export PYIREE_BAZEL_BUILD_ROOT="$(cygpath -w "$PYIREE_BAZEL_BUILD_ROOT")"
fi
```

### Building:

Optionally add: `--define=PYIREE_TF_DISABLE_KERNELS=1` to build a 'thin' (less
functional) version without TensorFlow kernels. This should not be done for
released binaries but can help while developing.

Note that bazel does not always build properly named artifacts. See the tool
`hack_python_package_from_runfiles.py` to extract and fixup artifacts from a
bazel-bin directory. If using this mechanism, then the environment variable
`PYIREE_PYTHON_ROOT` should be set to a suitable temp directory.

```shell
cd $IREE_SRC
bazel build -c opt \
  //packaging/python:all_pyiree_packages
```

# Packaging

```shell
(cd $IREE_SRC/packaging/python && (
  rm -Rf build;
  python3 setup_tf.py bdist_wheel))
```

```shell
(cd $IREE_SRC/packaging/python && (
  rm -Rf build;
  python3 setup_compiler.py bdist_wheel))
```

```shell
(cd $IREE_SRC/packaging/python && (
  rm -Rf build;
  python3 setup_rt.py bdist_wheel))
```
