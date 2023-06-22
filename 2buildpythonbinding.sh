
python -m pip install --upgrade pip 
python -m pip install -r ./runtime/bindings/python/iree/runtime/build_requirements.txt


cd iree-build-host
cmake \
    -GNinja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DIREE_BUILD_PYTHON_BINDINGS=ON \
    -DPython3_EXECUTABLE="$(which python)" \
    .
cmake --build .

# Add the bindings/python paths to PYTHONPATH and use the API.
source .env && export PYTHONPATH
python -c "import iree.compiler"
python -c "import iree.runtime"

