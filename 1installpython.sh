# Make sure your 'python' is what you expect. Note that on multi-python
# systems, this may have a version suffix, and on many Linuxes where
# python2 and python3 can co-exist, you may also want to use `python3`.
which python
python --version

# Create a persistent virtual environment (first time only).
python -m venv iree.venv

# Activate the virtual environment (per shell).
# Now the `python` command will resolve to your virtual environment
# (even on systems where you typically use `python3`).
source iree.venv/bin/activate

# Upgrade PIP. On Linux, many packages cannot be installed for older
# PIP versions. See: https://github.com/pypa/manylinux
python -m pip install --upgrade pip

# Install IREE build pre-requisites.
python -m pip install -r ./runtime/bindings/python/iree/runtime/build_requirements.txt

