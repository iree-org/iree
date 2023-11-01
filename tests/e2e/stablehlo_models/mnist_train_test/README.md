Test that IREE has the correct model and optimizer state after doing one train
step and after initialization of parameters. The ground truth is extracted from
a JAX model. The MLIR model is generated with IREE JAX.

To regenerate the model together with the test data use

```shell
python -m venv generate_mnist.venv
source generate_mnist.venv/bin/activate
# Add IREE Python to your PYTHONPATH, following
# https://iree.dev/building-from-source/getting-started/#python-bindings
pip install -r generate_test_data_requirements.txt
python ./generate_test_data.py
```

Upload to gcs

```shell
tar --remove-files -v -c -f mnist_train.tar *.npz *.mlirbc
DIGEST="$(sha256sum mnist_train.tar | awk '{print $1}')"
gcloud storage mv mnist_train.tar "gs://iree-model-artifacts/mnist_train.${DIGEST}.tar"
sed -i \
  "s|MODEL_ARTIFACTS_URL =.*|MODEL_ARTIFACTS_URL = \"https://storage.googleapis.com/iree-model-artifacts/mnist_train.${DIGEST}.tar\"|" \
  mnist_train_test.py
```
