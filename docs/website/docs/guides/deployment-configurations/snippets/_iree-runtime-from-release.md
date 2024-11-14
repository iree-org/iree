=== "Stable releases"

    Stable release packages are
    [published to PyPI](https://pypi.org/user/google-iree-pypi-deploy/).

    ``` shell
    python -m pip install iree-base-runtime
    ```

=== ":material-alert: Nightly releases"

    Nightly pre-releases are published on
    [GitHub releases](https://github.com/iree-org/iree/releases).

    ``` shell
    python -m pip install \
      --find-links https://iree.dev/pip-release-links.html \
      --upgrade --pre iree-base-runtime
    ```
