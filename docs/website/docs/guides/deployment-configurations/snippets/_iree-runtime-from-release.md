=== "Stable releases"

    Stable release packages are
    [published to PyPI](https://pypi.org/user/google-iree-pypi-deploy/).

    ``` shell
    python -m pip install iree-runtime
    ```

=== ":material-alert: Nightly releases"

    Nightly releases are published on
    [GitHub releases](https://github.com/iree-org/iree/releases).

    ``` shell
    python -m pip install \
      --find-links https://iree.dev/pip-release-links.html \
      --upgrade iree-runtime
    ```
