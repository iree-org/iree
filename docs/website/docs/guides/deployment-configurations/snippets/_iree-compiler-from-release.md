=== "Stable releases"

    Stable release packages are
    [published to PyPI](https://pypi.org/user/google-iree-pypi-deploy/).

    ``` shell
    python -m pip install iree-compiler
    ```

=== ":material-alert: Nightly releases"

    Nightly releases are published on
    [GitHub releases](https://github.com/iree-org/iree/releases).

    ``` shell
    python -m pip install \
      --find-links https://iree.dev/pip-release-links.html \
      --upgrade iree-compiler
    ```

!!! tip
    `iree-compile` and other tools are installed to your python module
    installation path. If you pip install with the user mode, it is under
    `${HOME}/.local/bin`, or `%APPDATA%Python` on Windows. You may want to
    include the path in your system's `PATH` environment variable:

    ```shell
    export PATH=${HOME}/.local/bin:${PATH}
    ```
