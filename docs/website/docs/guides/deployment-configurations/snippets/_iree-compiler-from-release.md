=== ":octicons-package-16: Stable releases"

    Stable release packages are [published to PyPI](https://pypi.org/).

    ``` shell
    python -m pip install iree-base-compiler
    ```

=== ":octicons-beaker-16: Nightly releases"

    Nightly pre-releases are published on
    [GitHub releases](https://github.com/iree-org/iree/releases).

    ``` shell
    python -m pip install \
      --find-links https://iree.dev/pip-release-links.html \
      --upgrade --pre iree-base-compiler
    ```

--8<-- "docs/website/docs/snippets/_iree-dev-packages.md"

!!! tip
    `iree-compile` and other tools are installed to your python module
    installation path. If you pip install with the user mode, it is under
    `${HOME}/.local/bin`, or `%APPDATA%Python` on Windows. You may want to
    include the path in your system's `PATH` environment variable:

    ```shell
    export PATH=${HOME}/.local/bin:${PATH}
    ```
