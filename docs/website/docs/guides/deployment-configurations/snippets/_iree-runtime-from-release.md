=== ":octicons-package-16: Stable releases"

    Stable release packages are [published to PyPI](https://pypi.org/).

    ``` shell
    python -m pip install iree-base-runtime
    ```

=== ":octicons-beaker-16: Nightly releases"

    Nightly pre-releases are published on
    [GitHub releases](https://github.com/iree-org/iree/releases).

    ``` shell
    python -m pip install \
      --find-links https://iree.dev/pip-release-links.html \
      --upgrade --pre iree-base-runtime
    ```

--8<-- "docs/website/docs/snippets/_iree-dev-packages.md"
