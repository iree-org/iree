=== ":octicons-git-branch-16: Development packages"

    Development packages are built at every commit and on pull requests, for
    limited configurations.

    On **Linux** with **Python 3.11**, development packages can be installed
    into a [Python `venv`](https://docs.python.org/3/library/venv.html) using
    the
    [`build_tools/pkgci/setup_venv.py`](https://github.com/iree-org/iree/blob/main/build_tools/pkgci/setup_venv.py)
    script:

    ``` shell
    # Install packages from a specific commit ref.
    # See also the `--fetch-latest-main` and `--fetch-gh-workflow` options.
    python ./build_tools/pkgci/setup_venv.py /tmp/.venv --fetch-git-ref=8230f41d
    source /tmp/.venv/bin/activate
    ```
