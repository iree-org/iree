# Python Test Sources

Many of our Python integration tests are parameterized to run on multiple
platforms, and they therefore are not ammenable to just including RUN
lines in them. For these cases, we locate the python sources for the test
here and add this to the PYTHONPATH.

This directory is not scanned by the test runner.
