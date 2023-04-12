Invoking Command Line Tools
===========================

As with many compilers, IREE's compiler consists of many command line tools,
some of which are designed for compiler devs and are only accessible via source
builds. User level tools are distributed via the Python packages and are also
accessible via dedicated Python APIs, documented here.

Core Compiler (`iree-compile`)
-----------------------

.. automodule:: iree.compiler.tools
  :members: compile_file, compile_str
  :imported-members:
  :undoc-members:

.. autoclass:: iree.compiler.tools.CompilerOptions
.. autoenum:: iree.compiler.tools.InputType
.. autoenum:: iree.compiler.tools.OutputFormat


Debugging
---------

.. automodule:: iree.compiler.tools.debugging
  :members:
  :imported-members:
  :undoc-members:


TFLite Importer (`iree-import-tflite`)
--------------------------------------

Using the API below to access `iree-import-tflite` presumes that the tool itself
is installed via the appropriate PIP package.

.. automodule:: iree.compiler.tools.tflite
  :members: compile_file, compile_str
  :imported-members:
  :undoc-members:

.. autoclass:: iree.compiler.tools.tflite.ImportOptions


TensorFlow Importer (`iree-import-tf`)
--------------------------------------

Using the API below to access `iree-import-tf` presumes that the tool itself
is installed via the appropriate PIP package.

.. automodule:: iree.compiler.tools.tf
  :members: compile_saved_model, compile_module

.. autoclass:: iree.compiler.tools.tf.ImportOptions
.. autoenum:: iree.compiler.tools.tf.ImportType

