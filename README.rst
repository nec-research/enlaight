enlAIght
========

**enlaight**
is a word creation of **enlight** and **AI** and refers to a key property of
the contained models: their built-in interpretability. By this, it is possible to
create machine learning models that surrogate an existing machine learning model so
that it is possible to explain the reasoning process of the original model to some
extent. *Thus, these models enlight AI.*


Available Models
----------------

The available models are prototype-based learning methods implemented in
`PyTorch-Lightning <https://lightning.ai/docs/pytorch/stable/>`_.
The available models are:

* Generalized Learning Vector Quantization (GLVQ),
* Generalized Tangent Learning Vector Quantization (GTLVQ),
* (Stable) Classification-by-Components (CBC),
* Radial Basis Function (RBF) networks,

Prototype models are interpretable machine learning models for classification tasks. In
a nutshell, a prototype model consists of a distance function and a set of
prototypes defined in the data space with fixed class labels. By computing the distance
between prototypes and a given input, the closest prototype can be determined. This
prototype determines by its class label the label of the input, so-called
winner-takes-all rule. By defining a suitable loss function and by having a training
dataset, the position of the prototypes in the data space can be learned from data so
that the classification accuracy is maximized. The main advantages of prototype-based
models are

* their built-in interpretability and
* their provably robust classification decisions.


Model Interface
---------------

The interface of the models is flexible. For instance, there is no requirement for how
the prototypes have to be provided. They can be the result of another module or can be
fixed and non-trainable. Moreover, the prototypes class supports constraints. Therefore,
prototypes can be constraint to be in a certain data space.


Available Distance Functions
----------------------------

The required distance operations are implemented such that they support fast and
memory efficient computations (by reformulating the distance operations with
dot-products). The following distance functions are supported as functions
and PyTorch Module classes (see :class:`enlaight.core.distance` for the full list):

* Cosine similarity is imported from PyTorch
* Lp distance is import from PyTorch
* (Squared) Euclidean distance
* (Squared) Tangent distance

All distance operations support batching with respect to both arguments.
The implementations support the computation of stable gradients.


.. Include begin: Installation


Installation
------------

To install the package, execute the following command from the root of the package
directory:

.. code-block:: shell

   pip install .

Note that the package requires *Python 3.9* or higher, which is checked during the
installation. Moreover, if you install the package inside a *conda* environment, be
aware of potential installation or package side-effects due to conflicts between *conda*
and *pip*. If you encounter errors, install all dependencies directly with *conda*.


.. Include begin: Documentation


Building the Documentation
--------------------------

To build the documentation HTML files, install the package with ``docs`` dependencies:

.. code-block:: shell

   pip install .[docs]

Then, execute:

.. code-block:: shell

   sphinx-build -b html docs docs/build

The compiled documentation is located in ``docs/build``.


Contribution to the Development
-------------------------------

For contributions, install the package in dev-mode:

.. code-block:: shell

   pip install .[dev]

or with all dependencies (including ``dev`` and ``docs``):

.. code-block:: shell

   pip install .[all]

If you are working in a *conda* environment and encounter any installation or
dependency errors, please install all packages using *conda*.

Documentation
^^^^^^^^^^^^^

If you prepare a code submission, always ensure that you provide docstrings and that
the documentation can be generated.

The documentation is completely generated from docstrings and this README file. So far,
we avoid providing additional information in additional documentation files.
If you encounter *pandoc* error during the documentation creation on Linux machines
even though *pandoc* is installed via *pip*, install it via

.. code-block::

    apt-get install pandoc

If you have errors with *ipykernel* during doc compilation
while using *conda*, uninstall the *pip* version and install it via *conda*.

Code Submission
^^^^^^^^^^^^^^^

Additionally, it is recommended to install *pre-commit* so that *pre-commit* checks
are triggered automatically before making a commit; thus, avoiding non-standardized commits:

.. code-block:: shell

    pip install pre-commit
    pre-commit install

Moreover, install

.. code-block:: shell

    pre-commit install --hook-type commit-msg

to ensure that your commit messages follow
`conventional commits <https://www.conventionalcommits.org/>`_, which is recommended.
Again, if you encounter errors while using *conda*, uninstall *pre-commit* in *pip* and
install it via *conda*.

If you prepare a commit, run

.. code-block:: shell

    pre-commit run --all-files

to test for errors with respect to *pre-commit* hooks. In case you really want to do
a non-standardized commit use ``--no-verify`` option of ``git commit`` to skip the
checks.


Reproducing the AAAI Experiments
--------------------------------

The package was used to create a part of the results of the corresponding
`AAAI 2025 paper <https://doi.org/10.1609/aaai.v39i19.34233>`_. In particular, the
models provide in this package were used for the shallow model experiments. For the
deep models, please check the
`HuggingFace <https://huggingface.co/si-cim/cbc-aaai-2025>`_ and the
`GitHub repository <https://github.com/si-cim/cbc-aaai-2025>`_.

To reproduce the results, install the package in dev-mode:

.. code-block:: shell

   pip install .[dev]

Then, execute

.. code-block:: shell

   cd ./experiments
   python model_comparison.py

to reproduce the results of the shallow model comparison. Please note that the script
uses *ray-tune* for parallel scheduling of the jobs and assumes that a GPU is
available. If multiple GPUs are available, *ray-tune* will execute the individual runs
in parallel. Since the models are relatively small, it could be possible to compute
multiple models in parallel on one GPU. For this, change the following line in the
Python script:

.. code-block:: python

   tune.with_resources(objective, {"gpu": 1})  # 100% job-allocation per GPU

to

.. code-block:: python

   tune.with_resources(objective, {"gpu": 1/2})  # 50% job-allocation per GPU

This will allow ray to run 2 jobs per GPU.

After the training of all the models is completed, you can use the script
:code:`./experiments/print_shallow_model_results.py` to generate one consolidated
dictionary with all the results and to render the results in an easy human-readable
format. Only specify the root path at the top of the file.

To reproduce the robustness analysis (robustness curves), execute:

.. code-block:: shell

   cd ./experiments
   python robustness_analysis.py

Similar to before, you can specify the GPU usage of *ray-tune* in the file.
Moreover, use the :code:`./experiments/robustness_plot.py` script to generate the plots
from the paper. Again, specify the root path at the top of the file.
