dls-bxflow
=======================================================================

Distributed beamline data processing workflow engine and gui platform core.

Intended advantages:

- develop and manage automated data processing workflows

Installation
-----------------------------------------------------------------------
::

    pip install git+https://gitlab.diamond.ac.uk/scisoft/bxflow/dls-bxflow.git 

    dls_bxflow --version

Documentation
-----------------------------------------------------------------------

See https://www.cs.diamond.ac.uk/dls-bxflow for more detailed documentation.

Building and viewing the documents locally::

    git clone git+https://gitlab.diamond.ac.uk/scisoft/bxflow/dls-bxflow.git 
    cd dls-bxflow
    virtualenv /scratch/$USER/venv/dls-bxflow
    source /scratch/$USER/venv/dls-bxflow/bin/activate 
    pip install -e .[dev]
    make -f .dls-bxflow/Makefile validate_docs
    browse to file:///scratch/$USER/venvs/dls-bxflow/build/html/index.html

Topics for further documentation:

- TODO list of improvements
- change log


..
    Anything below this line is used when viewing README.rst and will be replaced
    when included in index.rst

