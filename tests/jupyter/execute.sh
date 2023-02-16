#!/bin/bash

jupyter nbconvert --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3  --to notebook --inplace --execute $1 --log-level WARN
jupyter_exit_code=$?

exit $jupyter_exit_code
