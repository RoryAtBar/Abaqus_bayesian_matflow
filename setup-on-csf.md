# Set up instructions for CSF3

We need a python environment containing a few key packages.
There are a few different approaches you can take, but I'm most familiar
with venv and pip rather than conda.

- Load a recent version of python

    ```
    module load apps/binapps/anaconda3/2023.09
    ```
- Create a virtual environment

    ```
    python -m venv .venv
    ```

- Install required packages

    ```
    pip install numpy scipy pandas matplotlib gpflow pymc
    ```
