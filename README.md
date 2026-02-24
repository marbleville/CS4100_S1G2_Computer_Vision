# CS4100_S1G2_Computer_Vision

CS 4100 Section 1 Group 2 final project. Hands free video playback control using computer vision.

## Environment Setup

Clone this repo, then:

1. Instantiate a Python3 virtual environment with `uv`: `uv venv --python 3`

2. Install project requirements: `uv pip install -r requirements.txt`

3. Running scripts: `.venv/bin/python -m [module name].[file]`
   - Example: `.venv/bin/python -m preprocessor.example`
   - With profiler: `.venv/bin/python -m cProfile -s cumtime preprocessor/example.py > profile.txt`
