# xLLMBench
User-Defined Trade-Offs in LLM Benchmarking: Balancing Accuracy, Scale, and Sustainability

## Project Overview

### `xllmbench`
This folder contains the core code and experiment configuration organised as follows:
- **`ranking.py`**: Implements the core functions of the Promethee II method, which is used for ranking.
- **`preference.py`**: Implements the preference functions of the Promethee II method.
- **`pipeline.py`**: Demonstrates the application of the Promethee method using Experiment 1, Scenario 4, as an example. This script can be adapted to generate rankings for other scenarios by simply modifying the configuration settings.
- **`experiments.txt`**: Contains all experiment configurations used in the accompanying paper.
