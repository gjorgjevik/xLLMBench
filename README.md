# xLLMBench
This repository contains the code for the research article "User-Defined Trade-Offs in LLM Benchmarking: Balancing Accuracy, Scale, and Sustainability", authored by Ana Gjorgjevikj, Ana Nikolikj, Barbara Koroušić Seljak, and Tome Eftimov (Computer Systems Department, Jožef Stefan Institute).

## Project Overview

### `xllmbench`
This folder contains the core code and experiment configuration organised as follows:
- **`ranking.py`**: Implements the core functions of the Promethee II method, which is used for ranking.
- **`preference.py`**: Implements the preference functions of the Promethee II method.
- **`pipeline.py`**: Demonstrates the application of the Promethee method using an example experiment. Outputs two CSV files, (1) file with predicted rankings, predicted scores (net flows), positive preference flows, and negative preference flows, and (2) file with the preference matrix. The output files can be used to visualize the results. This script can be adapted to generate rankings for other scenarios by simply modifying the configuration settings.
- **`experiments.txt`**: Contains all experiment configurations used in the accompanying paper.
