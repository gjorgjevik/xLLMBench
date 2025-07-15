# xLLMBench
This repository contains the code for the research article "User-Defined Trade-Offs in LLM Benchmarking: Balancing Accuracy, Scale, and Sustainability", authored by Ana Gjorgjevikj, Ana Nikolikj, Barbara Koroušić Seljak, and Tome Eftimov (Computer Systems Department, Jožef Stefan Institute).

## Project Overview

### `xllmbench`
This folder contains the core code and experiment configuration organised as follows:
- **`ranking.py`**: Implements the core functions of the Promethee II method, which is used for ranking.
- **`preference.py`**: Implements the preference functions of the Promethee II method.
- **`pipeline.py`**: Demonstrates the application of the Promethee method using an example experiment. Outputs two CSV files, (1) file with predicted rankings, predicted scores (net flows), positive preference flows, and negative preference flows, and (2) file with the preference matrix. The output files can be used to visualize the results. This script can be adapted to generate rankings for other scenarios by simply modifying the configuration settings.
- **`experiments.txt`**: Contains all default experiment configurations used in the main text of the accompanying paper.
- **`experiments_sensitivity_preference.txt`**: Contains experiment configurations used in the sensitivity analysis of the preference functions parameters, presented in the Appendices of the paper.
- **`experiments_sensitivity_method.txt`**: Contains experiment configurations used in the sensitivity analysis comparing Promethee II to other Multi-Criteria Decision-Making (MCDM) methods, presented in the Appendices of the paper.
- **`experiments_sensitivity_weights.txt`**: Contains experiment configurations used in the sensitivity analysis comparing manually defined criteria weights with weights calculated using the Analytic Hierarchy Process (AHP) MCDM method, presented in the Appendices of the paper.
