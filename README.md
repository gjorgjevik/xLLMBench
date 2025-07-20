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

## Input Data Preparation Instructions

To prepare the input data for use with **`xllmbench`**, follow the steps outlined below.

1. The input data should be in CSV format, consisting of a selected portfolio of LLMs (each in a separate row) and (non-)performance criteria measurements (each in a separate column).
2. Each criteria column should be uniquely named in the header row.
3. Dedicate one column to the LLM names, the name of which will be assigned to the parameter 'name_column' in the experiment configuration.
4. If additional metadata is needed to uniquely identify each LLM, specify it in separate column(s). Assign the names of those columns as a list to the parameter 'id_columns' in the experiment configuration. The values of those columns will be concatenated with the LLM name, to uniquely identify the LLM.
5. The names of the columns containing the performance and non-performance criteria measurements should be provided as a list to the parameter 'metric_columns' in the experiment configuration.
6. The dataset should be complete, without any missing values.

## Defining Custom Experiment Configurations

To configure a custom experiment, specify the values of the parameters listed below. For more examples, see the **`experiments.txt`** file.

- **`scenario`**: Experiment name
- **`directory`**: Path to input CSV file
- **`input_file`**: Name of the input CSV file
- **`delimiter`**: CSV file delimiter
- **`name_column`**: Model name column
- **`id_columns`**: List of additional columns to be used for unique identification of each LLM in combination with its name
- **`metric_columns`**: List of criteria columns containing the measurement data
- **`preference_functions`**: Preference functions by criteria, in the same order as the criteria were specified in the list assigned to the **`metric_columns`** parameter
- **`user_specified_weights`**: User-specified weight by criteria, in the same order as the criteria were specified in the list assigned to the **`metric_columns`** parameter
- **`column_maximization`**: List indicating if the criterion is subjected to maximization or minimization. **`1`** for metric values maximization, **`-1`** for metric values minimization
- **`rank_output_file`**: Name of the output file to be created with the ranking results
- **`preference_parameters`**: Preference function parameter coefficients used during sensitivity analysis. See the main article for details. Use value of **`0.0`** for default parameter values in each preference function.

**`IMPORTANT NOTE`**: We strongly recommend that custom experiments are defined by skilled data scientists that can translate application-specific requirements into custom experiment configurations, and estimate the effects of such configurations on the results both theoretically and empirically.

## Defining Custom Preference Functions

The framework allows for the addition of other preference functions and use in the experiment configuration, following the steps outlined below and the implementation of the three currently available preference functions available in **`preference.py`**.

1. Define the new preference function in class **`PreferenceFunction`**, script **`preference.py`**.
2. Register the new preference function in **`PreferenceFunctionEnum`**, script **`preference.py`**.
3. Modify the method **`apply_preference_functions`** in script **`ranking.py`** by specifying the calculation of the default values of the function parameters (if any).

**`IMPORTANT NOTE`**: We strongly emphasize that definition of custom preference functions should be done only by skilled data scientists, be based on solid theoretical grounding, and have their effects on the results evaluated through extensive experiments.

## Analytic Hierarchy Process as Weighting Method

In the main article, as part of the sensitivity analysis in **`Appendix D`**, Analytic Hierarchy Process (AHP) was used as a more user-friendly way to specify pairwise judgments on the importance of one criterion over another and calculate criteria weights. The importance is defined on a scale from **`1`** to **`9`**, as described in **`Appendix D`**.

To calculate weights for **`m`** criteria, use the script **`weights.py`** and the steps outlined below. For example configuration used in **`Appendix D`**, see the file **`experiments_sensitivity_weights.txt`**.

1. Specify pairwise judgments on the importance of one criterion over another in an **`m x m`** matrix. See the example in script **`weights.py`**.
2. Calculate criteria weights.
3. Assign the list of weights to parameter **`user_specified_weights`** in the experiment configuration and run the experiment. The order should be the same as the one specified in the parameter **`metric_columns`**.
