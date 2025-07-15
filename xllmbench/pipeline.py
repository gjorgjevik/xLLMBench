import csv
import os
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from xllmbench.ranking import PrometheeIIBasedRanking, PreferenceFunctionEnum

class Pipeline:
    @staticmethod
    def read(file: str, name_column: str, id_columns: list, metric_columns: list, delimiter: str):
        keys = []
        decision_matrix = []

        with (open(file, encoding='utf-8') as f):
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                key = f'{row[name_column]}'
                if id_columns:
                    len_ = len(id_columns)
                    key += ' ('
                    for i, column in enumerate(id_columns):
                        key += f'{row[column]}'
                        if i < len_ - 1:
                            key += ', '
                    key += ')'
                keys.append(key)

                decision_matrix_row = []
                for column in metric_columns:
                    decision_matrix_row.append(float(row[column]))
                decision_matrix.append(decision_matrix_row)

        return [keys, decision_matrix]

    @staticmethod
    def write(save_file: str, rows: list):
        with open(save_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=',')
            for row in rows:
                writer.writerow(row)

    @staticmethod
    def heatmap(file: str, data: np.ndarray, xticklabels: list, yticklabels: list, figsize: list):
        fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
        sns.heatmap(data, annot=True, fmt='.2f', xticklabels=xticklabels, yticklabels=yticklabels,
                    cbar_kws={'shrink': 0.4})
        ax.tick_params(axis='x', rotation=90)

        plt.savefig(file, dpi=150)

    @staticmethod
    def histogram_by_metric(file: str, tensor: np.ndarray, metrics: list, preference_functions: list):
        m, n, l = tensor.shape
        fig, axs = plt.subplots(1, m, tight_layout=True, figsize=(m * 2.5, 2.5))

        for i in range(tensor.shape[0]):
            t = tensor[i, :, :].flatten()
            axs[i].set_xlabel(f'mean={np.mean(t):.2f} ({np.std(t):.2f})')
            if preference_functions[i] == PreferenceFunctionEnum.GAUSSIAN:
                t = t[t > 0.0]
            axs[i].hist(t, bins=int(n / 2))
            axs[i].set_title(metrics[i])

        plt.savefig(file, dpi=150)
    
    def run(self, experiment_config: dict, visualize: bool):
        experiment_config = experiment_config

        directory = experiment_config['directory']
        input_file = experiment_config['input_file']
        delimiter = experiment_config['delimiter']

        rank_output_file = experiment_config['rank_output_file']

        name_column = experiment_config['name_column']
        id_columns = experiment_config['id_columns']
        metric_columns = experiment_config['metric_columns']

        preference_functions = experiment_config['preference_functions']
        assert len(metric_columns) == len(preference_functions)

        user_specified_weights = experiment_config['user_specified_weights']
        assert len(metric_columns) == len(user_specified_weights)

        column_maximization = experiment_config['column_maximization']
        assert len(metric_columns) == len(column_maximization)

        preference_parameters = experiment_config['preference_parameters']

        [keys, decision_matrix] = self.read(
            file=os.path.join(directory, input_file),
            name_column=name_column,
            id_columns=id_columns,
            metric_columns=metric_columns,
            delimiter=delimiter)

        ranking = PrometheeIIBasedRanking()
        [preference_matrix, positive_preference_flow, negative_preference_flow, ranks, scores] = ranking.run(
            keys=keys,
            decision_matrix_raw=decision_matrix,
            column_maximization=column_maximization,
            preference_functions=preference_functions,
            user_specified_weights=user_specified_weights,
            preference_parameters=preference_parameters)

        models = []
        preference_flows = []
        results = [['model', 'predicted rank', 'predicted score', 'positive preference flow', 'negative preference flow']]
        for key in sorted(ranks, key=ranks.get):
            results.append([key, ranks[key], scores[key], positive_preference_flow[key], negative_preference_flow[key]])
            preference_flows.append([positive_preference_flow[key], negative_preference_flow[key]])
            models.append(key)

        self.write(os.path.join(directory, rank_output_file), results)
        if visualize:
            self.heatmap(
                data=preference_flows,
                xticklabels=[x.replace(' ', '\n') for x in ['positive preference flow', 'negative preference flow']],
                yticklabels=models,
                file=os.path.join(directory, rank_output_file.replace('.csv', '.pdf')),
                figsize=(6, len(models) / 4)
            )

        matrix = [['model'] + keys]
        for key, l in zip(keys, preference_matrix.tolist()):
            matrix.append([key] + l)

        self.write(os.path.join(directory, rank_output_file.replace('rank.csv', 'preference-matrix.csv')), matrix)
        if visualize:
            self.heatmap(
                data_raw=preference_matrix.tolist(),
                xticklabels=models,
                yticklabels=models,
                file=os.path.join(directory, rank_output_file.replace('rank.csv', 'preference-matrix.pdf')),
                figsize=(len(models) / 2, len(models) / 4)
            )

        if visualize:
            self.histogram_by_metric(
                file=os.path.join(
                    directory, rank_output_file.replace('rank.csv', 'pairwise-distance-before.jpg')),
                tensor=pairwise_distance_tensor_before, metrics=metric_columns,
                preference_functions=preference_functions)
            self.histogram_by_metric(
                file=os.path.join(
                    directory, rank_output_file.replace('rank.csv', 'pairwise-distance-after.jpg')),
                tensor=pairwise_distance_tensor, metrics=metric_columns,
                preference_functions=preference_functions)



if __name__ == '__main__':
    # example configuration for experiment open_llm, scenario s4 (for more experiment configurations see experiments.txt)
    config = {
        'scenario': 'open_llm_s4',  # experiment and scanario name
        'directory': r'\path\to\input\file',  # path to input csv file
        'input_file': 'input_file.csv',  # input csv file
        'delimiter': ',',  # csv file delimiter
        'name_column': 'Model',  # model name column
        'id_columns': ['#Params (B)', 'Average'],  # additional columns to be used for unique identification of the model
        'metric_columns': ['IFEval', 'BBH', 'MATH Lvl 5', 'GPQA', 'MUSR', 'MMLU-PRO'],  # columns containing performance data
        'preference_functions': [
            PreferenceFunctionEnum.LINEAR,
            PreferenceFunctionEnum.LINEAR,
            PreferenceFunctionEnum.LINEAR,
            PreferenceFunctionEnum.GAUSSIAN,
            PreferenceFunctionEnum.GAUSSIAN,
            PreferenceFunctionEnum.GAUSSIAN
        ],  # preference functions by metric
        'user_specified_weights': [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667],  # user-specified weight by metric
        'column_maximization': [1, 1, 1, 1, 1, 1],  # 1 - metric values maximization, -1 - metric values minimization
        'rank_output_file': r'open-llm-s4-rank.csv',  # output file to be created
        'preference_parameters': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    }

    Pipeline().run(config, False) 
