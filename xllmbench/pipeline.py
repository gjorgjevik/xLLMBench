import csv
import os

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

    def run(self, experiment_config: dict):
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
            user_specified_weights=user_specified_weights)

        results = [['model', 'predicted rank', 'predicted score', 'positive preference flow', 'negative preference flow']]
        for key in sorted(ranks, key=ranks.get):
            results.append([key, ranks[key], scores[key], positive_preference_flow[key], negative_preference_flow[key]])

        self.write(os.path.join(directory, rank_output_file), results)

        matrix = [['model'] + keys]
        for key, l in zip(keys, preference_matrix.tolist()):
            matrix.append([key] + l)

        self.write(os.path.join(directory, rank_output_file.replace('rank.csv', 'preference-matrix.csv')), matrix)



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
        'rank_output_file': r'open-llm-s4-rank.csv'  # output file to be created
    }

    Pipeline().run(config)
  
