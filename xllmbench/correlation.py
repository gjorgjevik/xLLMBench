import os
from typing import LiteralString

import numpy as np
import pandas as pd
import scipy as sp
from matplotlib import pyplot as plt

plt.rcParams['font.size'] = 7


class CorrelationAnalyzer:
    @staticmethod
    def read(files: dict):
        rank_matrix = None
        for key, file in files.items():
            data = pd.read_csv(file, encoding='utf-8', index_col='key', delimiter=',')
            data[key] = data['predicted rank']
            new_data = data[[key]].copy()
            if rank_matrix is None:
                rank_matrix = data[[key]].copy()
            else:
                rank_matrix = pd.concat([rank_matrix, new_data], axis=1).copy()

        return rank_matrix

    @staticmethod
    def correlate(decision_matrix_raw: list):
        correlations = []
        p_values = []
        decision_matrix = np.array(decision_matrix_raw, ndmin=2)

        for i in range(decision_matrix.shape[1]):
            correlations_local = []
            p_values_local = []

            metric_1 = decision_matrix[:, i]
            for j in range(decision_matrix.shape[1]):
                metric_2 = decision_matrix[:, j]
                c, p_value = sp.stats.spearmanr(metric_1, metric_2)

                correlations_local.append(c)
                p_values_local.append(p_value)

            correlations.append(correlations_local)
            p_values.append(p_values_local)

        return [correlations, p_values]

    @staticmethod
    def correlation_heatmap(file: LiteralString, data_raw: np.ndarray, rows: list, columns: list, cbarlabel: str):
        cbar_kw = {'cmap': 'OrRd'}
        data = np.array(data_raw, ndmin=2)

        fig, ax = plt.subplots(tight_layout=True)
        im = ax.imshow(data, **cbar_kw)

        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

        ax.set_xticks(np.arange(len(columns)), labels=columns)
        ax.set_yticks(np.arange(len(rows)), labels=rows)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        for i in range(len(rows)):
            for j in range(len(columns)):
                text = ax.text(j, i, f'{data[i, j]:.2f}', ha="center", va="center",
                               color=("black" if data[i, j] < 0.95 else "white"))

        plt.savefig(file, dpi=300)


if __name__ == '__main__':
    output_directory = r'\path\to\output\file'
    d = {
        'S1': r'path\to\rank\csv\file',
        'S2': r'path\to\rank\csv\file',
        'S3': r'path\to\rank\csv\file',
    }
    analyzer = CorrelationAnalyzer()
    rank_matrix = analyzer.read(d)

    [correlations, p_values] = analyzer.correlate(rank_matrix.values)
    analyzer.correlation_heatmap(file=os.path.join(output_directory, 'spearman.jpg'),
                                 data_raw=correlations,
                                 rows=list(d.keys()),
                                 columns=list(d.keys()),
                                 cbarlabel='Spearman rho')
