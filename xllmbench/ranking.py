import numpy as np
from pymcdm.methods import TOPSIS, VIKOR

from xllmbench.preference import PreferenceFunctionEnum


class PrometheeIIBasedRanking:
    @staticmethod
    def pairwise_distance(decision_matrix: np.ndarray) -> np.ndarray:
        tensor = []
        for j in range(decision_matrix.shape[1]):
            column = decision_matrix[:, j].reshape([-1, 1])
            difference = column - column.transpose()
            tensor.append(difference)
        return np.array(tensor)

    @staticmethod
    def min_max(pairwise_distance_tensor: np.ndarray, column_maximization: list) -> np.ndarray:
        return pairwise_distance_tensor * np.array(column_maximization).reshape([-1, 1, 1])

    @staticmethod
    def apply_preference_functions(preference_functions: list, preference_parameters: list,
                                   pairwise_distance_tensor: np.ndarray) -> np.ndarray:
        tensor = []
        for i in range(pairwise_distance_tensor.shape[0]):
            parameters = None
            matrix = pairwise_distance_tensor[i, :, :]
            if preference_functions[i] == PreferenceFunctionEnum.USUAL:
                parameters = []
            elif preference_functions[i] == PreferenceFunctionEnum.LINEAR:
                coefficient = preference_parameters[i]
                min_value = np.min(matrix)
                max_value = np.max(matrix)
                parameters = [min_value + coefficient * abs(min_value), max_value - coefficient * abs(max_value)]
            elif preference_functions[i] == PreferenceFunctionEnum.GAUSSIAN:
                coefficient = preference_parameters[i]
                s = np.std(matrix)
                parameters = [s + coefficient * s]

            preference = np.apply_along_axis(
                PreferenceFunctionEnum.get_function(preference_functions[i]),
                0,
                np.array(matrix),
                parameters)

            tensor.append(preference)

        return np.array(tensor)

    @staticmethod
    def average_preference_index(pairwise_distance_tensor: np.ndarray, weights: np.ndarray) -> np.ndarray:
        i, j, k = pairwise_distance_tensor.shape
        weights = weights.reshape([-1, 1, 1])
        average_matrix = np.average(pairwise_distance_tensor * weights, axis=0)
        assert average_matrix.shape == (j, k)
        return average_matrix

    @staticmethod
    def positive_preference_flow(preference_matrix: np.ndarray, n: int) -> np.ndarray:
        return 1 / (n - 1) * np.sum(preference_matrix, axis=1)

    @staticmethod
    def negative_preference_flow(preference_matrix: np.ndarray, n: int) -> np.ndarray:
        return 1 / (n - 1) * np.sum(preference_matrix, axis=0)

    @staticmethod
    def rank(models: list, net_flow: np.ndarray,
             positive_preference_flow: np.ndarray, negative_preference_flow: np.ndarray) -> list:
        previous_rank = 0
        previous_score = np.inf

        scoring = dict()
        ranking = dict()
        pos = dict()
        neg = dict

        for score, p, n, model in sorted(zip(net_flow, positive_preference_flow, negative_preference_flow, models),
                                         reverse=True):
            if score == previous_score:
                current_rank = previous_rank
            else:
                current_rank = previous_rank + 1

            scoring[model] = score
            ranking[model] = current_rank
            pos[model] = p
            neg[model] = n

            previous_rank = current_rank
            previous_score = score

        return [ranking, scoring, pos, neg]

    def run(self, keys: list, decision_matrix_raw: list,
            column_maximization: list, preference_functions: list,
            user_specified_weights: list, preference_parameters: list):

        # convert to nparray
        decision_matrix = np.array(decision_matrix_raw, ndmin=2)
        m, n = decision_matrix.shape

        # calculate pairwise differences for each column
        pairwise_distance_tensor_before = self.pairwise_distance(decision_matrix)

        # minimize or maximize
        pairwise_distance_tensor_before = self.min_max(pairwise_distance_tensor_before, column_maximization)

        # apply user specified preference function to each column
        pairwise_distance_tensor = self.apply_preference_functions(
            preference_functions, preference_parameters, pairwise_distance_tensor_before)

        weights = np.array(user_specified_weights)

        # average preference index
        preference_matrix = self.average_preference_index(pairwise_distance_tensor, weights)

        # positive preference flow by method
        positive_preference_flow = self.positive_preference_flow(preference_matrix, n)

        # negative preference flow by method
        negative_preference_flow = self.negative_preference_flow(preference_matrix, n)

        # preference flow difference
        net_flow = positive_preference_flow - negative_preference_flow

        [ranking, scoring, pos, neg] = self.rank(keys, net_flow, positive_preference_flow, negative_preference_flow)

        return [pairwise_distance_tensor_before, pairwise_distance_tensor, preference_matrix, pos, neg, ranking, scoring]


class TopsisBasedRanking:
    @staticmethod
    def rank(models: list, net_flow: np.ndarray) -> list:
        previous_rank = 0
        previous_score = np.inf

        scoring = dict()
        ranking = dict()

        for score, model in sorted(zip(net_flow, models), reverse=True):
            if score == previous_score:
                current_rank = previous_rank
            else:
                current_rank = previous_rank + 1

            scoring[model] = score
            ranking[model] = current_rank

            previous_rank = current_rank
            previous_score = score

        return [ranking, scoring]

    def run(self, keys: list, decision_matrix_raw: list,
            column_maximization: list, preference_functions: list,
            user_specified_weights: list, preference_parameters: list):

        # convert to nparray
        decision_matrix = np.array(decision_matrix_raw, ndmin=2)

        method = TOPSIS()

        weights = np.array(user_specified_weights)
        types = np.array(column_maximization)
        scores = method(decision_matrix, weights, types)

        [ranking, scoring] = self.rank(keys, scores)

        return [ranking, scoring]


class VikorBasedRanking:
    @staticmethod
    def rank(models: list, net_flow: np.ndarray) -> np.ndarray:
        previous_rank = 0
        previous_score = np.inf

        scoring = dict()
        ranking = dict()

        for score, model in sorted(zip(net_flow, models), reverse=True):
            if score == previous_score:
                current_rank = previous_rank
            else:
                current_rank = previous_rank + 1

            scoring[model] = score
            ranking[model] = current_rank

            previous_rank = current_rank
            previous_score = score

        return [ranking, scoring]

    def run(self, keys: list, decision_matrix_raw: list,
            column_maximization: list, preference_functions: list,
            user_specified_weights: list, preference_parameters: list):

        # convert to nparray
        decision_matrix = np.array(decision_matrix_raw, ndmin=2)

        method = VIKOR()

        weights = np.array(user_specified_weights)
        types = np.array(column_maximization)
        scores = method(decision_matrix, weights, types)
        scores = -1 * scores  # see pymcdm documentation for VIKOR method

        [ranking, scoring] = self.rank(keys, scores)

        return [ranking, scoring]
