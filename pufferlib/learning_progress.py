from pdb import set_trace as T

import numpy as np
from collections import defaultdict

class BidirectionalLearningProgess:
    def __init__(self, max_num_levels = 8192, ema_alpha = 0.001, p_theta = 0.05):
        self.max_num_levels = max_num_levels
        self.ema_alpha = ema_alpha
        self.p_theta = p_theta
        self.outcomes = {}
        for i in range(max_num_levels):
            self.outcomes[i] = []
        self.ema_tsr = None
        self.p_fast = None
        self.p_slow = None
        self.random_baseline = None
        self.task_success_rate = None
        self.task_sampled_tracker = []

        # should we continue collecting 
        #  or if we have enough data to update the learning progress
        self.collecting = True

    def _update(self):
        task_success_rates = np.array([np.mean(self.outcomes[i]) for i in range(self.max_num_levels)])

        if self.random_baseline is None:
            # Assume that any perfect success rate is actually 75% due to evaluation precision.
            # Prevents NaN probabilities and prevents task from being completely ignored.
            # high_success_idxs = np.where(task_success_rates > 0.75)
            # high_success_rates = task_success_rates[high_success_idxs]
            #  warnings.warn(
            #     f"Tasks {high_success_idxs} had very high success rates {high_success_rates} for random baseline. Consider removing them from the training set of tasks.")
            self.random_baseline = np.minimum(task_success_rates, 0.75)

        # Update task scores
        normalized_task_success_rates = np.maximum(
            task_success_rates - self.random_baseline, np.zeros(task_success_rates.shape)) / (1.0 - self.random_baseline)

        if self._p_fast is None:
            # Initial values
            self._p_fast = normalized_task_success_rates
            self._p_slow = normalized_task_success_rates
            self._p_true = task_success_rates
        else:
            # Exponential moving average
            self._p_fast = (normalized_task_success_rates * self.ema_alpha) + (self._p_fast * (1.0 - self.ema_alpha))
            self._p_slow = (self._p_fast * self.ema_alpha) + (self._p_slow * (1.0 - self.ema_alpha))
            self._p_true = (task_success_rates * self.ema_alpha) + (self._p_true * (1.0 - self.ema_alpha))

        self.task_rates = task_success_rates    # Logging only
        self._stale_dist = True
        self.task_dist = None

        return task_success_rates
    
    def collect_data(self, infos):
        for k, v in infos.items():
            if 'tasks' in k:
                task_id = int(k.split('/')[1])
                for res in v:
                    self.outcomes[task_id].append(res)

        self.task_sampled_tracker = [int(bool(o)) for k, o in self.outcomes.items()]
        if sum(self.task_sampled_tracker) >= self.max_num_levels:
            self.task_success_rate = np.array([np.mean(self.outcomes[i]) for i in range(self.max_num_levels)])
            self.collecting = False
            self.task_sampled_tracker = []
    
    def continue_collecting(self):
        return self.collecting
    
    def _learning_progress(self, reweight: bool = True) -> float:
        """
        Compute the learning progress metric for the given task.
        """
        fast = self._reweight(self._p_fast) if reweight else self._p_fast
        slow = self._reweight(self._p_slow) if reweight else self._p_slow

        return abs(fast - slow)

    def _reweight(self, p: np.ndarray, p_theta: float = 0.1) -> float:
        """
        Reweight the given success rate using the reweighting function from the paper.
        """
        numerator = p * (1.0 - p_theta)
        denominator = p + p_theta * (1.0 - 2.0 * p)
        return numerator / denominator

    def _sigmoid(self, x: np.ndarray):
        """ Sigmoid function for reweighting the learning progress."""
        return 1 / (1 + np.exp(-x))

    def _sample_distribution(self):
        """ Return sampling distribution over the task space based on the learning progress."""
        if not self._stale_dist:
            # No changes since distribution was last computed
            return self.task_dist

        task_dist = np.ones(self.num_tasks) / self.num_tasks

        learning_progress = self._learning_progress()
        posidxs = [i for i, lp in enumerate(learning_progress) if lp > 0 or self._p_true[i] > 0]
        any_progress = len(posidxs) > 0

        subprobs = learning_progress[posidxs] if any_progress else learning_progress
        std = np.std(subprobs)
        subprobs = (subprobs - np.mean(subprobs)) / (std if std else 1)  # z-score
        subprobs = self._sigmoid(subprobs)  # sigmoid
        subprobs = subprobs / np.sum(subprobs)  # normalize
        if any_progress:
            # If some tasks have nonzero progress, zero out the rest
            task_dist = np.zeros(len(learning_progress))
            task_dist[posidxs] = subprobs
        else:
            # If all tasks have 0 progress, return uniform distribution
            task_dist = subprobs

        self.task_dist = task_dist
        self._stale_dist = False
        # clear the outcome dict
        self.outcomes = defaultdict(list)
        for i in range(self.max_num_levels):
            self.outcomes[i] = []
        self.collecting = True
        return task_dist
    
    def calculate_dist(self):
        self._update()
        return self._sample_distribution()
