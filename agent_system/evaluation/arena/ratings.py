"""Fixed-sample paired-score bounds and order-independent Elo estimates."""
import math
import numpy as np


def score_interval(block_scores, alpha=0.05):
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    if any(not 0 <= x <= 1 for x in block_scores):
        raise ValueError("scores must be in [0,1]")
    if not block_scores:
        return [0.0, 1.0]
    mean = sum(block_scores) / len(block_scores)
    half = math.sqrt(math.log(2 / alpha) / (2 * len(block_scores)))
    return [max(0.0, mean-half), min(1.0, mean+half)]


def fit_elo(agent_ids, pairs, anchor):
    """pairs: (agent_a, agent_b, score_sum_a, number_of_games)."""
    ids = list(agent_ids)
    index = {key: i for i, key in enumerate(ids)}
    strength = np.zeros(len(ids))
    prior_precision = 1 / math.log(10)**2  # 400 Elo prior SD
    for _ in range(100):
        grad = prior_precision * strength
        hessian = np.eye(len(ids)) * prior_precision
        for a, b, score, n in pairs:
            i, j = index[a], index[b]
            p = 1 / (1 + math.exp(float(np.clip(strength[j]-strength[i], -40, 40))))
            delta = n*p-score
            grad[i] += delta
            grad[j] -= delta
            variance = n*p*(1-p)
            hessian[i, i] += variance
            hessian[j, j] += variance
            hessian[i, j] -= variance
            hessian[j, i] -= variance
        update = np.linalg.solve(hessian, grad)
        strength -= update
        if np.max(np.abs(update)) < 1e-9:
            break
    offset = strength[index[anchor]]
    return {key: 1500 + float(strength[i]-offset)*400/math.log(10) for key, i in index.items()}


def update_elo_pair(rating_a, rating_b, game_scores, k=20):
    if len(game_scores) != 2 or any(not 0 <= s <= 1 for s in game_scores):
        raise ValueError("A complete two-game seat-swapped pair is required")
    expected = 1 / (1 + 10**((rating_b-rating_a)/400))
    change = k * (sum(game_scores)/2 - expected)
    return rating_a+change, rating_b-change
