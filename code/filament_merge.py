from itertools import combinations, islice

import numpy as np
import torch
from torch import nn
from pycocotools import mask as mask_utils
from scipy.spatial import cKDTree

from filament_classifier import FEATURE_NAMES

PAIR_FEATURE_NAMES = ([f"mean_{name}" for name in FEATURE_NAMES]
                      + [f"difference_{name}" for name in FEATURE_NAMES]
                      + ["pca_angle_radians", "minimum_distance"])


class PairMergeMLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def pair_geometry(rle):
    mask = mask_utils.decode(rle)
    x, y, width, height = mask_utils.toBbox(rle).astype(int)
    yy, xx = np.nonzero(mask[y:y + height, x:x + width])
    points = np.column_stack((xx + x, yy + y)).astype(np.float64)
    if not len(points):
        raise ValueError("Empty candidate")
    centered = points - points.mean(axis=0)
    values, vectors = np.linalg.eigh(centered.T @ centered / len(points))
    direction = vectors[:, -1] if values[-1] - values[0] > 1e-12 else None
    return points, cKDTree(points), direction


def pair_features(a, b, geometry_a, geometry_b):
    points_a, tree_a, direction_a = geometry_a
    points_b, tree_b, direction_b = geometry_b
    angle = (np.arccos(np.clip(abs(direction_a @ direction_b), 0, 1))
             if direction_a is not None and direction_b is not None else 0.)
    distance = (tree_b.query(points_a)[0].min() if len(points_a) <= len(points_b)
                else tree_a.query(points_b)[0].min())
    return np.r_[(a + b) / 2, np.abs(a - b), angle, distance]


def purity_matches(rles, ground_truth):
    # COCO's crowd denominator is the candidate area: intersection / |candidate|.
    if not ground_truth:
        return np.zeros((len(rles), 0), dtype=bool)
    return mask_utils.iou(rles, ground_truth, [1] * len(ground_truth)) > 0.7


def pair_batches(entries, features, batch_size=4096):
    if len(entries) < 2:
        return
    geometry = [pair_geometry(rle) for _, rle in entries]
    pairs = combinations(range(len(entries)), 2)
    while batch := list(islice(pairs, batch_size)):
        vectors = np.asarray([pair_features(features[i], features[j], geometry[i], geometry[j])
                              for i, j in batch])
        yield batch, vectors


def merge_candidates(entries, features, model, threshold, score_writer=None):
    from filament_classifier import predict_probabilities

    parents = list(range(len(entries)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for pairs, vectors in pair_batches(entries, features):
        probabilities = predict_probabilities(model, vectors)
        for (i, j), probability in zip(pairs, probabilities):
            decision = probability >= threshold
            if score_writer is not None:
                score_writer.writerow([entries[i][0], entries[j][0], float(probability), int(decision)])
            if decision:
                parents[root(j)] = root(i)
    groups = {}
    for i in range(len(entries)):
        groups.setdefault(root(i), []).append(i)
    merged, membership = [], []
    for indices in groups.values():
        identifier = entries[indices[0]][0]
        rle = (entries[indices[0]][1] if len(indices) == 1 else
               mask_utils.merge([entries[i][1] for i in indices]))
        merged.append((identifier, rle))
        membership.extend((identifier, entries[i][0]) for i in indices)
    return merged, membership
