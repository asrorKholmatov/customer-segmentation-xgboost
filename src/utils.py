"""Export a trained XGBoost model to compact JSON trees the browser can evaluate."""
import json

import numpy as np
import xgboost as xgb


def compact_tree(tree: dict, names: list[str]) -> list[list]:
    """Flatten a dumped tree into parallel arrays indexed by node id:
    [feature (-1 = leaf), threshold or leaf value, yes, no, missing]."""
    nodes = {}
    stack = [tree]
    while stack:
        n = stack.pop()
        nodes[n["nodeid"]] = n
        stack.extend(n.get("children", []))
    size = max(nodes) + 1
    f, t, yes, no, miss = [-1] * size, [0.0] * size, [0] * size, [0] * size, [0] * size
    for i, n in nodes.items():
        if "leaf" in n:
            t[i] = round(n["leaf"], 6)
        else:
            f[i] = names.index(n["split"])
            t[i] = n["split_condition"]
            yes[i], no[i], miss[i] = n["yes"], n["no"], n["missing"]
    return [f, t, yes, no, miss]


def tree_margins(trees: list[list], x: np.ndarray, n_classes: int) -> np.ndarray:
    """Sum leaf values per class - the same algorithm web/app.js runs."""
    out = np.zeros(n_classes)
    for k, (f, t, yes, no, miss) in enumerate(trees):
        i = 0
        while f[i] >= 0:
            v = x[f[i]]
            i = miss[i] if np.isnan(v) else (yes[i] if v < t[i] else no[i])
        out[k % n_classes] += t[i]
    return out


def export_trees(model: xgb.XGBClassifier, X_check: np.ndarray, names: list[str]) -> dict:
    """Dump trees and derive the per-class intercept, verifying against XGBoost itself."""
    dumped = model.get_booster().get_dump(dump_format="json")
    trees = [compact_tree(json.loads(d), names) for d in dumped]
    n_classes = len(model.classes_)
    margins = model.predict(X_check, output_margin=True)
    sums = np.array([tree_margins(trees, x, n_classes) for x in X_check])
    intercepts = margins - sums
    assert np.allclose(intercepts, intercepts[0], atol=1e-4), "intercept is not constant"
    return {"trees": trees, "base_margin": intercepts[0].round(6).tolist()}
