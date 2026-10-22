CONDITIONS = {
    "is_immigrant": lambda row: row["target_ident"] == "immigrants",
    "is_women": lambda row: row["target_ident"] == "women",
    "is_trans": lambda row: row["target_ident"] == "trans people",
    "is_gay": lambda row: row["target_ident"] == "gay people",
    "is_muslim": lambda row: row["target_ident"] == "Muslims",
    "is_disabled": lambda row: row["target_ident"] == "disabled people",
    "is_black": lambda row: row["target_ident"] == "black people",
    "is_lgbtq": lambda row: row["target_ident"] in ["trans people", "gay people"],
    "is_slur": lambda row: str(row["functionality"]).startswith("slur"),
    "is_profanity": lambda row: str(row["functionality"]).startswith("profanity"),
    "is_non_prot_target": lambda row: str(row["functionality"]).startswith("target_")
}

RULES = {
    "immigrant_non_hate": {
        "condition": "is_immigrant",
        "label": "non-hateful",
        "priority": 1,
    },
    "immigrant_hate": {
        "condition": "is_immigrant",
        "label": "hateful",
        "priority": 1,
    },
    "women_non_hate": {
        "condition": "is_women",
        "label": "non-hateful",
        "priority": 1,
    },
    "women_hate": {
        "condition": "is_women",
        "label": "hateful",
        "priority": 1,
    },
    "trans_non_hate": {
        "condition": "is_trans",
        "label": "non-hateful",
        "priority": 1,
    },
    "trans_hate": {
        "condition": "is_trans",
        "label": "hateful",
        "priority": 1,
    },
    "gay_non_hate": {
        "condition": "is_gay",
        "label": "non-hateful",
        "priority": 1,
    },
    "gay_hate": {
        "condition": "is_gay",
        "label": "hateful",
        "priority": 1,
    },
    "muslim_non_hate": {
        "condition": "is_muslim",
        "label": "non-hateful",
        "priority": 1,
    },
    "muslim_hate": {
        "condition": "is_muslim",
        "label": "hateful",
        "priority": 1,
    },
    "disabled_non_hate": {
        "condition": "is_disabled",
        "label": "non-hateful",
        "priority": 1,
    },
    "disabled_hate": {
        "condition": "is_disabled",
        "label": "hateful",
        "priority": 1,
    },
    "black_non_hate": {
        "condition": "is_black",
        "label": "non-hateful",
        "priority": 1,
    },
    "black_hate": {
        "condition": "is_black",
        "label": "hateful",
        "priority": 1,
    },
    "lgbtq_non_hate": {
        "condition": "is_lgbtq",
        "label": "non-hateful",
        "priority": 1,
    },
    "lgbtq_hate": {
        "condition": "is_lgbtq",
        "label": "hateful",
        "priority": 1,
    },
    "profanity_hate": {
        "condition": "is_profanity",
        "label": "hateful",
        "priority": 0,
    },
    "profanity_non_hate": {
        "condition": "is_profanity",
        "label": "non-hateful",
        "priority": 0,
    },
    "slur_hate": {
        "condition": "is_slur",
        "label": "hateful",
        "priority": 0,
    },
    "slur_non_hate": {
        "condition": "is_slur",
        "label": "non-hateful",
        "priority": 0,
    },
    "non_prot_target_hate": {
        "condition": "is_non_prot_target",
        "label": "hateful",
        "priority": 0,
    },
}

GROUP_TEMPLATES = {
    "null": [],
    "A": ["immigrant_hate"],
    "B": ["immigrant_non_hate"],
    "C": ["women_hate"],
    "D": ["women_non_hate"],
    "E": ["lgbtq_hate"],
    "F": ["lgbtq_non_hate"],
    "G": ["profanity_hate"],
    "H": ["profanity_non_hate"],
    "I": ["slur_hate"],
    "J": ["slur_non_hate"],
    "K": ["immigrant_non_hate", "slur_hate"],
    "L": ["women_non_hate", "profanity_hate"],
    "M": ["lgbtq_non_hate", "profanity_hate"],
    "N": ["profanity_non_hate", "slur_non_hate"],
    "O": ["immigrant_hate", "women_hate", "lgbtq_hate"],
    "P": ["immigrant_non_hate", "women_non_hate", "lgbtq_non_hate"],
}

HATECHECK_EXPERIMENTS = {
    # Naming convention: first digit for number of groups, last three as experiment ID
    2001: {
        "groups": ["A", "null"]
    },
    2002: {
        "groups": ["B", "null"]
    },
    2003: {
        "groups": ["K", "null"]
    },
    2004: {
        "groups": ["P", "null"]
    },
    2005: {
        "groups": ["D", "I"]
    },
    2006: {
        "groups": ["M", "G"]
    },
    3001: {
        "groups": ["A", "C", "E"]
    },
    3002: {
        "groups": ["B", "D", "null"]
    },
    3003: {
        "groups": ["I", "L", "P"]
    },
    4001: {
        "groups": ["A", "C", "E", "null"]
    },
    4002: {
        "groups": ["B", "D", "F", "null"]
    },
    4003: {
        "groups": ["I", "L", "O", "P"]
    },
    5001: {
        "groups": ["A", "C", "E", "G", "I"]
    },
    5002: {
        "groups": ["B", "D", "F", "H", "null"]
    },
    5003: {
        "groups": ["I", "L", "M", "O", "P"]
    }
}


# ===============================================
# VALIDATION SETUP
# ===============================================

# Embedders
EMBEDDERS = {
    "mbert": "google-bert/bert-base-multilingual-cased",
    "roberta-base": "FacebookAI/roberta-base",
    "roberta-large": "FacebookAI/roberta-large",
    "sbert": "sentence-transformers/all-MiniLM-L12-v2",
}

# Rules used to build the synthetic validation groups: each group labels as non-hateful all examples targeting some identities
VALIDATION_RULES = ["immigrant_non_hate", "women_non_hate", "trans_non_hate", "gay_non_hate", "muslim_non_hate", "disabled_non_hate", "black_non_hate"]

VALIDATION_VARIANTS = ["balanced", "imbalanced", "bias"]


def build_group_templates(seed: int = 42) -> dict:
    """
    All combinations of VALIDATION_RULES (127 templates), randomly permuted, with "null" (gold labels) prepended.
    """
    from itertools import combinations
    import random

    rule_names = [r for r in RULES.keys() if r in VALIDATION_RULES]
    templates = {}
    idx = 1
    for r in range(1, len(rule_names) + 1):
        for combo in combinations(rule_names, r):
            templates[str(idx)] = list(combo)
            idx += 1

    # Random permutation: experiments take templates in this order, which amounts to random sampling
    items = list(templates.items())
    random.Random(seed).shuffle(items)
    return {"null": [], **dict(items)}


def _build_balanced_experiments(group_templates: dict) -> dict:
    """
    Experiments with 2 to 5 groups, interleaved by number of groups (2, 3, 4, 5, 2, 3, ...).
    """
    from collections import defaultdict

    group_ids = list(group_templates.keys())
    experiments = {}

    # 2 groups: consecutive pairs of templates
    exp_num = 20000
    for i in range(0, len(group_ids) - 1, 2):
        experiments[str(exp_num)] = {"groups": [group_ids[i], group_ids[i + 1]]}
        exp_num += 1

    # 3 to 5 groups: consecutive chunks of templates (incomplete final chunks are skipped)
    exp_num = 22000
    for num_groups in range(3, 6):
        for i in range(0, len(group_ids), num_groups):
            chunk = group_ids[i:i + num_groups]
            if len(chunk) < num_groups:
                continue
            experiments[str(exp_num)] = {"groups": chunk}
            exp_num += 1

    # Interleave by number of groups
    grouped = defaultdict(list)
    for exp_id, exp_config in experiments.items():
        grouped[len(exp_config["groups"])].append((exp_id, exp_config))
    ordered = {}
    for i in range(max(len(v) for v in grouped.values())):
        for num_groups in range(2, 6):
            if i < len(grouped[num_groups]):
                exp_id, exp_config = grouped[num_groups][i]
                ordered[exp_id] = exp_config
    return ordered


def _imbalanced_ratios(n_groups: int, rng, min_ratio: float = 0.1) -> list[float]:
    """
    Random group size ratios in [min_ratio, 1 - (n_groups - 1) * min_ratio], rounded to 2 decimals and summing exactly to 1.
    """
    while True:
        remaining = 1 - n_groups * min_ratio
        ratios = [round(float(r), 2) for r in min_ratio + rng.dirichlet([1.0] * n_groups) * remaining]
        ratios[-1] = round(1 - sum(ratios[:-1]), 2)
        if ratios[-1] >= min_ratio and sum(ratios) == 1.0: # HateCheckData requires an exact sum
            return ratios


def build_validation_experiments(variant: str = "balanced", group_templates: dict | None = None, seed: int = 42) -> dict:
    """
    Experiment configs for the synthetic HateCheck validation.
        - balanced: groups of equal size (Sec. 4)
        - imbalanced: random group size ratios (Appendix C.1)
        - bias: annotator-specific bias on a random functionality (Appendix C.2)
    The three variants share the same group compositions, in the same order.
    """
    import numpy as np

    if variant not in VALIDATION_VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Allowed: {VALIDATION_VARIANTS}")
    if group_templates is None:
        group_templates = build_group_templates(seed=seed)
    balanced = _build_balanced_experiments(group_templates)
    if variant == "balanced":
        return balanced

    rng = np.random.default_rng(seed)
    experiments = {}
    exp_num = 23000 if variant == "imbalanced" else 24000
    for exp_config in balanced.values():
        if variant == "imbalanced":
            experiments[str(exp_num)] = {
                "groups": exp_config["groups"],
                "imbalanced_groups": _imbalanced_ratios(len(exp_config["groups"]), rng),
            }
        else:
            experiments[str(exp_num)] = {
                "groups": exp_config["groups"],
                "individual_bias": 0.2,
            }
        exp_num += 1
    return experiments
