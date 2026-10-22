import sys
import pathlib
import argparse
import itertools

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from exp_config import EMBEDDERS
from exp_utils import MDAgreementData, MultiPICoData, seed_everything, demographic_cohort, COUNTRY_TO_CONTINENT
from flocke_utils import FlockeExperiment, likelihood_score, predicted_groups, pairwise_js_divergences
from metrics import cross_entropy


SAVE_PATH_TEMPLATE = "{dataset_name}_g{ntypes}_h{hidden[0]}x{hidden[1]}_pr{prior_tempering_beta}_pt{posterior_tempering_beta_train}_pi{posterior_tempering_beta_inference}_reop{p_reoptimization_period}_ep{max_epochs}_ck{use_best_checkpoint:d}_run{run}.pth"

HIDDEN = [8, 8]
NCLASSES = 2
TRAIN_PARAMS = {
    'max_epochs': 200,
    'lr': 2e-3,
    'batch_size': 64,
    'return_estimates': True,
    'prior_tempering_beta': 0.9,
    'posterior_tempering_beta_train': 1.0,
    'posterior_tempering_beta_inference': 1.0,
    'p_reoptimization_period': 100,
    'use_best_checkpoint': False, # keep the weights of the last epoch
    'normalize_embeddings': False,
    'pca_dim': None,
}

DATASETS = {
    'md': {
        'name': 'MD',
        'description': 'MD-Agreement',
        'data_dir': ROOT / 'data' / 'MD',
        'required_files': ['MD_train.json', 'MD_dev.json', 'MD_test.json'],
        'p_reoptimization_period': 100,
    },
    'mp': {
        'name': 'MPen',
        'description': 'MultiPICo (gb, us, au language varieties)',
        'data_dir': ROOT / 'data' / 'multipico',
        'required_files': ['MP_train.json', 'MP_annotators_meta.json'],
        'language_varieties': {'gb', 'us', 'au'},
        'p_reoptimization_period': 100,
    },
}

# Socio-demographic traits of MultiPICo annotators tested for association with the latent groups
DEMOGRAPHICS = ["Age", "Gender", "Ethnicity simplified", "Country of birth", "Country of residence", "Nationality", "Student status", "Employment status"]

TOP_RELEVANT_ITEMS = 100


def get_args():
    parser = argparse.ArgumentParser(description="Apply FLOCKE to real-world datasets with disaggregated annotations.")
    parser.add_argument('--datasets', nargs='+', choices=list(DATASETS.keys()), default=list(DATASETS.keys()), help='Datasets to analyze.')
    parser.add_argument('--ntypes', nargs='+', type=int, default=[1, 2, 3, 4, 5], help='Numbers of groups to test for group number detection.')
    parser.add_argument('--runs', type=int, default=5, help='Number of runs for each number of groups.')
    parser.add_argument('--num-groups', type=int, default=3, help='Number of groups used for the analyses (selected via group number detection).')
    parser.add_argument('--embedder', choices=list(EMBEDDERS.keys()), default='mbert', help='Text embedder used as input features.')
    parser.add_argument('--max-epochs', type=int, default=TRAIN_PARAMS['max_epochs'], help='Training epochs.')
    parser.add_argument('--output-dir', type=str, default=str(ROOT / 'results' / 'experiments'), help='Directory for results.')
    parser.add_argument('--embeddings-dir', type=str, default=str(ROOT / 'calculated_embeddings'), help='Directory for cached embeddings.')
    parser.add_argument('--models-dir', type=str, default=str(ROOT / 'trained_models'), help='Directory for trained models.')
    parser.add_argument('--verbose', action='store_true', help='Print training progress.')
    return parser.parse_args()


# DATA AND TRAINING =============================================================

def load_dataset(dataset_key: str):
    config = DATASETS[dataset_key]
    if dataset_key == 'md':
        return MDAgreementData(data_dir=str(config['data_dir']))
    return MultiPICoData(data_dir=str(config['data_dir']), lang=None, language_varieties=config['language_varieties'])


def _is_int_like(value) -> bool:
    return str(value).strip().lstrip('+-').isdigit()


def use_split_in_id(data_files: dict) -> bool:
    """
    Item ids may repeat across splits: when they are integers, the split is prepended to make embedding ids unique.
    """
    all_ids = [example_id for split_data in data_files.values() for example_id in split_data.keys()]
    if all(_is_int_like(example_id) for example_id in all_ids):
        return True
    if len(set(map(str, all_ids))) != len(all_ids):
        raise ValueError("Item ids are not integers and are not unique across splits: cannot index embeddings.")
    return False


def train_or_load(args, dataset_key: str, data_files: dict, ntypes: int, run: int) -> FlockeExperiment:
    """
    Prepare data and load the corresponding trained model, or train it on the training split (seeded by run) and save it.
    """
    config = DATASETS[dataset_key]
    train_params = {**TRAIN_PARAMS, 'max_epochs': args.max_epochs, 'p_reoptimization_period': config['p_reoptimization_period']}
    dr_exp = FlockeExperiment(
        model_config={'ntypes': ntypes, 'nclasses': NCLASSES, 'hidden': HIDDEN},
        train_params=train_params,
        embedder_name=EMBEDDERS[args.embedder],
        data_files=data_files,
    )
    embeddings_path = pathlib.Path(args.embeddings_dir) / args.embedder / f"{config['name']}.npz"
    dr_exp.prepare_data(embeddings_path=str(embeddings_path), from_npz=True, use_split_in_id=use_split_in_id(data_files), verbose=args.verbose)

    model_path = pathlib.Path(args.models_dir) / args.embedder / config['name'] / SAVE_PATH_TEMPLATE.format(
        dataset_name=config['name'], ntypes=ntypes, hidden=HIDDEN,
        prior_tempering_beta=int(train_params['prior_tempering_beta'] * 100),
        posterior_tempering_beta_train=int(train_params['posterior_tempering_beta_train'] * 100),
        posterior_tempering_beta_inference=int(train_params['posterior_tempering_beta_inference'] * 100),
        p_reoptimization_period=train_params['p_reoptimization_period'], max_epochs=train_params['max_epochs'],
        use_best_checkpoint=train_params['use_best_checkpoint'], run=run)

    seed_everything(run)
    loaded = dr_exp.load_or_train_model(filepath=str(model_path), verbose=args.verbose)
    if not loaded:
        dr_exp.save_model(str(model_path), verbose=args.verbose)
    return dr_exp


# 1. GROUP NUMBER DETECTION =============================================================

def groups_calibration(args, dataset_key: str, data_files: dict, output_dir: pathlib.Path) -> pd.DataFrame:
    """
    Train FLOCKE with each number of groups and run, and save the likelihood score on each split and the predicted groups.
    """
    calibration_csv = output_dir / 'calibration.csv'
    calibration_df = pd.read_csv(calibration_csv) if calibration_csv.is_file() else pd.DataFrame()

    for ntypes, run in itertools.product(args.ntypes, range(args.runs)):
        if not calibration_df.empty and ((calibration_df['ntypes'] == ntypes) & (calibration_df['run_id'] == run)).any():
            continue
        print(f"[{DATASETS[dataset_key]['description']}] ntypes={ntypes}, run={run}")
        dr_exp = train_or_load(args, dataset_key, data_files, ntypes, run)
        groups = predicted_groups(dr_exp)
        row = {
            'ntypes': ntypes,
            'run_id': run,
            'p_reoptimization_period': DATASETS[dataset_key]['p_reoptimization_period'],
            **{f'exp_score_{split}': likelihood_score(dr_exp, split) for split in data_files},
            'group_sizes': str({g: len(annotators) for g, annotators in groups.items()}),
            'pred_groups': str(groups),
            'last_loss': dr_exp.history[-1],
            'best_loss': min(dr_exp.history),
        }
        print(f"    score (train)={row['exp_score_train']:.4f}, group sizes={row['group_sizes']}")
        calibration_df = pd.concat([calibration_df, pd.DataFrame([row])], ignore_index=True)
        calibration_df.to_csv(calibration_csv, index=False)
    return calibration_df


def print_calibration(calibration_df: pd.DataFrame, num_groups: int):
    stats = calibration_df.groupby('ntypes')['exp_score_train'].agg(['mean', 'median', 'std'])
    print("\nLikelihood score (train) by number of groups:")
    print(stats.to_string(float_format=lambda x: f"{x:.4f}"))
    if num_groups not in stats.index:
        return
    # As in the paper: Δ-l L = L(n) - L(n-l) and Δ+l L = L(n+l) - L(n)
    for offset in [-2, -1, 1, 2]:
        other = num_groups + offset
        if other in stats.index:
            larger, smaller = max(num_groups, other), min(num_groups, other)
            gains = {stat: stats.loc[larger, stat] - stats.loc[smaller, stat] for stat in ['mean', 'median']}
            print(f"Calibration gain ΔL{offset:+d} (n={larger} vs n={smaller}): {gains['mean']:.4f} (mean), {gains['median']:.4f} (median)")


def best_run(calibration_df: pd.DataFrame, num_groups: int) -> int:
    """
    Run with the highest likelihood score on the training split, among those with `num_groups` groups.
    """
    candidates = calibration_df[calibration_df['ntypes'] == num_groups]
    if candidates.empty:
        raise ValueError(f"No trained model with {num_groups} groups: include it in --ntypes.")
    return int(candidates.sort_values('exp_score_train', ascending=False).iloc[0]['run_id'])


# 3. MOST RELEVANT ITEMS (MD-Agreement) =============================================================

def agreement_level(annotations: dict) -> str:
    """
    Agreement level of an item: full (unanimous), low (closest possible split, e.g. 3-2), or medium (e.g. 4-1).
    """
    counts = sorted(pd.Series(list(annotations.values())).value_counts().tolist(), reverse=True)
    if len(counts) == 1:
        return 'full'
    if counts[0] - counts[1] <= 1:
        return 'low'
    return 'medium'


def most_relevant_items(dr_exp: FlockeExperiment, test_data: dict, output_dir: pathlib.Path):
    ntypes = dr_exp.model_config['ntypes']
    group_predictions_df = dr_exp.group_predictions('test')
    js_df = pairwise_js_divergences(group_predictions_df, ntypes)

    print("\nMedian JS divergence between group predictions (test set):")
    for a, b in itertools.combinations(range(ntypes), 2):
        print(f"    groups {a}-{b}: {js_df[f'js_div_g{a}_g{b}'].median():.3f}")

    # Rank items by average pairwise JS divergence
    for g in range(ntypes):
        js_df[f'g{g}_label'] = group_predictions_df['predicted_labels'].apply(lambda labels: labels[g])
    js_df['agreement'] = js_df['example_id'].apply(lambda example_id: agreement_level(test_data[example_id]['annotations']))
    ranking = js_df.sort_values('avg_js_div', ascending=False).reset_index(drop=True)
    ranking.head(TOP_RELEVANT_ITEMS).to_csv(output_dir / 'most_relevant_items.csv', index=False)
    print(f"Top {TOP_RELEVANT_ITEMS} most relevant items saved to {output_dir / 'most_relevant_items.csv'}")

    # Agreement levels along the ranking (thirds)
    composition = []
    for third, indices in enumerate(np.array_split(ranking.index, 3)):
        shares = ranking.loc[indices, 'agreement'].value_counts(normalize=True) * 100
        composition.append({'third': third + 1, **{level: shares.get(level, 0.0) for level in ['full', 'medium', 'low']}})
    composition_df = pd.DataFrame(composition)
    composition_df.to_csv(output_dir / 'most_relevant_items_agreement.csv', index=False)
    print("Agreement levels (% of items) in each third of the ranking:")
    print(composition_df.to_string(index=False, float_format=lambda x: f"{x:.1f}"))


# 4. SOFT-LABEL PREDICTION (MD-Agreement) =============================================================

def soft_label_prediction(dr_exp: FlockeExperiment, test_data: dict, output_dir: pathlib.Path):
    """
    Cross-entropy between the soft labels of test items and:
        - the label distribution predicted by each group;
        - the average of the predictions for the item's annotators, each a mixture of group predictions weighted by the
          annotator's group membership (learned on the training set, or the group prior for annotators unseen in training).
    """
    ntypes = dr_exp.model_config['ntypes']
    group_predictions_df = dr_exp.group_predictions('test')
    labels = sorted(dr_exp.label_mapping, key=dr_exp.label_mapping.get)

    seen_annotators = set(int(a) for a in np.unique(dr_exp.flocke_data['train']._u))
    z_prior = np.asarray(dr_exp.dr._z_priors, dtype=float)

    def membership(annotator: str) -> np.ndarray:
        annotator_id = dr_exp.annotator_mapping[annotator]
        if annotator_id in seen_annotators and annotator_id < len(dr_exp.train_pug):
            return np.asarray(dr_exp.train_pug[annotator_id], dtype=float)
        return z_prior

    targets, weighted_predictions = [], []
    test_annotators = set()
    for _, row in group_predictions_df.iterrows():
        item = test_data[row['example_id']]
        group_preds = np.asarray(row['predictions']) # Shape: (n_groups, n_classes)
        annotator_preds = [membership(annotator) @ group_preds for annotator in item['annotations']]
        test_annotators.update(item['annotations'].keys())
        targets.append([item['soft_label'].get(label, 0.0) for label in labels])
        weighted_predictions.append(np.mean(annotator_preds, axis=0))
    targets = np.asarray(targets)

    results = {f'ce_group_{g}': cross_entropy(targets, np.asarray([preds[g] for preds in group_predictions_df['predictions']])) for g in range(ntypes)}
    results['ce_annotator_weighted'] = cross_entropy(targets, np.asarray(weighted_predictions))
    n_seen = sum(dr_exp.annotator_mapping[a] in seen_annotators for a in test_annotators)
    results['test_annotators_seen_in_training'] = n_seen
    results['test_annotators_unseen'] = len(test_annotators) - n_seen
    pd.DataFrame([results]).to_csv(output_dir / 'soft_label_prediction.csv', index=False)

    print("\nSoft-label prediction on the test set (cross-entropy):")
    for g in range(ntypes):
        print(f"    group {g}: {results[f'ce_group_{g}']:.3f}")
    print(f"    annotator-weighted: {results['ce_annotator_weighted']:.3f} ({n_seen} test annotators seen in training, {results['test_annotators_unseen']} unseen)")


# 5. SOCIO-DEMOGRAPHICS (MultiPICo) =============================================================

def demographics_association(groups: dict, metadata: dict, output_dir: pathlib.Path):
    """
    Pearson chi-square test of independence between latent groups and each socio-demographic trait, and each pairwise
    intersection of traits. Sparse traits are coarsened (age bins, continents), and unknown values are discarded.
    """
    def annotator_metadata(annotator: str):
        for key in [annotator, annotator.replace('Ann', '', 1)]:
            if key in metadata:
                return metadata[key]
        return None

    rows, missing = [], 0
    for group, annotators in groups.items():
        for annotator in annotators:
            ann_metadata = annotator_metadata(annotator)
            if ann_metadata is None:
                missing += 1
                continue
            rows.append({'annotator': annotator, 'group': group, **{demo: demographic_cohort(ann_metadata.get(demo), demo) for demo in DEMOGRAPHICS}})
    if missing:
        print(f"Warning: no metadata for {missing} annotators.")
    wide_df = pd.DataFrame(rows)

    unmapped = sorted({str(m.get(demo)) for m in metadata.values() for demo in ["Country of birth", "Country of residence", "Nationality"]
                       if demographic_cohort(m.get(demo), "") != "Unknown" and str(m.get(demo)).strip() not in COUNTRY_TO_CONTINENT})
    if unmapped:
        print(f"Warning: countries without a continent (treated as Unknown), add them to COUNTRY_TO_CONTINENT: {unmapped}")

    def chi2_test(values: pd.Series, groups_col: pd.Series):
        contingency = pd.crosstab(groups_col, values)
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            return None
        chi2, p, dof, expected = chi2_contingency(contingency)
        return {'chi2': chi2, 'dof': dof, 'p_value': p, 'n': int(contingency.values.sum()), 'cells_expected_below_5': float((expected < 5).mean())}

    results = []
    for demo in DEMOGRAPHICS:
        sub = wide_df[wide_df[demo] != 'Unknown']
        test = chi2_test(sub[demo], sub['group'])
        if test is not None:
            results.append({'variables': demo, **test})
    for d1, d2 in itertools.combinations(DEMOGRAPHICS, 2):
        sub = wide_df[(wide_df[d1] != 'Unknown') & (wide_df[d2] != 'Unknown')]
        test = chi2_test(sub[d1] + ' & ' + sub[d2], sub['group'])
        if test is not None:
            results.append({'variables': f"{d1} x {d2}", **test})

    results_df = pd.DataFrame(results).sort_values('p_value')
    results_df.to_csv(output_dir / 'demographics_chi2.csv', index=False)
    print("\nAssociation between latent groups and socio-demographics (chi-square, sorted by p-value):")
    print(results_df.head(10).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


# MAIN =============================================================

def run_dataset(args, dataset_key: str):
    config = DATASETS[dataset_key]
    missing = [f for f in config['required_files'] if not (config['data_dir'] / f).is_file()]
    if missing:
        print(f"Skipping {config['description']}: missing {missing} in {config['data_dir']} (see data/README.md).")
        return

    print("\n" + "=" * 60)
    print(config['description'])
    print("=" * 60)
    output_dir = pathlib.Path(args.output_dir) / args.embedder / config['name']
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_key)
    data_files = {split: dataset.get_split(split) for split in ['train', 'dev', 'test'] if dataset.get_split(split)}
    print("Items per split: " + ", ".join(f"{split}={len(items)}" for split, items in data_files.items()))

    # 1. Group number detection
    calibration_df = groups_calibration(args, dataset_key, data_files, output_dir)
    print_calibration(calibration_df, args.num_groups)

    # 2. Annotator group distribution
    run = best_run(calibration_df, args.num_groups)
    dr_exp = train_or_load(args, dataset_key, data_files, args.num_groups, run)
    groups = predicted_groups(dr_exp)
    print(f"\nBest run with {args.num_groups} groups: run {run}. Annotators per group (train): " + ", ".join(f"group {g}: {len(a)}" for g, a in groups.items()))

    if dataset_key == 'md':
        if 'test' not in data_files:
            print("No annotated test split: skipping most relevant items and soft-label prediction.")
            return
        # 3. Most relevant items
        most_relevant_items(dr_exp, data_files['test'], output_dir)
        # 4. Soft-label prediction
        soft_label_prediction(dr_exp, data_files['test'], output_dir)
    else:
        # 5. Socio-demographics
        demographics_association(groups, dataset.load_annotator_metadata(), output_dir)


def main():
    args = get_args()
    for dataset_key in args.datasets:
        run_dataset(args, dataset_key)


if __name__ == "__main__":
    main()
