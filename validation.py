import sys
import pathlib
import argparse
import itertools

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from exp_config import EMBEDDERS, VALIDATION_VARIANTS, build_group_templates, build_validation_experiments
from exp_utils import HateCheckData, seed_everything
from flocke_utils import FlockeExperiment, likelihood_score, hatecheck_grouping, group_annotation_f1, most_relevant_items_f1


SAVE_PATH_TEMPLATE = "hatecheck_v{version}_{num_annotators}x{annotations_per_entry}_n{random_noise_probability}_tr{train_ratio}_{split_criterion}_g{ntypes}_h{hidden[0]}x{hidden[1]}_pr{prior_tempering_beta}_pt{posterior_tempering_beta_train}_pi{posterior_tempering_beta_inference}_reop{p_reoptimization_period}_run{run}.pth"

HIDDEN = [8, 8]
NCLASSES = 2
RANDOM_NOISE_PROBABILITY = 0.0
TEST_RATIO = 0.2
TRAIN_PARAMS = {
    'max_epochs': 200,
    'lr': 2e-3,
    'batch_size': 64,
    'return_estimates': True,
    'prior_tempering_beta': 0.9,
    'posterior_tempering_beta_train': 1.0,
    'posterior_tempering_beta_inference': 1.0,
    'p_reoptimization_period': 100,
    'use_best_checkpoint': True,
    'normalize_embeddings': False,
    'pca_dim': None,
}


def get_args():
    parser = argparse.ArgumentParser(description="Validate FLOCKE on synthetic HateCheck annotations.")
    parser.add_argument('--variants', nargs='+', choices=VALIDATION_VARIANTS, default=VALIDATION_VARIANTS, help='Validation variants: balanced groups (Sec. 4), imbalanced groups (App. C.1), individual bias (App. C.2).')
    parser.add_argument('--embedder', choices=list(EMBEDDERS.keys()), default='mbert', help='Text embedder used as input features (App. C.3).')
    parser.add_argument('--runs', type=int, default=5, help='Number of runs (each with different synthetic groups).')
    parser.add_argument('--num-annotators', nargs='+', type=int, default=[10, 20, 50, 100], help='Total numbers of synthetic annotators.')
    parser.add_argument('--annotations-per-entry', nargs='+', type=int, default=[2, 5, 10], help='Numbers of annotations collected per item.')
    parser.add_argument('--max-epochs', type=int, default=TRAIN_PARAMS['max_epochs'], help='Training epochs.')
    parser.add_argument('--hatecheck-csv', type=str, default=str(ROOT / 'data' / 'hatecheck' / 'test.csv'), help='Path to the HateCheck CSV.')
    parser.add_argument('--output-dir', type=str, default=str(ROOT / 'results' / 'validation'), help='Directory for result CSVs.')
    parser.add_argument('--embeddings-dir', type=str, default=str(ROOT / 'calculated_embeddings'), help='Directory for cached embeddings.')
    parser.add_argument('--models-dir', type=str, default=str(ROOT / 'trained_models'), help='Directory for trained models.')
    parser.add_argument('--skip-test-split', action='store_true', help='Skip annotation prediction on held-out items.')
    parser.add_argument('--verbose', action='store_true', help='Print training progress.')
    return parser.parse_args()


# RESULTS TABLES =============================================================

class ResultsTable:
    """
    CSV-backed table of results, with one row per unique key (used to skip work that was already done).
    """
    def __init__(self, path: pathlib.Path):
        self.path = path
        self.df = pd.read_csv(path) if path.is_file() else pd.DataFrame()

    def has(self, key: str) -> bool:
        return 'key' in self.df.columns and (self.df['key'].astype(str) == key).any()

    def add(self, key: str, row: dict):
        self.df = pd.concat([self.df, pd.DataFrame([{'key': key, **row}])], ignore_index=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(self.path, index=False)


# EXPERIMENTS =============================================================

def assign_experiments(experiments: dict, runs: int, num_annotators_list: list, annotations_per_entry_list: list) -> list:
    """
    Cycle through the (interleaved) experiments, so that each run and each annotators configuration gets a different
    set of synthetic groups, and all numbers of ground-truth groups are equally represented.
    """
    experiment_ids = list(experiments.keys())
    assignments = []
    idx = 0
    for run in range(runs):
        for num_annotators, annotations_per_entry in itertools.product(num_annotators_list, annotations_per_entry_list):
            assignments.append((run, num_annotators, annotations_per_entry, experiment_ids[idx % len(experiment_ids)]))
            idx += 1
    return assignments


def make_dataset(args, version, num_annotators, annotations_per_entry, experiments, group_templates, test_split=False):
    parameters = {
        'version': version,
        'num_annotators': num_annotators,
        'annotations_per_entry': annotations_per_entry,
        'random_noise_probability': RANDOM_NOISE_PROBABILITY,
    }
    if test_split:
        return HateCheckData(hatecheck_csv_path=args.hatecheck_csv, parameters=parameters, experiments=experiments, group_templates=group_templates,
                             splits=['train', 'test'], split_criterion='by_item', split_ratios={'train': 1 - TEST_RATIO, 'test': TEST_RATIO})
    return HateCheckData(hatecheck_csv_path=args.hatecheck_csv, parameters=parameters, experiments=experiments, group_templates=group_templates)


def train_or_load(args, dataset, version, num_annotators, annotations_per_entry, ntypes, run, test_split=False) -> FlockeExperiment:
    """
    Prepare data and load the corresponding trained model, or train it (seeded by run) and save it.
    """
    train_params = {**TRAIN_PARAMS, 'max_epochs': args.max_epochs}
    data_files = {'train': dataset.get_split('train')}
    if test_split:
        data_files['test'] = dataset.get_split('test')

    dr_exp = FlockeExperiment(
        model_config={'ntypes': ntypes, 'nclasses': NCLASSES, 'hidden': HIDDEN},
        train_params=train_params,
        embedder_name=EMBEDDERS[args.embedder],
        data_files=data_files,
    )
    embeddings_path = pathlib.Path(args.embeddings_dir) / args.embedder / 'hatecheck.npz'
    dr_exp.prepare_data(embeddings_path=str(embeddings_path), from_npz=True, verbose=args.verbose)

    train_ratio = 1 - TEST_RATIO if test_split else 1.0
    model_path = pathlib.Path(args.models_dir) / args.embedder / 'hatecheck' / SAVE_PATH_TEMPLATE.format(
        version=version, num_annotators=num_annotators, annotations_per_entry=annotations_per_entry,
        random_noise_probability=int(RANDOM_NOISE_PROBABILITY * 100), train_ratio=round(train_ratio * 100), split_criterion='by_item',
        ntypes=ntypes, hidden=HIDDEN, prior_tempering_beta=int(train_params['prior_tempering_beta'] * 100),
        posterior_tempering_beta_train=int(train_params['posterior_tempering_beta_train'] * 100),
        posterior_tempering_beta_inference=int(train_params['posterior_tempering_beta_inference'] * 100),
        p_reoptimization_period=train_params['p_reoptimization_period'], run=run)

    seed_everything(run)
    loaded = dr_exp.load_or_train_model(filepath=str(model_path), verbose=args.verbose)
    if not loaded:
        dr_exp.save_model(str(model_path), verbose=args.verbose)
    return dr_exp


def run_variant(args, variant: str, output_dir: pathlib.Path):
    group_templates = build_group_templates()
    experiments = build_validation_experiments(variant, group_templates=group_templates)
    assignments = assign_experiments(experiments, args.runs, args.num_annotators, args.annotations_per_entry)

    calibration = ResultsTable(output_dir / 'calibration.csv')
    analyses = ResultsTable(output_dir / 'analyses.csv')
    test_prediction = ResultsTable(output_dir / 'annotation_prediction_test.csv')

    for i, (run, num_annotators, annotations_per_entry, version) in enumerate(assignments):
        gt_groups = len(experiments[version]['groups'])
        config = {'v': version, 'num_annotators': num_annotators, 'annotations_per_entry': annotations_per_entry, 'run_id': run, 'gt_groups': gt_groups}
        print(f"[{variant} {i + 1}/{len(assignments)}] experiment v{version} ({gt_groups} groups), {num_annotators} annotators x {annotations_per_entry} annotations/item, run {run}")

        # 1-2-3-4. FULL DATA: train with n_gt-2 ... n_gt+2 groups
        dataset = None
        for ntypes in range(max(1, gt_groups - 2), gt_groups + 3):
            key = f"{version}|{num_annotators}|{annotations_per_entry}|{run}|{ntypes}"
            needs_analyses = ntypes == gt_groups and not analyses.has(key)
            if calibration.has(key) and not needs_analyses:
                continue
            if dataset is None:
                dataset = make_dataset(args, version, num_annotators, annotations_per_entry, experiments, group_templates)
            dr_exp = train_or_load(args, dataset, version, num_annotators, annotations_per_entry, ntypes, run)

            if not calibration.has(key):
                exp_score = likelihood_score(dr_exp, 'train')
                calibration.add(key, {**config, 'ntypes': ntypes, 'exp_score': exp_score, 'last_loss': dr_exp.history[-1], 'best_loss': min(dr_exp.history)})
                print(f"    ntypes={ntypes}: score={exp_score:.4f}")

            if needs_analyses:
                grouping = hatecheck_grouping(dr_exp, dataset)
                train_f1 = group_annotation_f1(dr_exp, dataset, grouping['gt_to_pred_mapping'], split='train')
                mri = most_relevant_items_f1(dr_exp, dataset, grouping['pred_to_gt_mapping'])
                analyses.add(key, {
                    **config,
                    'grouping_accuracy': grouping['accuracy'],
                    'grouping_purity': grouping['purity'],
                    'grouping_completeness': grouping['completeness'],
                    'annotation_prediction_f1': train_f1,
                    'mri_f1': mri['f1'],
                    'mri_f1_by_group_pair': str(mri['f1_by_pair']),
                    'gt_groups_assignment': str(grouping['gt_groups']),
                    'pred_groups_assignment': str(grouping['pred_groups']),
                })
                print(f"    grouping accuracy={grouping['accuracy']:.3f}, annotation F1 (train)={train_f1:.3f}, most relevant items F1={mri['f1']:.3f}")

        # 3. HELD-OUT ITEMS: train on 80% of the items with n = n_gt, predict group labels on the remaining 20%
        if variant == 'balanced' and not args.skip_test_split:
            key = f"{version}|{num_annotators}|{annotations_per_entry}|{run}|{gt_groups}"
            if not test_prediction.has(key):
                dataset_split = make_dataset(args, version, num_annotators, annotations_per_entry, experiments, group_templates, test_split=True)
                dr_exp = train_or_load(args, dataset_split, version, num_annotators, annotations_per_entry, gt_groups, run, test_split=True)
                grouping = hatecheck_grouping(dr_exp, dataset_split)
                test_f1 = group_annotation_f1(dr_exp, dataset_split, grouping['gt_to_pred_mapping'], split='test')
                test_prediction.add(key, {**config, 'ntypes': gt_groups, 'annotation_prediction_f1': test_f1})
                print(f"    annotation F1 (held-out)={test_f1:.3f}")

    return summarize(variant, output_dir, calibration.df, analyses.df, test_prediction.df)


# SUMMARY =============================================================

def calibration_gains(calibration_df: pd.DataFrame) -> dict:
    """
    Calibration gain: difference between the likelihood score with the ground-truth number of groups and with n_gt - l groups,
    averaged over experiments, configurations and runs.
    """
    gains = {}
    config_cols = ['v', 'num_annotators', 'annotations_per_entry', 'run_id']
    scores = calibration_df.pivot_table(index=config_cols + ['gt_groups'], columns='ntypes', values='exp_score').reset_index()
    for offset in [-1, -2, 1, 2]:
        diffs = []
        for _, row in scores.iterrows():
            gt = int(row['gt_groups'])
            other = gt + offset
            if gt in scores.columns and other in scores.columns and not np.isnan(row[gt]) and not np.isnan(row[other]):
                diffs.append(row[gt] - row[other])
        gains[f'calibration_gain_{offset:+d}'] = float(np.mean(diffs)) if diffs else float('nan')
    return gains


def summarize(variant: str, output_dir: pathlib.Path, calibration_df, analyses_df, test_df) -> dict:
    summary = {'variant': variant}
    if not calibration_df.empty:
        summary.update(calibration_gains(calibration_df))
        relative = calibration_df.assign(relative_num_groups=calibration_df['ntypes'] - calibration_df['gt_groups'])
        relative.groupby('relative_num_groups')['exp_score'].describe().to_csv(output_dir / 'calibration_by_relative_num_groups.csv')
    for col in ['grouping_accuracy', 'grouping_purity', 'grouping_completeness', 'annotation_prediction_f1', 'mri_f1']:
        if col in analyses_df.columns:
            summary[f'{col}_mean'] = analyses_df[col].mean()
            summary[f'{col}_std'] = analyses_df[col].std()
    if 'annotation_prediction_f1' in test_df.columns:
        summary['annotation_prediction_f1_test_mean'] = test_df['annotation_prediction_f1'].mean()
        summary['annotation_prediction_f1_test_std'] = test_df['annotation_prediction_f1'].std()
    if not analyses_df.empty:
        # Detailed results by number of annotators and annotations per item (heatmaps in the paper)
        detail_cols = [c for c in ['grouping_accuracy', 'grouping_purity', 'grouping_completeness', 'annotation_prediction_f1', 'mri_f1'] if c in analyses_df.columns]
        analyses_df.groupby(['annotations_per_entry', 'num_annotators'])[detail_cols].mean().to_csv(output_dir / 'analyses_by_configuration.csv')
    pd.DataFrame([summary]).to_csv(output_dir / 'summary.csv', index=False)
    return summary


def main():
    args = get_args()
    if not pathlib.Path(args.hatecheck_csv).is_file():
        sys.exit(f"HateCheck not found at {args.hatecheck_csv}. Download it with: python data/download_data.py")

    summaries = []
    for variant in args.variants:
        output_dir = pathlib.Path(args.output_dir) / args.embedder / variant
        output_dir.mkdir(parents=True, exist_ok=True)
        summaries.append(run_variant(args, variant, output_dir))

    summary_df = pd.DataFrame(summaries).set_index('variant').T
    print("\n" + "=" * 60)
    print(f"VALIDATION SUMMARY ({args.embedder})")
    print("=" * 60)
    print(summary_df.to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
