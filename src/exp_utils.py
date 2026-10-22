import os
import contextlib
import io
import functools
import json
import itertools
import random
import torch
import copy
import numpy as np
import pandas as pd
from pathlib import Path
import datasets

from exp_config import CONDITIONS, RULES, GROUP_TEMPLATES, HATECHECK_EXPERIMENTS


def load_json_data(json_path) -> dict:
    """
    Loads JSON data from a file path or returns the dictionary if already a dict.
    """
    if hasattr(json_path, 'to_json_dict'):
        return json_path.to_json_dict()
    if isinstance(json_path, dict):
        return json_path
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def suppress_if_not_verbose(func):
    """
    Decorator that suppresses stdout from a function if verbose=False.
    The function must have 'verbose' as a keyword argument.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        verbose = kwargs.get('verbose', False)
        if verbose:
            return func(*args, **kwargs)
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                return func(*args, **kwargs)
    return wrapper


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



# ===============================================
# SYNTHETIC HATECHECK
# ===============================================

def _build_annotator_groups(num_annotators, group_names, rng=random.Random(42), imbalanced=[0.5, 0.5]):
    groups = {group_name: [] for group_name in group_names}
    n_groups = len(group_names)
    if not imbalanced: # Split annotators as evenly as possible across groups
        base = num_annotators // n_groups
        remainder = num_annotators % n_groups
        next_ann = 1
        for index, group_name in enumerate(group_names):
            count = base + (1 if index < remainder else 0)
            groups[group_name] = [f'Ann{i}' for i in range(next_ann, next_ann + count)]
            next_ann += count
    else: # Create imbalanced groups based on provided ratios
        if len(imbalanced) != n_groups:
            raise ValueError(f'Length of imbalanced list must match number of groups. Expected {n_groups}, got {len(imbalanced)}')
        if sum(imbalanced) != 1.0:
            raise ValueError(f'Imbalanced ratios must sum to 1.0. Got {sum(imbalanced)}')
        annotators = [f'Ann{i}' for i in range(1, num_annotators + 1)]
        rng.shuffle(annotators)
        start_idx = 0
        for group_name, ratio in zip(group_names, imbalanced):
            count = int(num_annotators * ratio)
            groups[group_name] = annotators[start_idx:start_idx + count]
            start_idx += count
        if start_idx < num_annotators: # If any remaining annotators, assigned to the last group
            groups[group_names[-1]].extend(annotators[start_idx:])
    return groups


# def _get_group_base_label(version_config, group_name, row, original_label):
#     rule = version_config['group_rules'].get(group_name)
#     if rule and rule['is_relevant_item'](row):
#         return rule['biased_label']
#     return original_label


def evaluate_group_label(group_name, row, original_label, group_templates=GROUP_TEMPLATES):
    rule_names = group_templates[group_name]
    rules = [RULES[r] for r in rule_names]

    # sort by priority
    rules = sorted(rules, key=lambda r: r["priority"])

    for rule in rules:
        condition_fn = CONDITIONS[rule["condition"]]
        if condition_fn(row):
            return rule["label"]

    return original_label


def _apply_noise(label, random_noise_probability, rng):
    if rng.random() < (1 - random_noise_probability):
        return label, False
    flipped_label = 'non-hateful' if label == 'hateful' else 'hateful'
    return flipped_label, True


class SplitWrapper:
    """Wrapper to provide indexed access to items within a split."""
    def __init__(self, split_data):
        self.split_data = split_data
        self.case_ids = list(split_data.keys())
    
    def __len__(self):
        return len(self.case_ids)
    
    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        return case_id, self.split_data[case_id]
    
    def keys(self):
        return self.split_data.keys()
    
    def values(self):
        return self.split_data.values()
    
    def items(self):
        return self.split_data.items()
    
    def get(self, key, default=None):
        return self.split_data.get(key, default)


class HateCheckData(torch.utils.data.Dataset):
    def __init__(
        self,
        hatecheck_csv_path: str = '../../data/hatecheck/test.csv',
        parameters: dict | None = None,
        splits: list | None = None,
        split_ratios: dict | None = None,
        split_criterion: str = 'by_annotator',
        seed: int = 42,
        experiments: dict = HATECHECK_EXPERIMENTS,
        group_templates: dict = GROUP_TEMPLATES,
    ):  
        if parameters is None:
            parameters = {
                'version': 2001,
                'num_annotators': 10,
                'annotations_per_entry': 5,
                'random_noise_probability': 0.0,
            }
        if splits is None:
            splits = ['train']
        if split_ratios is None:
            split_ratios = {'train': 1.0}

        # Checks
        if parameters["version"] not in experiments:
            raise ValueError(f'Invalid bias version: {parameters["version"]}')
        total_ratio = sum(split_ratios.values())
        if total_ratio != 1.0:
            raise ValueError(f'Split ratios must sum to 1.0, but got {total_ratio}')
        for split in splits:
            if split not in split_ratios:
                raise ValueError(f'Split "{split}" is in splits but not defined in split_ratios')
        if split_criterion not in ["by_annotator", "by_item"]:
            raise ValueError(f'Invalid split criterion: {split_criterion}. Must be "by_annotator" or "by_item".')

        self.hatecheck_csv_path = Path(hatecheck_csv_path)
        self.parameters = copy.deepcopy(parameters)
        self.splits = copy.deepcopy(splits)
        self.split_ratios = copy.deepcopy(split_ratios)
        self.split_criterion = split_criterion
        self.version_config = experiments[self.parameters['version']]
        self.imbalanced_groups = self.version_config.get('imbalanced_groups', None)
        self.seed = seed
        self.group_templates = group_templates
        self.individual_bias = self.version_config.get('individual_bias', None)

        self.df = pd.read_csv(self.hatecheck_csv_path)
        if self.individual_bias is not None and 'functionality' not in self.df.columns:
            raise ValueError('individual_bias requires a "functionality" column in the HateCheck CSV')
        self.groups = _build_annotator_groups(self.parameters['num_annotators'], self.version_config['groups'], rng=random.Random(seed), imbalanced=self.imbalanced_groups)
        self.all_annotators = []
        self.annotator_to_group = {}
        for group_name, annotators in self.groups.items():
            self.all_annotators.extend(annotators)
            for annotator in annotators:
                self.annotator_to_group[annotator] = group_name

        self._functionality_values = None
        self._annotator_to_functionality = None
        if self.individual_bias is not None:
            self._functionality_values = sorted(self.df['functionality'].dropna().unique().tolist())
            if not self._functionality_values:
                raise ValueError('individual_bias requires non-empty functionality values in the HateCheck CSV')
            self._annotator_to_functionality = self._build_annotator_to_functionality()

        self._dataset_dict = None
        self._info_dict = None
        self._disagreement_df = None  # NOTE: These are the TRUE RELEVANT ITEMS (even if disagreement is not observed because of random sampling of annotators, or noise)
        self._case_ids = None
        self._annotator_to_split = None
        self._item_to_split = None

    def _build_annotator_to_functionality(self):
        rng = random.Random(self.seed)
        return {
            annotator: rng.choice(self._functionality_values)
            for annotator in self.all_annotators
        }

    def _build_annotator_to_split(self):
        """Build mapping of annotators to splits based on split_ratios."""
        rng = random.Random(self.seed)
        annotator_to_split = {}
        
        # Shuffle all annotators
        shuffled_annotators = rng.sample(self.all_annotators, len(self.all_annotators))
        
        # Assign to splits based on ratios
        start_idx = 0
        for split_name in self.splits:
            ratio = self.split_ratios[split_name]
            count = int(len(self.all_annotators) * ratio)
            end_idx = start_idx + count if split_name != self.splits[-1] else len(self.all_annotators)
            
            for annotator in shuffled_annotators[start_idx:end_idx]:
                annotator_to_split[annotator] = split_name
            start_idx = end_idx
        
        return annotator_to_split

    def _build_item_to_split(self):
        """Build mapping of items to splits based on split_ratios."""
        rng = random.Random(self.seed)
        item_to_split = {}
        
        # Get all case IDs from dataframe
        case_ids = []
        for _, row in self.df.iterrows():
            case_id_raw = row['case_id']
            try:
                case_id = int(case_id_raw)
            except (TypeError, ValueError):
                case_id = case_id_raw
            case_ids.append(case_id)
        
        # Shuffle case IDs
        shuffled_case_ids = rng.sample(case_ids, len(case_ids))
        
        # Assign to splits based on ratios
        start_idx = 0
        for split_name in self.splits:
            ratio = self.split_ratios[split_name]
            count = int(len(case_ids) * ratio)
            end_idx = start_idx + count if split_name != self.splits[-1] else len(case_ids)
            
            for case_id in shuffled_case_ids[start_idx:end_idx]:
                item_to_split[case_id] = split_name
            start_idx = end_idx
        
        return item_to_split

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Handle string indexing (split names)
        if isinstance(idx, str):
            if idx not in self.splits:
                raise KeyError(f'Split "{idx}" not found. Available splits: {self.splits}')
            dataset = self.to_json_dict()
            return SplitWrapper(dataset[idx])
        
        # Handle integer indexing (items across all splits)
        dataset = self.to_json_dict()
        # Get all case_ids from all splits
        all_case_ids = []
        for split_data in dataset.values():
            all_case_ids.extend(split_data.keys())
        
        if self._case_ids is None:
            self._case_ids = all_case_ids
        case_id = self._case_ids[idx]
        
        # Find which split this case_id belongs to
        for split_name, split_data in dataset.items():
            if case_id in split_data:
                return case_id, split_data[case_id]

    def _get_group_labels_for_row(self, row, original_label):
        labels_by_group = {}
        for group_name in self.version_config['groups']:
            # labels_by_group[group_name] = _get_group_base_label(
            #     version_config=self.version_config,
            #     group_name=group_name,
            #     row=row,
            #     original_label=original_label,
            # )
            labels_by_group[group_name] = evaluate_group_label(group_name, row, original_label, group_templates=self.group_templates)
        return labels_by_group

    def _apply_individual_bias(self, label, annotator, row, rng):
        if self.individual_bias is None:
            return label, False
        if rng.random() >= self.individual_bias:
            return label, False
        annotator_functionality = self._annotator_to_functionality.get(annotator)
        if annotator_functionality is None:
            return label, False
        if row['functionality'] != annotator_functionality:
            return label, False
        flipped_label = 'non-hateful' if label == 'hateful' else 'hateful'
        return flipped_label, True

    def _generate_annotations(self):
        rng = random.Random(self.seed)
        
        # Build split assignments
        if self.split_criterion == 'by_annotator':
            if self._annotator_to_split is None:
                self._annotator_to_split = self._build_annotator_to_split()
            split_assignments = self._annotator_to_split
        else:  # by_item
            if self._item_to_split is None:
                self._item_to_split = self._build_item_to_split()
            split_assignments = self._item_to_split
        
        synthetic_data = {split_name: {} for split_name in self.splits}
        disagreement_rows = []

        for _, row in self.df.iterrows():
            case_id_raw = row['case_id']
            try:
                case_id = int(case_id_raw)
            except (TypeError, ValueError):
                case_id = case_id_raw
            text = row['test_case']
            original_label = row['label_gold']
            labels_by_group = self._get_group_labels_for_row(row, original_label)

            if len(set(labels_by_group.values())) > 1:
                disagreeing_group_pairs = [
                    (group_a, group_b)
                    for group_a, group_b in itertools.combinations(self.version_config['groups'], 2)
                    if labels_by_group[group_a] != labels_by_group[group_b]
                ]
                disagreement_row = {
                    'id': case_id,
                    'text': text,
                    'disagreeing_group_pairs': disagreeing_group_pairs,
                }
                for group_name, label in labels_by_group.items():
                    disagreement_row[f'{group_name.lower()}_label'] = label
                disagreement_rows.append(disagreement_row)

            selected_annotators = rng.sample(self.all_annotators, self.parameters['annotations_per_entry'])
            
            if self.split_criterion == 'by_annotator':
                # Group annotations by split based on annotator assignment
                annotations_by_split = {split_name: {} for split_name in self.splits}
                for annotator in selected_annotators:
                    group_name = self.annotator_to_group[annotator]
                    # base_label = _get_group_base_label(self.version_config, group_name, row, original_label)
                    base_label = evaluate_group_label(group_name, row, original_label, group_templates=self.group_templates)
                    label, _ = self._apply_individual_bias(base_label, annotator, row, rng)
                    label, _ = _apply_noise(label, self.parameters['random_noise_probability'], rng)
                    assigned_split = self._annotator_to_split[annotator]
                    annotations_by_split[assigned_split][annotator] = '1' if label == 'hateful' else '0'
                
                # Create entries for each split that has annotations for this item
                for split_name, annotations in annotations_by_split.items():
                    if annotations:  # Only create entry if there are annotations
                        annotation_values = list(annotations.values())
                        count_0 = annotation_values.count('0')
                        count_1 = annotation_values.count('1')
                        
                        synthetic_data[split_name][case_id] = {
                            'id': case_id,
                            'annotation task': 'hate speech detection',
                            'text': text,
                            'number of annotators': len(annotations),
                            'annotators': ','.join(sorted(annotations.keys())),
                            'number of annotations': len(annotations),
                            'annotations': annotations,
                            'soft_label': {
                                '0': count_0 / len(annotations),
                                '1': count_1 / len(annotations),
                            },
                        }
            else:  # by_item
                # All annotations go to the same split
                assigned_split = self._item_to_split[case_id]
                annotations = {}
                for annotator in selected_annotators:
                    group_name = self.annotator_to_group[annotator]
                    # base_label = _get_group_base_label(self.version_config, group_name, row, original_label)
                    base_label = evaluate_group_label(group_name, row, original_label, group_templates=self.group_templates)
                    label, _ = self._apply_individual_bias(base_label, annotator, row, rng)
                    label, _ = _apply_noise(label, self.parameters['random_noise_probability'], rng)
                    annotations[annotator] = '1' if label == 'hateful' else '0'

                annotation_values = list(annotations.values())
                count_0 = annotation_values.count('0')
                count_1 = annotation_values.count('1')

                synthetic_data[assigned_split][case_id] = {
                    'id': case_id,
                    'annotation task': 'hate speech detection',
                    'text': text,
                    'number of annotators': self.parameters['annotations_per_entry'],
                    'annotators': ','.join(sorted(selected_annotators)),
                    'number of annotations': self.parameters['annotations_per_entry'],
                    'annotations': annotations,
                    'soft_label': {
                        '0': count_0 / self.parameters['annotations_per_entry'],
                        '1': count_1 / self.parameters['annotations_per_entry'],
                    },
                }

        self._case_ids = sorted(set(cid for split_data in synthetic_data.values() for cid in split_data.keys()))
        return synthetic_data, pd.DataFrame(disagreement_rows)

    def to_json_dict(self):
        if self._dataset_dict is None:
            self._dataset_dict, self._disagreement_df = self._generate_annotations()
        return self._dataset_dict

    def get_split(self, split_name):
        """Get data for a specific split.
        
        Args:
            split_name: Name of the split (e.g., 'train', 'test')
            
        Returns:
            Dictionary with case_id -> annotation data for the split
        """
        if split_name not in self.splits:
            raise ValueError(f'Split "{split_name}" not found. Available splits: {self.splits}')
        dataset = self.to_json_dict()
        return dataset[split_name]

    def to_info_dict(self):
        if self._info_dict is None:
            dataset = self.to_json_dict()
            split_stats = {}
            for split_name in self.splits:
                split_data = dataset[split_name]
                total_annotations = sum(
                    len(item.get('annotations', {})) for item in split_data.values()
                )
                split_stats[split_name] = {
                    'num_items': len(split_data),
                    'num_annotations': total_annotations,
                }
            
            self._info_dict = {
                'description': 'Synthetic HateCheck dataset with simulated annotator bias',
                'total_annotators': len(self.all_annotators),
                'annotations_per_entry': self.parameters['annotations_per_entry'],
                'random_noise_probability': self.parameters['random_noise_probability'],
                'total_entries': len(self.df),
                'splits': self.splits,
                'split_ratios': self.split_ratios,
                'split_criterion': self.split_criterion,
                'split_stats': split_stats,
                'groups': {
                    group_name: {
                        'annotators': self.groups[group_name],
                        'count': len(self.groups[group_name]),
                        # 'bias': self.version_config['group_rules'].get(group_name, {}).get('bias', 'No systematic bias'), # NOTE: No more bias description here (refer to config file)
                    }
                    for group_name in self.version_config['groups']
                },
                'label_mapping': {
                    '0': 'non-hateful',
                    '1': 'hateful',
                },
                'random_seed': self.seed,
            }
        return self._info_dict

    def save_json(self, output_json_path, info_json_path=None):
        output_json_path = Path(output_json_path)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        
        dataset = self.to_json_dict()
        
        # Save each split to a separate file or combine based on splits
        if len(self.splits) > 1:
            # Save each split separately
            for split_name, split_data in dataset.items():
                split_output_path = output_json_path.parent / f"{output_json_path.stem}_{split_name}{output_json_path.suffix}"
                with open(split_output_path, 'w') as f:
                    json.dump(split_data, f, indent=4)
        else:
            # Save single split as main file
            with open(output_json_path, 'w') as f:
                json.dump(dataset[self.splits[0]], f, indent=4)

        if info_json_path is not None:
            info_json_path = Path(info_json_path)
            info_json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(info_json_path, 'w') as f:
                json.dump(self.to_info_dict(), f, indent=4)

        return self._disagreement_df if self._disagreement_df is not None else pd.DataFrame()


class HSBrexitData(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str = '../../data/HsBrexit',
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.seed = seed
        self.splits = ['train', 'validation', 'test']

        self.data_dir.mkdir(parents=True, exist_ok=True)
        # Use HF cache in data_dir; if missing, it will download.
        self.dataset = datasets.load_dataset(
            "silvia-casola/BREXIT",
            cache_dir=str(self.data_dir),
        )

        self._dataset_dict = None

    def __len__(self):
        return sum(len(self.dataset[split]) for split in self.splits if split in self.dataset)

    def __getitem__(self, idx):
        # String indexing returns split wrapper
        if isinstance(idx, str):
            if idx not in self.splits:
                raise KeyError(f'Split "{idx}" not found. Available splits: {self.splits}')
            dataset = self.to_json_dict()
            return SplitWrapper(dataset[idx])

        # Integer indexing across all splits
        dataset = self.to_json_dict()
        all_case_ids = []
        for split_data in dataset.values():
            all_case_ids.extend(split_data.keys())
        case_id = all_case_ids[idx]
        for split_name, split_data in dataset.items():
            if case_id in split_data:
                return case_id, split_data[case_id]

    def _build_split_dict(self, split_name):
        split = self.dataset[split_name]
        df = split.to_pandas()

        grouped = df.groupby('instance_id', sort=False)
        split_dict = {}
        for instance_id, group in grouped:
            # Preserve original instance_id (already split-qualified like "train-1").
            instance_id = str(instance_id)
            text = group['tweet'].iloc[0]
            annotations = {
                str(annotator_id): '1' if int(hs) == 1 else '0'
                for annotator_id, hs in zip(group['annotator_id'], group['hs'])
            }
            annotation_values = list(annotations.values())
            count_0 = annotation_values.count('0')
            count_1 = annotation_values.count('1')
            split_dict[instance_id] = {
                'id': instance_id,
                'annotation task': 'hate speech detection',
                'text': text,
                'number of annotators': len(annotations),
                'annotators': ','.join(sorted(annotations.keys())),
                'number of annotations': len(annotations),
                'annotations': annotations,
                'soft_label': {
                    '0': count_0 / len(annotations),
                    '1': count_1 / len(annotations),
                },
            }
        return split_dict

    def to_json_dict(self):
        if self._dataset_dict is None:
            self._dataset_dict = {}
            for split_name in self.splits:
                if split_name in self.dataset:
                    self._dataset_dict[split_name] = self._build_split_dict(split_name)
        return self._dataset_dict

    def get_split(self, split_name):
        if split_name not in self.splits:
            raise ValueError(f'Split "{split_name}" not found. Available splits: {self.splits}')
        dataset = self.to_json_dict()
        return dataset.get(split_name, {})


class MDAgreementData(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str = '../../data/MD',
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.seed = seed
        self.splits = ['train', 'dev', 'test']

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.split_files = {
            'train': self.data_dir / 'MD_train.json',
            'dev': self.data_dir / 'MD_dev.json',
            'test': self.data_dir / 'MD_test.json',
        }

        self._dataset_dict = None

    def __len__(self):
        dataset = self.to_json_dict()
        return sum(len(dataset[split]) for split in self.splits if split in dataset)

    def __getitem__(self, idx):
        if isinstance(idx, str):
            if idx not in self.splits:
                raise KeyError(f'Split "{idx}" not found. Available splits: {self.splits}')
            dataset = self.to_json_dict()
            return SplitWrapper(dataset[idx])

        dataset = self.to_json_dict()
        all_case_ids = []
        for split_data in dataset.values():
            all_case_ids.extend(split_data.keys())
        case_id = all_case_ids[idx]
        for split_name, split_data in dataset.items():
            if case_id in split_data:
                return case_id, split_data[case_id]

    def _build_split_dict(self, split_name):
        split_file = self.split_files[split_name]
        raw_split = load_json_data(split_file)

        split_dict = {}
        for case_id, item in raw_split.items():
            case_id = str(case_id)
            text = item['text']
            annotations_raw = item.get('annotations', {})
            annotations = {
                str(annotator_id): str(label)
                for annotator_id, label in annotations_raw.items()
            }

            annotation_values = list(annotations.values())
            count_0 = annotation_values.count('0')
            count_1 = annotation_values.count('1')
            total_annotations = len(annotations)

            split_dict[case_id] = {
                'id': case_id,
                'annotation task': item.get('annotation task', 'offensiveness detection'),
                'text': text,
                'number of annotators': len(item.get('annotators', '').split(',')) if item.get('annotators') else total_annotations,
                'annotators': item.get('annotators', ''),
                'number of annotations': item.get('number of annotations', total_annotations),
                'annotations': annotations,
                'soft_label': item.get('soft_label', {
                    '0': count_0 / total_annotations if total_annotations else 0.0,
                    '1': count_1 / total_annotations if total_annotations else 0.0,
                }),
                'hard_label': str(item.get('hard_label', '1' if count_1 >= count_0 else '0')),
                'lang': item.get('lang', 'en'),
                'split': item.get('split', split_name),
                'other_info': item.get('other_info', {}),
            }
        return split_dict

    def to_json_dict(self):
        if self._dataset_dict is None:
            self._dataset_dict = {}
            for split_name in self.splits:
                split_file = self.split_files[split_name]
                if split_file.is_file():
                    self._dataset_dict[split_name] = self._build_split_dict(split_name)
        return self._dataset_dict

    def get_split(self, split_name):
        if split_name not in self.splits:
            raise ValueError(f'Split "{split_name}" not found. Available splits: {self.splits}')
        dataset = self.to_json_dict()
        return dataset.get(split_name, {})


class MultiPICoData(torch.utils.data.Dataset):
    """
    MultiPICo in the LeWiDi-2025 format (MP_{train,dev,test}.json), optionally restricted to one language (`lang`) and/or
    to a set of language varieties (`language_varieties`, matched against other_info['language_variety']).
    Items without annotations (e.g., unlabeled test items) are skipped.
    """
    def __init__(
        self,
        data_dir: str = 'data/multipico',
        lang: str | None = 'en',
        language_varieties: set[str] | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.lang = lang
        self.language_varieties = set(language_varieties) if language_varieties is not None else None
        self.splits = ['train', 'dev', 'test']
        self.split_files = {split: self.data_dir / f'MP_{split}.json' for split in self.splits}
        self.metadata_file = self.data_dir / 'MP_annotators_meta.json'
        self._dataset_dict = None

    def __len__(self):
        dataset = self.to_json_dict()
        return sum(len(split_data) for split_data in dataset.values())

    def __getitem__(self, idx):
        if isinstance(idx, str):
            if idx not in self.splits:
                raise KeyError(f'Split "{idx}" not found. Available splits: {self.splits}')
            return SplitWrapper(self.to_json_dict()[idx])
        dataset = self.to_json_dict()
        all_items = [(case_id, item) for split_data in dataset.values() for case_id, item in split_data.items()]
        return all_items[idx]

    @staticmethod
    def _item_lang(item):
        if 'lang' in item:
            return item['lang']
        other_info = item.get('other_info') or {}
        return other_info.get('lang', other_info.get('language'))

    def _build_split_dict(self, split_name):
        raw_split = load_json_data(self.split_files[split_name])
        split_dict = {}
        for case_id, item in raw_split.items():
            if self.lang is not None and self._item_lang(item) != self.lang:
                continue
            if self.language_varieties is not None and (item.get('other_info') or {}).get('language_variety') not in self.language_varieties:
                continue
            annotations_raw = item.get('annotations') or {}
            if not isinstance(annotations_raw, dict) or not annotations_raw:
                continue
            annotations = {str(annotator_id): str(label) for annotator_id, label in annotations_raw.items()}
            annotation_values = list(annotations.values())
            count_0 = annotation_values.count('0')
            count_1 = annotation_values.count('1')
            total_annotations = len(annotations)
            case_id = str(case_id)
            split_dict[case_id] = {
                'id': case_id,
                'annotation task': item.get('annotation task', 'irony detection'),
                'text': item['text'], # dict with post and reply, joined by FlockeExperiment._normalize_text
                'number of annotators': total_annotations,
                'annotators': ','.join(annotations.keys()),
                'number of annotations': total_annotations,
                'annotations': annotations,
                'soft_label': {
                    '0': count_0 / total_annotations,
                    '1': count_1 / total_annotations,
                },
                'lang': self._item_lang(item),
                'split': split_name,
            }
        return split_dict

    def to_json_dict(self):
        if self._dataset_dict is None:
            self._dataset_dict = {}
            for split_name in self.splits:
                if self.split_files[split_name].is_file():
                    split_dict = self._build_split_dict(split_name)
                    if split_dict:
                        self._dataset_dict[split_name] = split_dict
        return self._dataset_dict

    def get_split(self, split_name):
        if split_name not in self.splits:
            raise ValueError(f'Split "{split_name}" not found. Available splits: {self.splits}')
        return self.to_json_dict().get(split_name, {})

    def load_annotator_metadata(self) -> dict:
        """
        Socio-demographic metadata of the annotators: {annotator_id: {trait: value}}.
        """
        return load_json_data(self.metadata_file)


# ===============================================
# SOCIO-DEMOGRAPHICS (MultiPICo)
# ===============================================

COUNTRY_TO_CONTINENT = {
    # Europe
    "Albania": "Europe", "Austria": "Europe", "Belarus": "Europe", "Belgium": "Europe", "Bosnia and Herzegovina": "Europe",
    "Bulgaria": "Europe", "Croatia": "Europe", "Cyprus": "Europe", "Czech Republic": "Europe", "Czechia": "Europe",
    "Denmark": "Europe", "Estonia": "Europe", "Finland": "Europe", "France": "Europe", "Germany": "Europe", "Greece": "Europe",
    "Hungary": "Europe", "Iceland": "Europe", "Ireland": "Europe", "Italy": "Europe", "Kosovo": "Europe", "Latvia": "Europe",
    "Lithuania": "Europe", "Luxembourg": "Europe", "Malta": "Europe", "Moldova": "Europe", "Montenegro": "Europe",
    "Netherlands": "Europe", "North Macedonia": "Europe", "Norway": "Europe", "Poland": "Europe", "Portugal": "Europe",
    "Romania": "Europe", "Russia": "Europe", "Russian Federation": "Europe", "Serbia": "Europe", "Slovakia": "Europe",
    "Slovenia": "Europe", "Spain": "Europe", "Sweden": "Europe", "Switzerland": "Europe", "Ukraine": "Europe",
    "United Kingdom": "Europe", "UK": "Europe", "England": "Europe", "Scotland": "Europe", "Wales": "Europe", "Northern Ireland": "Europe",
    # Americas
    "Argentina": "Americas", "Bolivia": "Americas", "Brazil": "Americas", "Canada": "Americas", "Chile": "Americas",
    "Colombia": "Americas", "Costa Rica": "Americas", "Cuba": "Americas", "Dominican Republic": "Americas", "Ecuador": "Americas",
    "El Salvador": "Americas", "Guatemala": "Americas", "Haiti": "Americas", "Honduras": "Americas", "Jamaica": "Americas",
    "Mexico": "Americas", "Nicaragua": "Americas", "Panama": "Americas", "Paraguay": "Americas", "Peru": "Americas",
    "Puerto Rico": "Americas", "Trinidad and Tobago": "Americas", "United States": "Americas", "USA": "Americas",
    "United States of America": "Americas", "Uruguay": "Americas", "Venezuela": "Americas",
    # Asia
    "Afghanistan": "Asia", "Bangladesh": "Asia", "China": "Asia", "Hong Kong": "Asia", "India": "Asia", "Indonesia": "Asia",
    "Iran": "Asia", "Iraq": "Asia", "Israel": "Asia", "Japan": "Asia", "Jordan": "Asia", "Kazakhstan": "Asia", "Lebanon": "Asia",
    "Malaysia": "Asia", "Nepal": "Asia", "Pakistan": "Asia", "Philippines": "Asia", "Saudi Arabia": "Asia", "Singapore": "Asia",
    "South Korea": "Asia", "Korea": "Asia", "Sri Lanka": "Asia", "Syria": "Asia", "Taiwan": "Asia", "Thailand": "Asia",
    "Turkey": "Asia", "United Arab Emirates": "Asia", "Vietnam": "Asia", "Viet Nam": "Asia",
    # Africa
    "Algeria": "Africa", "Cameroon": "Africa", "Egypt": "Africa", "Ethiopia": "Africa", "Ghana": "Africa", "Kenya": "Africa",
    "Morocco": "Africa", "Mozambique": "Africa", "Nigeria": "Africa", "Rwanda": "Africa", "Senegal": "Africa",
    "Somalia": "Africa", "South Africa": "Africa", "Sudan": "Africa", "Tanzania": "Africa", "Tunisia": "Africa",
    "Uganda": "Africa", "Zambia": "Africa", "Zimbabwe": "Africa",
    # Oceania
    "Australia": "Oceania", "Fiji": "Oceania", "New Zealand": "Oceania",
}

UNKNOWN_DEMOGRAPHIC_VALUES = {"DATA_EXPIRED", "CONSENT_REVOKED", "Other", "Prefer not to say", ""}


def demographic_cohort(value, demographic: str) -> str:
    """
    Coarser cohorts for sparse socio-demographic traits: age bins, and continents instead of countries.
    Missing or uninformative values are mapped to "Unknown".
    """
    if value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() in UNKNOWN_DEMOGRAPHIC_VALUES:
        return "Unknown"
    if demographic == "Age":
        try:
            age = int(float(value))
        except (TypeError, ValueError):
            return "Unknown"
        if age < 25:
            return "18-24"
        if age < 35:
            return "25-34"
        if age < 45:
            return "35-44"
        if age < 55:
            return "45-54"
        return "55+"
    if demographic in ["Country of birth", "Country of residence", "Nationality"]:
        return COUNTRY_TO_CONTINENT.get(str(value).strip(), "Unknown")
    return str(value)


if __name__ == "__main__":
    print("Test case 1: Create dataset with default parameters")
    dataset = HateCheckData()
    print("Length of dataset:", len(dataset))
    print("Example item:", dataset[0])

    print("\nSanity check: individual bias flips when functionality matches")
    dataset_bias = HateCheckData(parameters={
        'version': "24001",
        'num_annotators': 10,
        'annotations_per_entry': 5,
        'random_noise_probability': 0.0,
    }, group_templates=load_json_data("exp_1/1_pt3_group_templates.json"), experiments=load_json_data("exp_1/1_pt9_exp_configs.json"))
    if dataset_bias.individual_bias is None:
        print("Skipping: individual_bias is not set in version_config")
    else:
        annotator = dataset_bias.all_annotators[0]
        annotator_group = dataset_bias.annotator_to_group[annotator]
        annotator_functionality = dataset_bias._annotator_to_functionality[annotator]
        matching_row = dataset_bias.df[
            dataset_bias.df['functionality'] == annotator_functionality
        ].iloc[0]
        base_label = evaluate_group_label(
            annotator_group,
            matching_row,
            matching_row['label_gold'],
            group_templates=dataset_bias.group_templates,
        )
        biased_label, _ = dataset_bias._apply_individual_bias(
            base_label,
            annotator,
            matching_row,
            rng=random.Random(0),
        )
        print("Annotator:", annotator)
        print("Functionality:", annotator_functionality)
        print("Case ID:", matching_row['case_id'])
        print("Text:", matching_row['test_case'])
        print("Base label:", base_label)
        print("Biased label:", biased_label)

    # print("\nTest case 2: Split by item")
    # splits = ['train', 'test']
    # split_ratios = {'train': 0.8, 'test': 0.2}
    # dataset = HateCheckData(splits=splits, split_ratios=split_ratios, split_criterion='by_item')
    # print("Length of dataset:", len(dataset))
    # print("\nExample item from train split:", dataset['train'][0])
    # print("\nExample item from test split:", dataset.get_split('test').get(next(iter(dataset.get_split('test')))))
    # print("\n\nInfo dict:", dataset.to_info_dict())

    # print("\nTest case 3: Split by annotator")
    # dataset = HateCheckData(splits=splits, split_ratios=split_ratios, split_criterion='by_annotator')
    # print("Length of dataset:", len(dataset))
    # print("\nExample item from train split:", dataset['train'][0])
    # print("\nExample item from test split:", dataset.get_split('test').get(next(iter(dataset.get_split('test')))))
    # print("\n\nInfo dict:", dataset.to_info_dict())

    # print("All annotators existing in train split:", all(ann in dataset._annotator_to_split and dataset._annotator_to_split[ann] == 'train' for ann in dataset.groups['GROUP_A']))
    # print("All annotators existing in test split:", all(ann in dataset._annotator_to_split and dataset._annotator_to_split[ann] == 'test' for ann in dataset.groups['GROUP_B']))
    # train_annotators = set(ann for ann, split in dataset._annotator_to_split.items() if split == 'train')
    # print("Train annotators:", train_annotators)
    # test_annotators = set(ann for ann, split in dataset._annotator_to_split.items() if split == 'test')
    # print("Test annotators:", test_annotators)
    # print("Overlap between train and test annotators:", train_annotators.intersection(test_annotators))

