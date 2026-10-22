import pathlib
import itertools
import copy
import numpy as np
import pandas as pd
import torch
import transformers
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModel
from IPython import get_ipython
from collections import defaultdict
from scipy.spatial.distance import jensenshannon
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score

from exp_utils import load_json_data, suppress_if_not_verbose
from metrics import clustering_accuracy, clustering_performance
from flocke import MLPClassifier, Flocke, FlockeData
from flocke.engine import Callback


class BestCheckpoint(Callback):
    """
    Keep the model weights of the epoch with the lowest average training loss, and restore them at the end of training
    (before Flocke computes the final annotator grouping and group priors).
    """
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.best_loss = float('inf')
        self.best_epoch = None
        self.best_state = None

    def on_epoch_end(self, flocke, epoch, loss, **kwargs):
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_epoch = epoch
            # Copy the tensors: state_dict() returns references to the live parameters
            self.best_state = {k: v.detach().clone() for k, v in flocke.mdl.state_dict().items()}

    def on_train_end(self, flocke, **kwargs):
        if self.best_state is not None:
            flocke.mdl.load_state_dict(self.best_state)
            if self.verbose:
                print(f"Loaded best checkpoint from epoch {self.best_epoch} with loss {self.best_loss}")


class FlockeExperiment():
    """
    Class to manage Flocke experiments.
    """

    _tokenizer_cache = {}
    _embedder_cache = {}
    _device_reported = False

    def __init__(self,
                 model_config: dict | None = None,
                 train_params: dict | None = None,
                 embedder_name: str = 'google-bert/bert-base-multilingual-cased',
                 data_files: dict | None = None,
                ):
        """
        Initialize FlockeExperiment.
        """
        # Suppress warnings when loading transformer checkpoints
        transformers.logging.set_verbosity_error()

        if model_config is None:
            model_config = {'ntypes': 2, 'nclasses': 2, 'hidden': [8, 8]}
        if train_params is None:
            train_params = {'max_epochs': 200, 'lr': 2e-3, 'batch_size': 64, 'return_estimates': True, 'prior_tempering_beta': 0.9, 'posterior_tempering_beta_train': 1.0, 'posterior_tempering_beta_inference': 1.0, 'p_reoptimization_period': 100, 'use_best_checkpoint': True, 'normalize_embeddings': False, 'pca_dim': None}
        self.model_config = copy.deepcopy(model_config)
        self.train_params = copy.deepcopy(train_params)
        self.dr = None # Flocke model
        self.train_pug = None # NOTE: Previously named 'amap'. Annotator grouping computed on training data
        self.history = None # Training history
        self.flocke_data = {} # Dict of FlockeData objects: {'train': FlockeData, 'dev': FlockeData, 'test': FlockeData}
        self.metadata = {} # Dict of metadata for each split: {'train': dict, 'dev': dict, 'test': dict}
        self.annotator_mapping = None
        self.label_mapping = None
        self.normalize_embeddings = train_params.get('normalize_embeddings', False)
        self.pca_dim = train_params.get('pca_dim', None)
        self.pca = None

        # Check data_files keys are allowed
        allowed_splits = {'train', 'dev', 'test'}
        if data_files:
            for split in data_files.keys():
                if split not in allowed_splits:
                    raise ValueError(f"Invalid split '{split}' in data_files. Allowed splits: {allowed_splits}")
        self.data_files = dict(data_files) if data_files else {}

        if embedder_name in self._tokenizer_cache:
            self.tokenizer = self._tokenizer_cache[embedder_name]
        else:
            # For sentence-transformers style models we don't need the AutoTokenizer
            # but keep a tokenizer only for non-sentence-transformers backends.
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(embedder_name)
            except Exception:
                self.tokenizer = None
            self._tokenizer_cache[embedder_name] = self.tokenizer

        # Prefer using SentenceTransformer when available for models from the
        # sentence-transformers family (they expose a higher-level encode API).
        self._is_sentence_transformer = False
        if str(embedder_name).startswith("sentence-transformers/"):
            if embedder_name in self._embedder_cache:
                self.embedder = self._embedder_cache[embedder_name]
            else:
                self.embedder = SentenceTransformer(embedder_name)
                self.embedder.eval()
                self._embedder_cache[embedder_name] = self.embedder
            self._is_sentence_transformer = True
        else:
            if embedder_name in self._embedder_cache:
                self.embedder = self._embedder_cache[embedder_name]
            else:
                self.embedder = AutoModel.from_pretrained(embedder_name).eval()
                self._embedder_cache[embedder_name] = self.embedder
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not FlockeExperiment._device_reported: # Report once per process, not for every experiment
            print(f"Using device: {self.device}")
            FlockeExperiment._device_reported = True
        # Move embedder to correct device. SentenceTransformer.to accepts a string or torch.device.
        try:
            # Some embedder objects accept string device names, convert accordingly
            self.embedder.to(str(self.device))
        except Exception:
            try:
                self.embedder.to(self.device)
            except Exception:
                # If embedder has no .to method, ignore
                pass

        # Set model input dimension based on embedding size of the embedder.
        # For SentenceTransformer use its helper; for AutoModel use config.hidden_size.
        if self._is_sentence_transformer:
            try:
                self.model_config['ninputs'] = self.embedder.get_sentence_embedding_dimension()
            except Exception:
                # Fallback: compute a single embedding to infer dimension (cheap)
                sample_emb = self.embedder.encode(["."], convert_to_numpy=True)
                self.model_config['ninputs'] = int(np.asarray(sample_emb).shape[1])
        else:
            # Some AutoModel configs expose hidden_size
            try:
                self.model_config['ninputs'] = self.embedder.config.hidden_size
            except Exception:
                # Fallback: attempt a one-shot forward pass to infer hidden size.
                if self.tokenizer is None:
                    raise RuntimeError("Cannot infer embedding dimension: tokenizer missing for AutoModel embedder")
                sample = self.tokenizer(["."], return_tensors='pt', padding=True)
                sample = {k: v.to(self.device) for k, v in sample.items()}
                with torch.no_grad():
                    out = self.embedder(**sample)
                try:
                    self.model_config['ninputs'] = out.last_hidden_state.shape[-1]
                except Exception:
                    raise RuntimeError("Unable to determine embedding dimension from AutoModel outputs")
    
    def is_running_in_notebook(self):
        try:
            shell = get_ipython().__class__.__name__
            return shell == "ZMQInteractiveShell"
        except Exception:
            return False


    def __str__(self):
        info = "+----------------------------+\n"
        info += "| Flocke Experiment\n"
        info += f"| Model config: {self.model_config}\n"
        info += f"| Training params: {self.train_params}\n"
        info += f"| Embedding transforms: normalize={self.normalize_embeddings}, pca_dim={self.pca_dim}\n"
        if isinstance(self.data_files[next(iter(self.data_files))], dict):
            # print only keys
            info += f"| Data files: {{split: {list(self.data_files.keys())}}}\n"
        else:
            info += f"| Data files: {self.data_files}\n"
        info += f"| Data prepared: {'Yes' if self.flocke_data else 'No'}\n"
        if self.flocke_data:
            for split in self.flocke_data:
                info += f"| - {split} data: {self.flocke_data[split]._x.shape[0]} samples, {self.flocke_data[split]._x.shape[1]} features (hidden size)\n"
        info += f"| Flocke model trained: {'Yes' if self.dr is not None else 'No'}\n"
        if self.dr is not None:
            info += f"| - Annotator grouping shape: {self.train_pug.shape if self.train_pug is not None else 'N/A'}\n"
            info += f"| - Training epochs: {len(self.history) if self.history is not None else 'N/A'}\n"
        info += "+----------------------------+"
        return info

    def _plot_history(self, label=None, figsize=(12, 4)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.history)
        ax.set_xlabel('epoch')
        ax.set_ylabel('average loss')
        if label is not None:
            ax.set_title(label)
        ax.grid()
        return fig

    def _handle_figure(self, fig, filename=None):
        if self.is_running_in_notebook():
            plt.show()
        elif filename:
            fig.savefig(filename, dpi=150, bbox_inches="tight")
            plt.close(fig)

    def print_trained_Flocke(self):

        # VISUALIZE GROUPS
        k = self.model_config['ntypes']
        cols = self.train_pug[:, :k]
        y = cols.sum(axis=0)
        x = np.arange(1, k + 1)
        fig = plt.figure()
        plt.bar(x, y)
        plt.xlabel('Column')
        plt.ylabel('Sum across rows')
        plt.title(f'Annotator group membership sums (ntypes={k})')
        plt.xticks(x)
        plt.tight_layout()
        self._handle_figure(fig)

        # PLOT HISTORY
        fig2 = self._plot_history()
        self._handle_figure(fig2)
        print(f'Z priors: {self.dr.z_priors}')
        print(f'Final loss value: {self.history[-1]:.4f}')
        print(f'Avg. loss in the last 50 epochs: {np.mean(self.history[-50:]):.4f}')


    def embed_texts(self, texts: list[str], batch_size: int = 32, max_length: int = 256, preprocess_fn=None) -> np.ndarray:
        # Use SentenceTransformer encode API if available
        if self._is_sentence_transformer:
            all_texts = texts
            if preprocess_fn:
                all_texts = [preprocess_fn(t) for t in texts]
            # SentenceTransformer.encode handles batching internally
            try:
                embs = self.embedder.encode(all_texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False, device=str(self.device))
            except TypeError:
                # Older SentenceTransformer versions may not accept device in encode
                embs = self.embedder.encode(all_texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
            return np.asarray(embs).astype(np.float32)

        # Default AutoModel-based embedding: use [CLS] token
        try:
            hidden_size = self.embedder.config.hidden_size
        except Exception:
            # Fall back to inferring dim by running a single forward pass
            if self.tokenizer is None:
                raise RuntimeError("Cannot infer embedding dimension: tokenizer missing for AutoModel embedder")
            sample = self.tokenizer(["."], return_tensors='pt', padding=True)
            sample = {k: v.to(self.device) for k, v in sample.items()}
            with torch.no_grad():
                out = self.embedder(**sample)
            hidden_size = out.last_hidden_state.shape[-1]

        embeddings = np.empty((len(texts), hidden_size))
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                if preprocess_fn:
                    batch_texts = [preprocess_fn(t) for t in batch_texts]
                encodings = self.tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
                encodings = {k: v.to(self.device) for k, v in encodings.items()}
                outputs = self.embedder(**encodings)
                batch_embeddings = outputs.last_hidden_state[:, 0, :]
                embeddings[i:i+len(batch_texts)] = batch_embeddings.cpu().numpy()#.astype(np.float32)
        return embeddings.astype(np.float32)
    
    def _get_all_annotators(self):
        all_annotators = set()
        for split, json_path in self.data_files.items():
            data = load_json_data(json_path)
            for example_data in data.values():
                if 'annotations' in example_data:
                    all_annotators.update(example_data['annotations'].keys())
        return sorted(all_annotators)
    
    def get_annotator_mapping(self):
        return {annotator: idx for idx, annotator in enumerate(self._get_all_annotators())}
    
    def _get_all_labels(self):
        all_labels = set()
        for split, json_path in self.data_files.items():
            data = load_json_data(json_path)
            for example_data in data.values():
                if 'annotations' in example_data:
                    for label in example_data['annotations'].values():
                        all_labels.add(label)
        return sorted(all_labels)
    
    def get_label_mapping(self):
        return {label: idx for idx, label in enumerate(self._get_all_labels())}

    def _normalize_text(self, t):
        if isinstance(t, str):
            return t
        if isinstance(t, dict):
            if "context" in t and "statement" in t:
                return f"Context: {t['context']} Statement: {t['statement']}"
            return " ".join(str(v) for v in t.values())
        if t is None:
            return ""
        return str(t)

    def _is_int_like(self, value) -> bool:
        if isinstance(value, (int, np.integer)):
            return True
        if isinstance(value, str):
            v = value.strip()
            if not v:
                return False
            if v[0] in {"+", "-"}:
                return v[1:].isdigit()
            return v.isdigit()
        return False

    def _split_to_int(self, split):
        if isinstance(split, (int, np.integer)) or (isinstance(split, str) and str(split).isdigit()):
            return int(split)
        split_to_int = {'train': 1, 'dev': 2, 'test': 3, 'val': 4, 'eval': 5}
        if split not in split_to_int:
            raise ValueError(f"Cannot convert split '{split}' to int for use_split_in_id.")
        return split_to_int[split]

    def _embedding_lookup_id(self, example_id, split=None, use_split_in_id: bool = False) -> str:
        if use_split_in_id:
            split_int = self._split_to_int(split)
            if not self._is_int_like(example_id):
                raise ValueError(f"Example id '{example_id}' cannot be converted to int for use_split_in_id.")
            example_id_int = int(str(example_id).strip())
            # Keep backward-compatible split+id concatenation format.
            return str(int(f"{split_int}{example_id_int}"))
        return str(example_id)

    def _id_sort_key(self, value):
        sval = str(value)
        if self._is_int_like(sval):
            return (0, int(sval))
        return (1, sval)

    def _fit_embedding_transform(self, embeddings: np.ndarray) -> np.ndarray:
        transformed = embeddings.astype(np.float32)
        if self.normalize_embeddings:
            transformed = normalize(transformed)
        if self.pca_dim is not None:
            n_components = min(self.pca_dim, transformed.shape[0], transformed.shape[1])
            if n_components < 1:
                raise ValueError("pca_dim must be at least 1 after accounting for the available samples and dimensions.")
            self.pca = PCA(n_components=n_components)
            transformed = self.pca.fit_transform(transformed)
            self.model_config['ninputs'] = self.pca.n_components_
        else:
            self.pca = None
        return transformed.astype(np.float32)

    def _transform_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        transformed = embeddings.astype(np.float32)
        if self.normalize_embeddings:
            transformed = normalize(transformed)
        if self.pca_dim is not None:
            if self.pca is None:
                raise RuntimeError("PCA has not been fitted yet. Call the fitting path before transforming embeddings.")
            transformed = self.pca.transform(transformed)
        return transformed.astype(np.float32)
    
    @suppress_if_not_verbose
    def prepare_data(self, embeddings_path:str = None, overwrite_embeddings: bool = False, from_npz: bool=False, use_split_in_id: bool = False, verbose: bool = False):
        """
        Load data, compute embeddings, and prepare FlockeData object with metadata.
        If `from_npz` is provided, embeddings are extracted from a single .npz file that contains `ids` and `embeddings` arrays (saved by id).
        """
    
        self.annotator_mapping = self.get_annotator_mapping()
        self.label_mapping = self.get_label_mapping()

        # If using a single .npz with all embeddings, compute/load it once and build a map
        npz_id_to_index = None
        npz_embeddings = None
        if from_npz:
            # Use compute_embeddings_with_ids which will load existing .npz or compute and save it
            emb_data = self.compute_embeddings_with_ids(embeddings_path, overwrite=overwrite_embeddings, use_split_in_id=use_split_in_id, verbose=verbose)
            npz_ids = emb_data['ids']
            npz_embeddings = emb_data['embeddings']
            npz_id_to_index = {str(_id): idx for idx, _id in enumerate(npz_ids)}

        split_payloads = {}
        split_order = list(self.data_files.keys())
        for split, json_path in self.data_files.items():
            data = load_json_data(json_path)
            # prepend split to filename for per-split npy files
            embeddings_splitpath = embeddings_path.replace('.npy', f'_{split}.npy') if embeddings_path else None

            # 1. EMBEDDINGS (before expanding examples for annotator)
            if from_npz:
                # Extract embeddings for this split by matching example ids to the .npz
                embeddings_list = []
                for example_id in data.keys():
                    iid = self._embedding_lookup_id(example_id, split=split, use_split_in_id=use_split_in_id)
                    if iid not in npz_id_to_index:
                        raise ValueError(f"Example id {example_id} not found in {embeddings_path}")
                    embeddings_list.append(npz_embeddings[npz_id_to_index[iid]])
                embeddings = np.vstack(embeddings_list).astype(np.float32)
            else:
                # Legacy behavior: per-split .npy or compute on the fly
                if embeddings_splitpath is None:
                    raise ValueError("embeddings_path must be provided to compute or load embeddings when from_npz is not used.")
                if pathlib.Path(embeddings_splitpath).is_file() and not overwrite_embeddings:
                    print(f"Loading embeddings from {embeddings_splitpath}...")
                    embeddings = np.load(embeddings_splitpath)
                else:
                    print(f"Computing embeddings for {split} data...")
                    texts = [self._normalize_text(example_data["text"]) for example_data in data.values()]
                    embeddings = self.embed_texts(texts)
                    if embeddings_splitpath is not None:
                        pathlib.Path(embeddings_splitpath).parent.mkdir(parents=True, exist_ok=True)
                        np.save(embeddings_splitpath, embeddings)
                        print(f"Saved embeddings to {embeddings_splitpath}.")

            split_payloads[split] = {
                'data': data,
                'embeddings': embeddings,
            }

        if self.pca_dim is not None and self.pca is None:
            fit_split = 'train' if 'train' in split_payloads else split_order[0]
            self._fit_embedding_transform(split_payloads[fit_split]['embeddings'])

        for split in split_order:
            data = split_payloads[split]['data']
            embeddings = split_payloads[split]['embeddings']
            embeddings = self._transform_embeddings(embeddings) if (self.normalize_embeddings or self.pca_dim is not None) else embeddings

            # 1.1 CREATE UNIQUE EXAMPLES DATAFRAME (before duplication for annotators)
            unique_example_ids = []
            unique_texts = []
            unique_soft_labels = []
            
            for example_id, example_data in data.items():
                unique_example_ids.append(example_id)
                unique_texts.append(example_data['text'])
                unique_soft_labels.append(example_data['soft_label'])
            
            unique_examples_df = pd.DataFrame({
                'example_id': unique_example_ids,
                'text': unique_texts,
                'soft_label': unique_soft_labels
            })

            # 2. PROCESS DATA
            example_ids = []
            annotator_ids = []
            texts = []
            labels = []
            soft_labels = []
            n_annotations = []
            per_annotator_embeddings = []

            for i, (example_id, example_data) in enumerate(data.items()):
                text = example_data['text']
                annotations = example_data['annotations']
                annotators = list(annotations.keys())
                soft_label = example_data['soft_label']
                n_annotation = example_data['number of annotations']
                embedding = embeddings[i]
            
                for annotator in annotators:
                    example_ids.append(example_id)
                    annotator_ids.append(annotator)
                    texts.append(text)
                    label = annotations[annotator]
                    labels.append(label)
                    soft_labels.append(soft_label)
                    n_annotations.append(n_annotation)
                    per_annotator_embeddings.append(embedding)
            x = np.array(per_annotator_embeddings).astype(np.float32)

            # 2.1 Annotator mapping
            numeric_annotator_ids = [self.annotator_mapping[ann] for ann in annotator_ids]
            u = np.array(numeric_annotator_ids)

            # 2.2 Label and soft label mapping
            numeric_labels = [self.label_mapping[label] for label in labels]
            # remapped_soft_labels = []
            # for soft_label in soft_labels:
            #     remapped_dict = {}
            #     for label_key, value in soft_label.items():
            #         # Cast key to int (in case it's float string)
            #         int_key = int(float(label_key))
            #         if int_key in self.label_mapping:
            #             new_key = self.label_mapping[int_key]
            #             remapped_dict[new_key] = value
            #     remapped_soft_labels.append(remapped_dict)
            y = np.array(numeric_labels, dtype=np.int64)

            # 3. CREATE FlockeData
            self.flocke_data[split] = FlockeData(u, x, y)
            self.metadata[split] = {
                'annotator_to_id': self.annotator_mapping,
                'id_to_annotator': {v: k for k, v in self.annotator_mapping.items()},
                'texts': texts,
                'annotator_ids': annotator_ids,
                'label_to_id': self.label_mapping,
                'id_to_label': {v: k for k, v in self.label_mapping.items()},
                'soft_labels': soft_labels, # NOTE: Maybe remapped_soft_labels could be used here instead
                'example_ids': example_ids,
                'n_annotators': len(self.annotator_mapping),
                'n_examples': len(set(example_ids)),
                'unique_examples_df': unique_examples_df,  # Unique examples (no duplicates)
                'unique_embeddings': embeddings.astype(np.float32)  # Embeddings before duplication
            }

        # Make sure model input dimension matches embedding size (important when loading embeddings from file)
        assert self.model_config['ninputs'] == self.flocke_data['train']._x.shape[1], f"Model input dimension {self.model_config['ninputs']} does not match embedding size {self.flocke_data['train']._x.shape[1]} found in training data (x.shape[1])."
    
    @suppress_if_not_verbose
    def train_model(self, train_params: dict = None, training_split: str | list[str] = 'train', verbose: bool = False):
        if train_params is not None:
            self.train_params.update(train_params) # Update training parameters of FlockeExperiment if provided
        mdl = MLPClassifier(ninputs = self.model_config['ninputs'],
                            ntypes = self.model_config['ntypes'],
                            nclasses = self.model_config['nclasses'],
                            hidden = self.model_config['hidden'])
        self.dr = Flocke(mdl,
                            use_accelerator=torch.cuda.is_available(),
                            prior_tempering_beta=self.train_params['prior_tempering_beta'],
                            posterior_tempering_beta_train=self.train_params['posterior_tempering_beta_train'],
                            posterior_tempering_beta_inference=self.train_params['posterior_tempering_beta_inference'])

        if isinstance(training_split, str):
            train_data = self.flocke_data[training_split]
        else:
            if not training_split:
                raise ValueError("training_splits must contain at least one split name.")
            missing_splits = [split for split in training_split if split not in self.flocke_data]
            if missing_splits:
                raise ValueError(f"Unknown training split(s): {missing_splits}")
            u_all = np.concatenate([self.flocke_data[split]._u for split in training_split])
            x_all = np.concatenate([self.flocke_data[split]._x for split in training_split], axis=0)
            y_all = np.concatenate([self.flocke_data[split]._y for split in training_split])
            train_data = FlockeData(u_all, x_all, y_all)

        self.history, self.train_pug = self.dr.fit(data = train_data,
                                                   max_epochs = self.train_params['max_epochs'],
                                                   verbose = verbose,
                                                   optimizer_params={'lr': self.train_params['lr']},
                                                   return_estimates=self.train_params['return_estimates'],
                                                   batch_size=self.train_params['batch_size'],
                                                   # p_reoptimization_period=math.ceil(len(self.flocke_data['train'])/self.train_params['batch_size']),
                                                   p_reoptimization_period=self.train_params['p_reoptimization_period'],
                                                   callbacks=[BestCheckpoint(verbose=verbose)] if self.train_params.get('use_best_checkpoint') else None)

    def group_predictions(self, split: str, groups: list[int] = None, montecarlo_sampling: bool = False) -> pd.DataFrame:
        """
        Generate group predictions for a given data split, for specified groups.
        Parameters:
            - split (str): Data split to use ('train', 'dev', 'test').
            - groups (list of int): List of group indices to consider.
            - montecarlo_sampling (bool): Whether to use Monte Carlo sampling for predictions.
        Returns:
            - pd.DataFrame: DataFrame with example IDs, predictions from all groups, predicted labels from all groups, and soft labels.
        """
        if not groups:
            groups = list(range(self.dr._ntypes))
        self.dr.mdl.to('cpu') # Ensure model is on CPU for prediction (since ClassificationModel.predict_proba works on CPU)
            
        all_groups_preds = []
        all_groups_sampled_preds = []
        x = self.metadata[split]['unique_embeddings']#.astype(np.float32)
        unique_df = self.metadata[split]['unique_examples_df']

        for group in groups:
            group_preds = self.dr.mdl.restrict(group).predict_proba(x) # Shape: (n_samples, n_classes)
            all_groups_preds.append(group_preds) # Shape: (n_groups, n_samples, n_classes)
            if not montecarlo_sampling:
                most_prob_preds = np.argmax(group_preds, axis=1) # Shape: (n_samples,)
                all_groups_sampled_preds.append(most_prob_preds) # Shape: (n_groups, n_samples)
            else:
                probs = group_preds / group_preds.sum(axis=1, keepdims=True)
                u = np.random.rand(probs.shape[0], 1) # Shape: (n_samples, 1)
                cumulative_probs = np.cumsum(probs, axis=1) # Shape: (n_samples, n_classes)
                sampled_preds = (cumulative_probs >= u).argmax(axis=1) # Shape: (n_samples,)
                all_groups_sampled_preds.append(sampled_preds) # Shape: (n_groups, n_samples)
        all_groups_preds = np.asarray(all_groups_preds) # Shape: (n_groups, n_samples, n_classes)
        all_groups_sampled_preds = np.asarray(all_groups_sampled_preds) # Shape: (n_groups, n_samples)

        n_groups = len(groups)
        n_samples = x.shape[0]

        predictions_col = []
        predicted_labels_col = []
        soft_label_col = []
        example_id_col = []

        for i in range(n_samples):
            sample_preds = [
                all_groups_preds[g, i, :].tolist()
                for g in range(n_groups)
            ]
            sample_labels = [
                int(all_groups_sampled_preds[g, i])
                for g in range(n_groups)
            ]
            predictions_col.append(sample_preds)
            predicted_labels_col.append(sample_labels)
            soft_label_col.append(list(unique_df.iloc[i]['soft_label'].values()))
            example_id_col.append(unique_df.iloc[i]['example_id'])

        group_predictions_df = pd.DataFrame({
            "example_id": example_id_col,
            "text": unique_df['text'], # Original text of the example
            "predictions": predictions_col, # Distributions over classes
            "predicted_labels": predicted_labels_col, # Most probable (or sampled, if montecarlo_sampling=True) labels
            "soft_label": soft_label_col,
        })

        self.dr.mdl.to(self.device) # Move model back to original device after prediction
        
        return group_predictions_df

    def perspectivist_predictions(self, split: str, hard_assignment: bool = True, montecarlo_sampling: bool = False) -> pd.DataFrame:
        """
        Generate perspectivist predictions for a given data split, according to who annotated each example.
        Parameters:
            - split (str): Data split to use ('train', 'dev', 'test').
            - hard_assignment (bool): Whether to use hard assignment of annotators to groups.
            - montecarlo_sampling (bool): Whether to use Monte Carlo sampling for predictions.
        Returns:
            - pd.DataFrame: DataFrame with example IDs, predicted distributions over classes, predicted classes, soft labels, and true labels.
        """
        self.dr.mdl.to('cpu') # Ensure model is on CPU for prediction (since ClassificationModel.predict_proba works on CPU)

        x = self.flocke_data[split]._x
        u = self.flocke_data[split]._u
        y = self.flocke_data[split]._y

        example_ids = self.metadata[split]["example_ids"]
        soft_labels_md = self.metadata[split]["soft_labels"]
        avg_ll_score, annotators_type_probs = self.dr.score(self.flocke_data[split], return_proba=True) # Shape: (n_samples, n_groups)

        n_groups = self.dr._ntypes
        n_samples = x.shape[0]

        # Shape: (n_groups, n_samples, n_classes)
        all_group_preds = []

        for group in range(n_groups):
            preds = self.dr.mdl.restrict(group).predict_proba(x)
            all_group_preds.append(preds)

        all_group_preds = np.asarray(all_group_preds)

        example_id_col = []
        prediction_col = []
        predicted_label_col = []
        soft_label_col = []
        true_label_col = []

        if hard_assignment:
            assigned_groups = np.argmax(annotators_type_probs, axis=1)
        for i in range(n_samples):
            annotator_idx = u[i]
            if hard_assignment:
                assigned_group = assigned_groups[annotator_idx]
                perspectivist_prediction = all_group_preds[assigned_group, i] # Shape: (n_classes,)
            else:
                all_preds_for_sample = all_group_preds[:, i, :] # Shape: (n_groups, n_classes)

                perspectivist_prediction = np.average(
                    all_preds_for_sample,
                    axis=0,
                    weights=annotators_type_probs[annotator_idx]
                )

            if montecarlo_sampling:
                p = perspectivist_prediction / perspectivist_prediction.sum()
                sampled_prediction = np.argmax(
                    np.random.multinomial(1, p.ravel())
                )
            else:
                sampled_prediction = int(np.argmax(perspectivist_prediction))

            example_id_col.append(example_ids[i])
            prediction_col.append(perspectivist_prediction.tolist())
            predicted_label_col.append(sampled_prediction)
            soft_label_col.append(list(soft_labels_md[i].values()))
            true_label_col.append(y[i])

        perspectivist_df = pd.DataFrame({
            "example_id": example_id_col,
            "prediction": prediction_col, # Distributions over classes
            "predicted_label": predicted_label_col, # Most probable (or sampled, if montecarlo_sampling=True) labels
            "soft_label": soft_label_col,
            "true_label": true_label_col,
            "avg_log_likelihood": avg_ll_score,
        })
        self.dr.mdl.to(self.device) # Move model back to original device after prediction
        return perspectivist_df

    def most_relevant_items(self, verbose=False):
        # 1. Group predictions
        group_predictions_df = self.group_predictions(split='train') # columns: example id, text, predictions, predicted labels, soft labels
        group_predictions = group_predictions_df["predictions"]

        pair_indices = list(itertools.combinations(range(self.dr._ntypes), 2))

        # 2. JS divergence between first 2 groups predictions
        # NOTE: This is kept for backward compatibility
        js_divergences = []
        for entry_idx in range(len(group_predictions)):
            group_0_pred = group_predictions[entry_idx][0]
            group_1_pred = group_predictions[entry_idx][1]
            js_divergence = jensenshannon(group_0_pred, group_1_pred)
            if np.isnan(js_divergence):
                js_divergence = 0.0
            js_divergences.append(js_divergence)
        group_predictions_df['js_divergence'] = js_divergences

        # 3. JS divergence between all pairs of groups predictions
        mri_rows = []
        for entry_idx in range(len(group_predictions)):
            example_id = group_predictions_df.iloc[entry_idx]['example_id']
            text = group_predictions_df.iloc[entry_idx]['text']
            row = {
                'id': example_id,
                'text': text,
            }
            for (a, b) in pair_indices:
                ga = np.asarray(group_predictions[entry_idx][a])
                gb = np.asarray(group_predictions[entry_idx][b])
                val = jensenshannon(ga, gb)
                if np.isnan(val):
                    val = 0.0
                row[f'js_div_g{a}_g{b}'] = val
            mri_rows.append(row)
        most_relevant_items_df = pd.DataFrame(mri_rows)

        return {
            "group_predictions_df": group_predictions_df,
            "js_divergences": js_divergences,
            "most_relevant_items_df": most_relevant_items_df,
        }


    @suppress_if_not_verbose
    def save_model(self, filepath: str, overwrite=False, verbose=False):
        """
        Save a trained Flocke model.
        """
        if not overwrite and pathlib.Path(filepath).is_file():
            print(f"File {filepath} already exists. Use overwrite=True to overwrite.")
            return
        
        checkpoint = {
            'model_state_dict': self.dr._mdl.state_dict(),
            'z_priors': self.dr._z_priors.tolist(),
            'ntypes': self.dr._ntypes,
            'use_accelerator': self.dr._use_accelerator,
            'prior_tempering_beta': self.dr._pr_beta,
            'posterior_tempering_beta_train': self.dr._ps_beta_tr,
            'posterior_tempering_beta_inference': self.dr._ps_beta_in,
            'map_semantic_override': self.dr.map_semantic_override,
            'model_config': self.model_config,
            'normalize_embeddings': self.normalize_embeddings,
            'pca_dim': self.pca_dim,
            'pca': self.pca,
        }
        
        pathlib.Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")

        # Save annotator grouping
        train_pug_path = filepath.replace('.pth', '_amap.npy')
        np.save(train_pug_path, self.train_pug)
        print(f"Annotator grouping saved to {train_pug_path}")

        # Save training history
        history_path = filepath.replace('.pth', '_history.npy')
        np.save(history_path, np.array(self.history))
        print(f"Training history saved to {history_path}")
    
    @suppress_if_not_verbose
    def load_model(self, filepath, verbose=False):
        if not pathlib.Path(filepath).is_file():
            raise FileNotFoundError(f"File {filepath} does not exist.")
        
        checkpoint = torch.load(filepath, weights_only=False, map_location=self.device)
        
        # Recreate the model using saved config
        config = checkpoint['model_config']
        self.model_config = config
        self.normalize_embeddings = checkpoint.get('normalize_embeddings', self.normalize_embeddings)
        self.pca_dim = checkpoint.get('pca_dim', self.pca_dim)
        self.pca = checkpoint.get('pca', None)
        if self.pca is not None and hasattr(self.pca, 'n_components_'):
            self.model_config['ninputs'] = self.pca.n_components_
        mdl = MLPClassifier(**config)
        mdl.load_state_dict(checkpoint['model_state_dict'])
        mdl.to(self.device)
        
        # Recreate Flocke
        self.dr = Flocke(
            mdl,
            prior_tempering_beta=checkpoint['prior_tempering_beta'],
            posterior_tempering_beta_train=checkpoint['posterior_tempering_beta_train'],
            posterior_tempering_beta_inference=checkpoint['posterior_tempering_beta_inference'],
            use_accelerator=checkpoint['use_accelerator'],
            map_semantic_override=checkpoint['map_semantic_override'],
        )
        self.dr._z_priors = np.array(checkpoint['z_priors'])

        # Load annotator grouping
        train_pug_path = filepath.replace('.pth', '_amap.npy')
        if pathlib.Path(train_pug_path).is_file():
            self.train_pug = np.load(train_pug_path, allow_pickle=True)
        else:
            self.train_pug = None
        
        # Load training history
        history_path = filepath.replace('.pth', '_history.npy')
        if pathlib.Path(history_path).is_file():
            self.history = np.load(history_path, allow_pickle=True).tolist()
        else:
            self.history = None
        
        print(f"Model loaded from {filepath}")
        print(f"Annotator grouping loaded from {train_pug_path}. Shape {self.train_pug.shape}" if self.train_pug is not None else "No annotator grouping found.")

    @suppress_if_not_verbose
    def load_or_train_model(self, filepath: str, train_params: dict = None, verbose=False):
        """
        Load a trained Flocke model if exists, otherwise train a new one.
        Return True if model was loaded, False if trained.
        """
        if pathlib.Path(filepath).is_file(): # NOTE: We use checkpath here to match the naming convention in save_model
            self.load_model(filepath, verbose=verbose)
        else:
            print(f"Model file {filepath} not found. Training a new model...")
            self.train_model(verbose=verbose, train_params=train_params)
            return False
        return True

    @suppress_if_not_verbose
    def compute_embeddings_with_ids(self, filepath: str, overwrite: bool = False, use_split_in_id: bool = False, verbose: bool = False):
        """
        Compute and save embeddings indexed by example ID to a single .npz file.
        Combines data from all splits and deduplicates by example ID (each unique example embedded once).
        Can be called right after initialization without needing prepare_data().
        
        The .npz file contains two arrays:
        - 'ids': array of unique example IDs (int64 if all IDs are integer-like, otherwise string)
        - 'embeddings': array of embeddings (one row per id, shape: n_unique_examples x embedding_dim)
        
        Parameters:
            - filepath (str): Path to save the .npz file.
            - overwrite (bool): Whether to overwrite existing file.
            - verbose (bool): Whether to print detailed information.
        
        Note: If example IDs appear in multiple splits, the embedding is computed only once.
        Returns the embeddings paired with id.
        """
        if not self.data_files:
            raise ValueError("No data files configured. Provide data_files when initializing FlockeExperiment.")
        
        if not overwrite and pathlib.Path(filepath).is_file():
            print(f"File {filepath} already exists. Use overwrite=True to overwrite. Loading existing embeddings...")
            data = np.load(filepath)
            ids = data['ids']
            embeddings = data['embeddings']
            print(f"Loaded {len(ids)} unique embeddings from {filepath}.")
            return {
                "ids": ids,
                "embeddings": embeddings,
            }
        
        # Collect unique examples across all splits
        unique_examples = {}  # {lookup_id: example_data}
        split_info = {}
        for split, json_path in self.data_files.items():
            data = load_json_data(json_path)
            split_count = 0
            for example_id, example_data in data.items():
                unique_key = self._embedding_lookup_id(example_id, split=split, use_split_in_id=use_split_in_id)
                if unique_key not in unique_examples:
                    unique_examples[unique_key] = example_data
                    split_count += 1
            split_info[split] = split_count
        
        # Sort by example ID for consistency
        sorted_example_ids = sorted(unique_examples.keys(), key=self._id_sort_key)
        
        # Extract texts and embed
        texts = [self._normalize_text(unique_examples[example_id]["text"]) for example_id in sorted_example_ids]
        embeddings = self.embed_texts(texts)
        
        # Keep int64 when possible for backward compatibility; otherwise store as strings.
        if all(self._is_int_like(example_id) for example_id in sorted_example_ids):
            ids_array = np.array([int(str(example_id).strip()) for example_id in sorted_example_ids], dtype=np.int64)
        else:
            ids_array = np.array([str(example_id) for example_id in sorted_example_ids], dtype=str)
        embeddings_array = embeddings.astype(np.float32)
        
        # Ensure we have the same number of ids and embeddings
        assert len(ids_array) == embeddings_array.shape[0], \
            f"Mismatch between number of ids ({len(ids_array)}) and embeddings ({embeddings_array.shape[0]})"
        
        # Create parent directory if it doesn't exist
        pathlib.Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to .npz file (compressed for efficiency)
        np.savez_compressed(filepath, ids=ids_array, embeddings=embeddings_array)
        
        file_size_mb = pathlib.Path(filepath).stat().st_size / (1024 * 1024)
        print(f"Embeddings saved to {filepath}")
        print(f"  - Unique examples: {len(ids_array)}")
        print(f"  - Embedding dimension: {embeddings_array.shape[1]}")
        print(f"  - File size: {file_size_mb:.2f} MB")
        print(f"  - Split breakdown (before deduplication):")
        for split, count in split_info.items():
            print(f"    - {split}: {count} examples")
        
        return {
            "ids": ids_array,
            "embeddings": embeddings_array,
        }


@suppress_if_not_verbose
def annotator_disagreement(dr_exp, num_annotators, annotations_per_entry, verbose=False) -> dict:
    """
    Return a dict of stats.
    """
    example_annotations = {}
    for i, example_id in enumerate(dr_exp.metadata['train']['example_ids']):
        annotator_idx = dr_exp.flocke_data['train']._u[i]
        annotator = [ann for ann, idx in dr_exp.annotator_mapping.items() if idx == annotator_idx][0]
        label = dr_exp.flocke_data['train']._y[i]
        if example_id not in example_annotations:
            example_annotations[example_id] = []
        example_annotations[example_id].append((annotator, label))
    
    disagreement_bins = defaultdict(list)
    for example_id, annotations in example_annotations.items():
        label_counts = {}
        for _, label in annotations:
            label_counts[label] = label_counts.get(label, 0) + 1
        sorted_counts = sorted(label_counts.values(), reverse=True)
        if len(sorted_counts) == 1:
            bin_label = f"{sorted_counts[0]}v0"
        else:
            bin_label = "v".join(map(str, sorted_counts))
        disagreement_bins[bin_label].append(example_id)
    def bin_key(bin_label):
        return tuple(map(int, bin_label.split("v")))
    sorted_bin_labels = sorted(
        disagreement_bins.keys(),
        key=bin_key,
        reverse=True
    )
    counts = [len(disagreement_bins[b]) for b in sorted_bin_labels]
    total = sum(counts)

    # useful_items = total - len(disagreement_bins.get(f"{annotations_per_entry}v0", [0]))
    zero_disagreement = sum(len(v) for k, v in disagreement_bins.items() if k.endswith("v0"))
    useful_items = total - zero_disagreement
    avg_useful_items_per_annotator = useful_items / num_annotators * annotations_per_entry
    avg_items_per_annotator = total / num_annotators * annotations_per_entry

    return {
        "total_examples": total,
        "useful_items": useful_items,
        "avg_useful_items_per_annotator": avg_useful_items_per_annotator,
        "avg_items_per_annotator": avg_items_per_annotator,
        "bin_counts": {bin_label: len(examples) for bin_label, examples in disagreement_bins.items()},
        "disagreement_bins": disagreement_bins,
        "sorted_bin_labels": sorted_bin_labels,
        "counts": counts,
    }


@suppress_if_not_verbose
def grouping_performance(dr_exp, info_json_path, split='train', verbose=False) -> dict:
    groups = {group_idx: [] for group_idx in range(dr_exp.model_config['ntypes'])}
    for annotator_id in range(len(dr_exp.train_pug)):
        most_probable_group = int(np.argmax(dr_exp.train_pug[annotator_id]))
        annotator_name = dr_exp.metadata[split]['id_to_annotator'][annotator_id]
        groups[most_probable_group].append((annotator_name, dr_exp.train_pug[annotator_id, most_probable_group]))
    predicted_groups = {i: [annotator for annotator, prob in annotators] for i, annotators in groups.items()}

    info_data = load_json_data(info_json_path)
    true_groups = info_data['groups']
    annotator_to_true_group = {}
    for group_name, group_info in true_groups.items():
        for annotator_name in group_info['annotators']:
            annotator_to_true_group[annotator_name] = group_name
    gt_groups = {group_name: group_info['annotators'] for group_name, group_info in true_groups.items()}
    print("Predicted Groups:", predicted_groups)
    print("Ground Truth Groups:", gt_groups)

    return clustering_performance(gt_groups, predicted_groups)


# ===============================================
# ANALYSES (shared by validation.py and experiments.py)
# ===============================================

def predicted_groups(dr_exp, split='train') -> dict:
    """
    Hard clustering of the annotators observed in `split`: each annotator is assigned to its most probable group.
    Returns {group_idx: [annotator names]}.
    """
    observed = set(int(a) for a in np.unique(dr_exp.flocke_data[split]._u))
    groups = {group_idx: [] for group_idx in range(dr_exp.model_config['ntypes'])}
    for annotator_id in range(len(dr_exp.train_pug)):
        if annotator_id not in observed:
            continue
        most_probable_group = int(np.argmax(dr_exp.train_pug[annotator_id]))
        groups[most_probable_group].append(dr_exp.metadata[split]['id_to_annotator'][annotator_id])
    return groups


def likelihood_score(dr_exp, split='train') -> float:
    """
    Likelihood score (exponential of the average log-likelihood) of the trained model on `split`.
    """
    avg_ll_score, _ = dr_exp.dr.score(dr_exp.flocke_data[split], return_proba=True)
    return float(np.exp(avg_ll_score))


def hatecheck_grouping(dr_exp, dataset) -> dict:
    """
    Compare predicted annotator groups with the ground-truth groups of a synthetic HateCheckData.
    """
    pred_groups = predicted_groups(dr_exp)
    observed = set(a for annotators in pred_groups.values() for a in annotators)
    gt_groups = {name: [a for a in annotators if a in observed] for name, annotators in dataset.groups.items()}
    return clustering_performance(gt_groups, pred_groups)


def build_ground_truth_df(dataset) -> pd.DataFrame:
    """
    One row per HateCheck item, with the ground-truth label (0/1) of each synthetic group in column '{group_name}_label'.
    """
    rows = []
    for _, row in dataset.df.iterrows():
        try:
            case_id = int(row['case_id'])
        except (TypeError, ValueError):
            case_id = row['case_id']
        labels_by_group = dataset._get_group_labels_for_row(row=row, original_label=row['label_gold'])
        gt_row = {'example_id': case_id}
        for group_name, label in labels_by_group.items():
            gt_row[f'{group_name}_label'] = 1 if label == 'hateful' else 0
        rows.append(gt_row)
    return pd.DataFrame(rows)


def group_annotation_f1(dr_exp, dataset, gt_to_pred_mapping: dict, split='train') -> float:
    """
    Group-level annotation prediction: macro-F1 between the labels predicted by each group and the ground-truth labels
    of the matched synthetic group, averaged across groups.
    """
    group_predictions_df = dr_exp.group_predictions(split)
    n_groups = len(group_predictions_df['predicted_labels'].iloc[0])
    predicted_labels_df = pd.DataFrame(group_predictions_df['predicted_labels'].tolist(), columns=[f'group{i}_predicted_label' for i in range(n_groups)])
    group_predictions_df = pd.concat([group_predictions_df.drop(columns=['predicted_labels']), predicted_labels_df], axis=1)

    gt_df = build_ground_truth_df(dataset)
    gt_df = gt_df[gt_df['example_id'].isin(group_predictions_df['example_id'])]
    merged = group_predictions_df.merge(gt_df, on='example_id', how='inner')

    f1_scores = []
    for gt_group, pred_group in gt_to_pred_mapping.items():
        f1_scores.append(f1_score(merged[f'{gt_group}_label'], merged[f'group{pred_group}_predicted_label'], average='macro'))
    return float(np.mean(f1_scores))


def most_relevant_items_f1(dr_exp, dataset, pred_to_gt_mapping: dict) -> dict:
    """
    Most relevant items retrieval: for each pair of predicted groups, the top-K items by JS divergence are compared with the
    K items on which the matched ground-truth groups disagree. Returns the macro-F1 averaged across pairs, and per pair.
    """
    true_mri_all = dataset._disagreement_df
    pred_mri_all = dr_exp.most_relevant_items()['most_relevant_items_df']

    f1_by_pair = {}
    for g1, g2 in itertools.combinations(range(dr_exp.model_config['ntypes']), 2):
        pred_mri_sorted = pred_mri_all.sort_values(by=f'js_div_g{g1}_g{g2}', ascending=False)
        g1_gt, g2_gt = pred_to_gt_mapping[g1], pred_to_gt_mapping[g2]
        true_mri = true_mri_all[true_mri_all['disagreeing_group_pairs'].apply(lambda pairs: (g1_gt, g2_gt) in pairs or (g2_gt, g1_gt) in pairs)]
        K = len(true_mri)
        if K == 0:
            continue
        true_labels = pred_mri_sorted['id'].isin(true_mri['id']).astype(int).to_numpy()
        pred_labels = np.zeros(len(pred_mri_sorted), dtype=int)
        pred_labels[:K] = 1
        f1_by_pair[(g1, g2)] = float(f1_score(true_labels, pred_labels, average='macro'))
    return {
        'f1': float(np.mean(list(f1_by_pair.values()))) if f1_by_pair else float('nan'),
        'f1_by_pair': f1_by_pair,
    }


def pairwise_js_divergences(group_predictions_df: pd.DataFrame, ntypes: int) -> pd.DataFrame:
    """
    Per-item Jensen-Shannon divergence between the label distributions predicted by each pair of groups
    (columns 'js_div_g{a}_g{b}'), plus their average ('avg_js_div').
    """
    rows = []
    for predictions in group_predictions_df['predictions']:
        row = {}
        for a, b in itertools.combinations(range(ntypes), 2):
            val = jensenshannon(np.asarray(predictions[a]), np.asarray(predictions[b]))
            row[f'js_div_g{a}_g{b}'] = 0.0 if np.isnan(val) else float(val)
        rows.append(row)
    js_df = pd.DataFrame(rows, index=group_predictions_df.index)
    js_df['avg_js_div'] = js_df.mean(axis=1) if ntypes > 1 else 0.0
    return pd.concat([group_predictions_df[['example_id', 'text']], js_df], axis=1)

