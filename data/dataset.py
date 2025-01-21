import os

import numpy as np
import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader, IterableDataset


class CS2Dataset(IterableDataset):
    """Counter-Strike 2 player trajectory dataset over keyboard/mouse signals."""

    def __init__(self, demo_directory: str, random_seed: int = 410, n_iter: int = 1_000, batch_size: int = 32,
                 p: float = 0.5):
        """Construct a new CS2 dataset over the given demo directory.

        :param demo_directory: the directory containing the demo data.
        :param random_seed: random seed set for repeatable samples.
        :param n_iter: number of random samples per epoch.
        :param p: probability of sampling a positive class sample. Default is 0.5.
        """

        # the probability [0, 1) of which a positive class is sampled during training.
        # positive class indicates trajectories belong to same player
        # negative class indicates trajectories belong to different players
        self._p = p

        # number of batches to yield during each epoch of training
        self._n_iter = n_iter

        # the batch size controls how many trajectory pairs are sampled during one iteration.
        self._batch_size = batch_size

        # create random number generator for consistent sampling
        self._generator = torch.Generator().manual_seed(random_seed)

        # collect all numpy array filenames, these will make up our dataset
        self._demo_directory = demo_directory
        self._filenames = [filename for filename in os.listdir(self._demo_directory) if filename.endswith(".npy")]

        # map trajectory filenames by player ID
        self._trajectories = dict()
        for filename in self._filenames:
            match_id_unused, demo_id_unused, player_id, round_number_unused = filename.split("_")
            if player_id not in self._trajectories:
                self._trajectories[player_id] = []
            self._trajectories[player_id].append(filename)
        self._player_ids = sorted(self._trajectories.keys())

    def __iter__(self):
        """Yield a generator over randomly sampled batches for one epoch from this dataset."""
        for idx in range(self._n_iter):
            batch = self._accumulate_batch()
            yield batch

    def _accumulate_batch(self):
        """Accumulate a single, randomly sampled batch of trajectory pairs.

        :returns: a single batch.
        """
        samples_left = []
        samples_right = []
        class_labels = []
        for i in range(self._batch_size):
            # randomly select either positive or negative class
            is_positive_class = self._random_p() < self._p
            if is_positive_class:
                sample_left, sample_right, class_label = self._sample_positive_class()
            else:
                sample_left, sample_right, class_label = self._sample_negative_class()
            samples_left.append(sample_left)
            samples_right.append(sample_right)
            class_labels.append(class_label)

        return (
            self._pad_and_concat(samples_left),
            self._pad_and_concat(samples_right),
            torch.tensor(class_labels)
        )

    def _load_sample(self, filename: str):
        """Load a .npy array from the filename and return it as a tensor.

        :param filename: the sample filename.
        :return: the tensor.
        """
        return torch.from_numpy(
            np.load(
                os.path.join(self._demo_directory, filename), allow_pickle=True
            ).astype(np.float32)
        )

    def _pad_and_concat(self, samples: list[torch.Tensor]) -> torch.Tensor:
        """Pad and concatenate jagged samples together into one tensor.

        :param samples: list of samples to pad.

        :return: padded samples."""
        max_dim = max(map(lambda sample: sample.shape[0], samples))
        padded_samples = [
            pad(sample, (0, 0, 0, max_dim - sample.shape[0])) for sample in samples
        ]
        return torch.stack(padded_samples, dim=0)

    def _random_index(self, length: int):
        """Sample a random index from a list of arbitrary length.

        :param length: the length of the list to sample an index for.
        :return: the index.
        """
        return int(torch.rand((1,), generator=self._generator).item() * length)

    def _random_p(self):
        """Return a random float in the range [0, 1).

        This float is interpreted as the random event of sampling a positive or negative class from the dataset.

        :return: the float.
        """
        return self._random_index(1)

    def _random_player_idx(self):
        """Sample a random player index from the dataset.

        :return: the player index.
        """
        return self._random_index(len(self._player_ids))

    def _sample_negative_class(self):
        """Sample a negative pair of trajectories from the dataset.

        A negative sample signifies that the trajectories belong to different players.

        :return: the samples and class, as a three-tuple (trajectory_1, trajectory_2, class_label).
        """
        player_1 = self._player_ids[self._random_player_idx()]
        player_2 = self._player_ids[self._random_player_idx()]
        while player_1 == player_2:
            player_2 = self._player_ids[self._random_player_idx()]
        trajectories_1 = self._trajectories[player_1]
        trajectory_idx_1 = self._random_index(len(trajectories_1))
        trajectory_1 = self._load_sample(trajectories_1[trajectory_idx_1])
        trajectories_2 = self._trajectories[player_2]
        trajectory_idx_2 = self._random_index(len(trajectories_2))
        trajectory_2 = self._load_sample(trajectories_2[trajectory_idx_2])
        return trajectory_1, trajectory_2, 0

    def _sample_positive_class(self):
        """Sample a positive pair of trajectories from the dataset.

        A positive sample signifies that the trajectories belong to the same player.

        :return: the samples and class, as a three-tuple (trajectory_1, trajectory_2, class_label).
        """
        player = self._player_ids[self._random_player_idx()]
        trajectories = self._trajectories[player]
        trajectory_idx_1 = self._random_index(len(trajectories))
        trajectory_idx_2 = self._random_index(len(trajectories))
        trajectory_1 = self._load_sample(trajectories[trajectory_idx_1])
        trajectory_2 = self._load_sample(trajectories[trajectory_idx_2])
        return trajectory_1, trajectory_2, 1
