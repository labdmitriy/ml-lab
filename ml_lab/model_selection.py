"""Implements model selection for time series."""

from itertools import groupby
from typing import Iterable, Iterator, Literal, Optional, Tuple

import numpy as np
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples


class GroupTimeSeriesSplit:
    """Time series cross validation with custom grouping."""

    def __init__(
        self,
        test_size: int,
        train_size: Optional[int] = None,
        n_splits: Optional[int] = None,
        gap: int = 0,
        shift_size: int = 1,
        window: Literal['rolling', 'expanding'] = 'rolling'
    ):
        """Initializes cross validation parameters.

        Args:
            test_size (int):
                Size of test dataset.
            train_size (Optional[int], optional):
                Size of train dataset. Defaults to None.
            n_splits (int, optional):
                Number of splits. Defaults to None.
            gap (int, optional):
                Gap size. Defaults to 0.
            shift_size (int, optional):
                Step to shift for the next fold. Defaults to 1.
            window (str):
                Type of the window. Defaults to 'rolling'.
        """
        self.test_size = test_size
        self.train_size = train_size
        self.n_splits = n_splits
        self.gap = gap
        self.shift_size = shift_size
        self.window = window

    def split(self,
              X: Iterable,
              y: Optional[Iterable] = None,
              groups: Optional[Iterable] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Calculates train/test indices based on split parameters.

        Args:
            X (Iterable): Dataset with features.
            y (Iterable): Dataset with target.
            groups (Iterable): Array with group numbers.

        Yields:
            Iterator[Tuple[np.ndarray, np.ndarray]]: Train/test dataset indices.
        """
        self._check_split_params()

        # train_size = self.train_size
        test_size = self.test_size
        # n_splits = self.n_splits
        gap = self.gap
        shift_size = self.shift_size

        # Convert to indexable data structures with additional lengths consistency check
        X, y, groups = indexable(X, y, groups)

        # Check if groups are specified
        if groups is None:
            raise ValueError('Groups must be specified')

        # Check if groups are sorted in dataset
        group_seqs = [group[0] for group in groupby(groups)]
        unique_groups, group_starts_idx = np.unique(groups, return_index=True)
        n_groups = _num_samples(unique_groups)
        self.n_groups = n_groups

        if group_seqs != sorted(unique_groups):
            raise ValueError('Groups must be presorted in increasing order')

        # Create mapping between groups and its start indices in array
        groups_dict = dict(zip(unique_groups, group_starts_idx))

        # Calculate number of samples
        n_samples = _num_samples(X)

        # Calculate remaining split params
        train_size, n_splits, train_start_idx = self._calculate_split_params()

        # Calculate start/end indices for initial train/test datasets
        train_end_idx = train_start_idx + train_size
        test_start_idx = train_end_idx + gap
        test_end_idx = test_start_idx + test_size

        # Process each split
        for _ in range(n_splits):
            # Calculate train indices range
            train_idx = np.r_[slice(groups_dict[train_start_idx], groups_dict[train_end_idx])]

            # Calculate test indices range
            if test_end_idx < n_groups:
                test_idx = np.r_[slice(groups_dict[test_start_idx], groups_dict[test_end_idx])]
            else:
                test_idx = np.r_[slice(groups_dict[test_start_idx], n_samples)]

            # Yield train/test indices range
            yield (train_idx, test_idx)

            # Shift train dataset start index by shift size for rolling window
            if self.window == 'rolling':
                train_start_idx = train_start_idx + shift_size

            # Shift train dataset end index by shift size
            train_end_idx = train_end_idx + shift_size

            # Shift test dataset indices range by shift size
            test_start_idx = test_start_idx + shift_size
            test_end_idx = test_end_idx + shift_size

    def get_n_splits(
        self, X: Iterable, y: Optional[Iterable] = None, groups: Optional[Iterable] = None
    ) -> int:
        """Calculates number of splits given specified parameters.

        Args:
            X (Iterable): Dataset with features. Defaults to None.
            y (Optional[Iterable], optional): Dataset with target. Defaults to None.
            groups (Optional[Iterable], optional): Array with group numbers. Defaults to None.

        Returns:
            int: Calculated number of splits.
        """
        if self.n_splits is not None:
            return self.n_splits
        else:
            raise ValueError('Number of splits is not defined')

    def _check_split_params(self):
        if (self.train_size is None) and (self.n_splits is None):
            raise ValueError('Either train_size or n_splits have to be defined')

        if self.window not in ['rolling', 'expanding']:
            raise ValueError('Window can be either "rolling" or "expanding"')

        if (self.train_size is not None) and (self.window == 'expanding'):
            raise ValueError('Train size can be specified only with rolling window')

    def _calculate_split_params(self):
        train_size = self.train_size
        test_size = self.test_size
        n_splits = self.n_splits
        gap = self.gap
        shift_size = self.shift_size
        n_groups = self.n_groups

        not_enough_data_error = (
            'Not enough data to split number of groups ({0})'
            ' for number splits ({1})'
            ' with train size ({2}),'
            ' test size ({3}), gap ({4}), shift_size ({5})'
        )

        if train_size is None:
            train_size = int(n_groups - gap - (test_size + (n_splits - 1) * shift_size))
            self.train_size = train_size

            if train_size <= 0:
                raise ValueError(
                    not_enough_data_error.format(
                        n_groups, n_splits, train_size, test_size, gap, shift_size
                    )
                )
            train_start_idx = 0
        elif self.n_splits is None:
            n_splits = ((n_groups - train_size - gap - test_size) // shift_size) + 1
            self.n_splits = n_splits

            if n_splits <= 0:
                raise ValueError(
                    not_enough_data_error.format(
                        n_groups, n_splits, train_size, test_size, gap, shift_size
                    )
                )
            train_start_idx = n_groups - train_size - gap - test_size - (n_splits - 1) * shift_size
        else:
            train_start_idx = n_groups - train_size - gap - test_size - (n_splits - 1) * shift_size

            if train_start_idx < 0:
                raise ValueError(
                    not_enough_data_error.format(
                        n_groups, n_splits, train_size, test_size, gap, shift_size
                    )
                )

        return train_size, n_splits, train_start_idx
