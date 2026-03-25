# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import logging
import os

import numpy as np

from tqdm.auto import tqdm


from opentslm.time_series_datasets.monash.monash_utils import (
    download_and_extract_monash_ucr,
    load_from_tsfile_to_dataframe,
)


class MonashDataset:
    def __init__(self, _data_dir=None, data_name=None):
        self.logger = logging.getLogger(__name__)
        self._data_dir = _data_dir
        self.data_name = data_name

        if not os.path.exists(_data_dir):
            download_and_extract_monash_ucr(destination="monash_datasets")
        dataset_file = os.path.join(_data_dir, f"{data_name}.ts")

        # Load the dataset
        print(f"Loading dataset: {data_name}")
        X, y = load_from_tsfile_to_dataframe(dataset_file, return_separate_X_and_y=True)

        # Convert sktime’s nested DataFrame to a 3D NumPy array:
        #   shape = (n_instances, n_dimensions, series_length)
        n_samples, n_dims = X.shape
        # assume all series in a given column have the same length
        series_length = X.iloc[0, 0].to_numpy().shape[0]

        # Converting from a dataframe that contains a list of time series
        # to a 3d tensor where X_np[idx] gives the 2d tensor of the time series
        # at the position idx equivalent to the df[idx] time series list
        X_np = np.zeros((n_samples, n_dims, series_length), dtype=float)
        for i in range(n_samples):
            for j in range(n_dims):
                X_np[i, j, :] = X.iloc[i, j].to_numpy()

        y_np = np.array(y)

        self.feature = X_np
        self.target = y_np

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        item = self.feature[idx]
        label = self.target[idx]
        item = np.expand_dims(item, axis=0)

        return {"time_series": item, "answer": label}


if __name__ == "__main__":
    loader = MonashDataset(
        _data_dir="monash_datasets", data_name="IEEEPPG/IEEEPPG_TRAIN"
    )

    prog = tqdm(loader)

    for item, label in prog:
        print("item", item, "; label", label)
