# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import numpy as np
from torch.utils.data import Sampler
from collections import defaultdict

class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.label_to_indices[label].append(idx)
        self.labels_set = list(set(self.labels))
        self.num_classes = len(self.labels_set)
        assert batch_size % self.num_classes == 0, "Batch size must be divisible by number of classes"
        self.samples_per_class = batch_size // self.num_classes

    def __iter__(self):
        # Shuffle indices for each class
        for label in self.labels_set:
            np.random.shuffle(self.label_to_indices[label])
        # Calculate how many batches we can make
        min_class_len = min([len(self.label_to_indices[label]) for label in self.labels_set])
        num_batches = min_class_len // self.samples_per_class
        for i in range(num_batches):
            batch = []
            for label in self.labels_set:
                start = i * self.samples_per_class
                end = start + self.samples_per_class
                batch.extend(self.label_to_indices[label][start:end])
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        min_class_len = min([len(self.label_to_indices[label]) for label in self.labels_set])
        return (min_class_len // self.samples_per_class) * self.num_classes 