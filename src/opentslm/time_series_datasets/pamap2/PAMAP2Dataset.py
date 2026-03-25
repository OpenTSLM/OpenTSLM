# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

from typing import Tuple
import pandas as pd
from opentslm.time_series_datasets.pamap2.pamap2_loader import ensure_pamap2_data
from torch.utils.data import Dataset

ACTIVITIY_ID_DICT = {
    0: "transient",
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "nordic walking",
    9: "watching TV",
    10: "computer work",
    11: "car driving",
    12: "ascending stairs",
    13: "descending stairs",
    16: "vacuum cleaning",
    17: "ironing",
    18: "folding laundry",
    19: "house cleaning",
    20: "playing soccer",
    24: "rope jumping",
}


class PAMAP2Dataset(Dataset):
    # source: https://github.com/andreasKyratzis/PAMAP2-Physical-Activity-Monitoring-Data-Analysis-and-ML/blob/master/pamap2.ipynb
    def _data_cleaning(self, dataCollection):
        dataCollection = dataCollection.drop(
            [
                "handOrientation1",
                "handOrientation2",
                "handOrientation3",
                "handOrientation4",
                "chestOrientation1",
                "chestOrientation2",
                "chestOrientation3",
                "chestOrientation4",
                "ankleOrientation1",
                "ankleOrientation2",
                "ankleOrientation3",
                "ankleOrientation4",
            ],
            axis=1,
        )  # removal of orientation columns as they are not needed
        dataCollection = dataCollection.drop(
            dataCollection[dataCollection.activityID == 0].index
        )  # removal of any row of activity 0 as it is transient activity which it is not used
        dataCollection = dataCollection.apply(
            pd.to_numeric, errors="coerce"
        )  # removal of non numeric data in cells
        dataCollection = dataCollection.interpolate()  # removal of any remaining NaN value cells by constructing new data points in known set of data points

        return dataCollection

    def _load_data(self, list_of_files):
        # Load data
        ensure_pamap2_data()

        colNames = ["timestamp", "activityID", "heartrate"]
        IMUhand = [
            "handTemperature",
            "handAcc16_1",
            "handAcc16_2",
            "handAcc16_3",
            "handAcc6_1",
            "handAcc6_2",
            "handAcc6_3",
            "handGyro1",
            "handGyro2",
            "handGyro3",
            "handMagne1",
            "handMagne2",
            "handMagne3",
            "handOrientation1",
            "handOrientation2",
            "handOrientation3",
            "handOrientation4",
        ]

        IMUchest = [
            "chestTemperature",
            "chestAcc16_1",
            "chestAcc16_2",
            "chestAcc16_3",
            "chestAcc6_1",
            "chestAcc6_2",
            "chestAcc6_3",
            "chestGyro1",
            "chestGyro2",
            "chestGyro3",
            "chestMagne1",
            "chestMagne2",
            "chestMagne3",
            "chestOrientation1",
            "chestOrientation2",
            "chestOrientation3",
            "chestOrientation4",
        ]

        IMUankle = [
            "ankleTemperature",
            "ankleAcc16_1",
            "ankleAcc16_2",
            "ankleAcc16_3",
            "ankleAcc6_1",
            "ankleAcc6_2",
            "ankleAcc6_3",
            "ankleGyro1",
            "ankleGyro2",
            "ankleGyro3",
            "ankleMagne1",
            "ankleMagne2",
            "ankleMagne3",
            "ankleOrientation1",
            "ankleOrientation2",
            "ankleOrientation3",
            "ankleOrientation4",
        ]

        columns = colNames + IMUhand + IMUchest + IMUankle  # all columns in one list
        dataCollection = pd.DataFrame()
        for file in list_of_files:
            procData = pd.read_table(file, header=None, sep=r"\s+")
            procData.columns = columns
            procData["subject_id"] = int(file[-5])
            dataCollection = pd.concat([dataCollection, procData], ignore_index=True)

        dataCollection.reset_index(drop=True, inplace=True)
        dataCol = self._data_cleaning(dataCollection)
        dataCol.reset_index(drop=True, inplace=True)
        for i in range(0, 4):
            dataCol.loc[i, "heartrate"] = 100
        dataCol["activityID"] = dataCol["activityID"].map(ACTIVITIY_ID_DICT)
        return dataCol

    def __init__(self, list_of_files):
        super().__init__()
        self.df = self._load_data(list_of_files)

        # create 3-second windows and store them as tensors + labels
        self.time_series, self.labels = self._make_windows(
            window_size="3s", min_pct=0.5
        )

    def _make_windows(self, window_size, min_pct=0.5):
        """
        Returns:
          windows: list of numpy arrays of shape (n_features, n_steps)
          labels : list of activity labels (the mode of each window)
        Drops any window whose modal activity < min_pct of that window's rows,
        and only ever groups rows from the same subject.
        """
        # 1) copy & timestamp → datetime → index
        df = self.df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df.set_index("timestamp")

        # 2) define which columns are features (everything except label & subject_id)
        feature_cols = df.columns.drop(["activityID"])

        windows = []
        labels = []

        # 3) first split by subject_id so no window can span subjects
        for subject, df_sub in df.groupby("subject_id"):
            # align your 2‑minute bins to this subject’s first timestamp
            origin = df_sub.index[0]
            for window_start, win in df_sub.resample(window_size, origin=origin):
                if win.empty:
                    continue

                # 4) find most common activity in this window
                mode_ser = win["activityID"].mode()
                if mode_ser.empty:
                    continue
                mode = mode_ser.iloc[0]

                # 5) check that it covers at least min_pct of the rows
                if (win["activityID"] == mode).sum() < min_pct * len(win):
                    continue

                # 6) store the **transposed** feature‑array + label
                #    now the shape is (n_features, n_steps)

                ts_dict = {col: win[col].values for col in feature_cols}
                windows.append(ts_dict)
                labels.append(mode)

        return windows, labels

    def __len__(self):
        # now based on windows, not raw rows
        return len(self.time_series)

    def __getitem__(self, idx):
        # returns (n_steps × n_features array, activity string)
        return {"time_series": self.time_series[idx], "label": self.labels[idx]}


if __name__ == "__main__":
    dataset = PAMAP2Dataset()

    for data_point in dataset:
        window = data_point["time_series"]
        label = data_point["label"]
        print(f"{window}, {label}")
