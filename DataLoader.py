import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self, trainings_data_folder_path: str, timesteps: int):
        self.timesteps = timesteps
        self.trainings_data_folder_path = trainings_data_folder_path
        self.error_folder_path = join(trainings_data_folder_path, "errors")

    def get_error(self, path):
        trajectory = path.split("/")[-1]
        trajectory_name = f"{trajectory}.csv"
        error_file = join(self.error_folder_path, trajectory_name)
        df = pd.read_csv(error_file, sep=",", header=None)
        return np.array(df.values[0])

    def extract_sample_files(self, calculated_trajectory_path):
        only_files = [f for f in listdir(calculated_trajectory_path) if isfile(join(calculated_trajectory_path, f))]
        filter_name = '.py'
        only_py_files = list(filter(lambda filename: filter_name in filename, only_files))
        only_py_files.sort()
        size_of_folder = len(only_py_files)
        every_x_steps = (size_of_folder - 1) / (self.timesteps - 1)

        assert (every_x_steps == int(every_x_steps))
        sample = []
        for i in range(0, size_of_folder, int(every_x_steps)):
            sample.append(only_py_files[i])
        return sorted(sample)

    @staticmethod
    def get_values_of_channel(channel_number, file):
        data = file['data']
        channels = data[1]['components']
        return np.array(channels[channel_number]['values'])

    def extract_channels_from_sample(self, calculated_trajectory_path, file_name):
        path = join(calculated_trajectory_path, file_name)
        file = np.load(path, allow_pickle=True)
        channels = np.zeros(shape=(4, 1191))
        for i in range(4):
            channels[i] = self.get_values_of_channel(i, file)
        return channels.T  # for analogy of RGB channels in images (channel is last dimension)

    def get_single_sample(self, calculated_trajectory_path):
        sample_files = self.extract_sample_files(calculated_trajectory_path)
        sample = np.zeros(shape=(self.timesteps, 1191, 4))
        for index, file_name in enumerate(sample_files):
            sample[index] = self.extract_channels_from_sample(calculated_trajectory_path, file_name)
        return np.array(sample)

    def get_data(self, limit: int = None):
        only_files = [f for f in listdir(self.error_folder_path) if isfile(join(self.error_folder_path, f))]
        filter_name = '.csv'
        only_csv_files = list(filter(lambda filename: filter_name in filename, only_files))
        only_csv_files.sort()

        if limit:
            limit = min(limit, len(only_csv_files))
            only_csv_files = only_csv_files[:limit]

        trainings_data = np.zeros(shape=(len(only_csv_files), self.timesteps, 1191, 4))
        errors = np.zeros(shape=(len(only_csv_files), 2))
        for index, filename in enumerate(only_csv_files):
            directory = join(self.trainings_data_folder_path, filename)
            directory = directory[:-4]  # get rid of ".csv"
            trainings_data[index] = self.get_single_sample(directory)
            errors[index] = self.get_error(directory)
        return trainings_data, np.log10(errors)

    def get_training_and_validation_data(self, seed, test_size: float = 0.2):
        """
        Loads the Training and validation data where X_i:= 4 channels of the i-th trajectory over the t timesteps 
        and y_i:= is the corresponding error
        :param seed: the seed for the random state of the train test split
        :param test_size: If float, should be between 0.0 and 1.0 and represent the proportion of the
        dataset to include in the test split. Defaults to 0.2
        :return: X_train, X_validation, y_train, y_validation
        """
        X, y = self.get_data()
        return train_test_split(X, y, test_size=test_size, random_state=seed)