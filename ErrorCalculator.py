from os import listdir
from os.path import isfile, isdir, join
import numpy as np
import logging
import re


class ErrorCalculator:
    def __init__(self, trainings_data_folder_path: str):
        self.trainings_data_folder_path = trainings_data_folder_path

    @staticmethod
    def get_end_result_file_path(folder_path):
        only_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
        fibre_filter_name = "fibre_0_0D"
        only_fibre_correct_dimension = list(filter(lambda filename: fibre_filter_name in filename, only_files))
        only_fibre_correct_dimension.sort()
        return f"{folder_path}/{only_fibre_correct_dimension[-1]}"

    @staticmethod
    def get_values_of_channel(channel_number, path):
        file = np.load(path, allow_pickle=True)
        data = file['data']
        channels = data[1]['components']
        return np.array(channels[channel_number]['values'])

    def calculate_error(self, path, trajectory_name, nt_0D, nt_1D):
        trajectory = f"{path}/{trajectory_name}"
        max_accuracy = 32
        coarse_calculation_path = f"{trajectory}-{nt_0D}-{nt_1D}"
        nt_0D_fine_calculation_path = f"{trajectory}-{max_accuracy}-{nt_1D}"
        nt_1D_fine_calculation_path = f"{trajectory}-{nt_0D}-{max_accuracy}"
        coarse_calculation = self.get_values_of_channel(0, self.get_end_result_file_path(coarse_calculation_path))
        nt_0D_fine_calculation = self.get_values_of_channel(0,
                                                            self.get_end_result_file_path(nt_0D_fine_calculation_path))
        nt_1D_fine_calculation = self.get_values_of_channel(0,
                                                            self.get_end_result_file_path(nt_1D_fine_calculation_path))
        assert (np.shape(coarse_calculation) == np.shape(nt_0D_fine_calculation) == np.shape(nt_1D_fine_calculation))
        error0 = np.linalg.norm(coarse_calculation - nt_0D_fine_calculation)
        error1 = np.linalg.norm(coarse_calculation - nt_1D_fine_calculation)
        return np.array([error0, error1])

    def calculate_all_errors_for_trajectory(self, trajectory_name):
        errors_folder = join(self.trainings_data_folder_path, "errors")
        for nt_0D in [2, 4, 8, 16]:
            for nt_1D in [1, 2, 4, 8, 16]:
                errors = self.calculate_error(self.trainings_data_folder_path, trajectory_name, nt_0D, nt_1D)
                logging.info(errors, trajectory_name, nt_0D, nt_1D)
                error_filename = f"{trajectory_name}-{nt_0D}-{nt_1D}.csv"
                error_output_path = join(errors_folder, error_filename)
                np.savetxt(error_output_path, [errors], delimiter=",")

    def calculate_all_errors(self):
        only_folders = [f for f in listdir(self.trainings_data_folder_path) if isdir(join(self.trainings_data_folder_path, f))]
        # filter out errors folder
        filtered_folders = list(filter(lambda foldername: "errors" not in foldername, only_folders))
        # filter out hidden folders
        filtered_folders = list(filter(lambda foldername: "." not in foldername, filtered_folders))
        # RegEx. to get only the trajectory name
        reg = r'^([\w]+-[\d]+)'
        processed_trajectories = []
        for trajectory_folder_name in filtered_folders:
            trajectory_name = re.match(reg, trajectory_folder_name).group()
            if trajectory_name not in processed_trajectories:
                processed_trajectories.append(trajectory_name)
                self.calculate_all_errors_for_trajectory(trajectory_name)