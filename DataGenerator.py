from os import listdir, system
from os.path import isfile, join
import logging


class DataGenerator:
    def __init__(self, trajectories_folder_path: str, opendihu_release_folder_path: str,
                 data_output_folder_path: str, settings_multiple_fibers_path: str = "../settings_multiple_fibers.py"):
        self.trajectories_folder_path = trajectories_folder_path
        self.opendihu_release_folder_path = opendihu_release_folder_path
        self.data_output_folder_path = data_output_folder_path
        self.settings_multiple_fibers_path = settings_multiple_fibers_path
        

    def generate(self):
        only_files = [f for f in listdir(self.trajectories_folder_path) if
                      isfile(join(self.trajectories_folder_path, f))]
        filter_name = '.py'
        only_py_files = list(filter(lambda filename: filter_name in filename, only_files))
        for file in only_py_files:
            self._run_opendihu_on(file)
        logging.info("Trainings data generated")

    def _run_opendihu_on(self, trajectory):
        trajectory_name = trajectory[:-3]
        for nt_0D in [2, 4, 8, 16, 32]:
            for nt_1D in [1, 2, 4, 8, 16, 32]:
                command = f'cd {self.opendihu_release_folder_path}; \
                    ./multiple_fibers {self.settings_multiple_fibers_path} \
                    --tend 6 --dt_splitting 3e-3 --nt_0D {nt_0D} --nt_1D {nt_1D} \
                    --output_interval_0D {64 * nt_0D} \
                    --disable_firing 0.0 \
                    --initial_value_file {join(self.trajectories_folder_path,trajectory)}; \
                    mkdir {self.data_output_folder_path}/{trajectory_name}-{nt_0D}-{nt_1D}; \
                    mv {self.opendihu_release_folder_path}/out/* {self.data_output_folder_path}/{trajectory_name}-{nt_0D}-{nt_1D}'
                assert (system(command) == 0)
        logging.info("done with trajectory", trajectory)