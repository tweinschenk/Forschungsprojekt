import pickle
import numpy as np
from random import Random
from os.path import join


class TrajectoryGenerator:
    def __init__(self, seed):
        self.random = Random(seed)

    def generate_trajectory(self, file):
        data = file['data']
        channels = data[1]['components']

        # assign random values to be passed to manipulator functions
        a_sin1_channel_0 = self.random.uniform(-1.5, 1.5)
        b_sin1_channel_0 = self.random.randrange(1, 100, 1)
        a_sin2_channel_0 = self.random.uniform(-1.5, 1.5)
        b_sin2_channel_0 = self.random.randrange(1, 100, 1)

        a_sin1_channel_1 = self.random.uniform(-1 / 100, 1 / 100)
        b_sin1_channel_1 = self.random.randrange(1, 100, 1)
        a_sin2_channel_1 = self.random.uniform(-1 / 100, 1 / 100)
        b_sin2_channel_1 = self.random.randrange(1, 100, 1)

        a_sin1_channel_2 = self.random.uniform(-1 / 200, 1 / 200)
        b_sin1_channel_2 = self.random.randrange(1, 100, 1)
        a_sin2_channel_2 = self.random.uniform(-1 / 200, 1 / 200)
        b_sin2_channel_2 = self.random.randrange(1, 100, 1)

        a_sin1_channel_3 = self.random.uniform(-1 / 200, 1 / 200)
        b_sin1_channel_3 = self.random.randrange(1, 100, 1)
        a_sin2_channel_3 = self.random.uniform(-1 / 200, 1 / 200)
        b_sin2_channel_3 = self.random.randrange(1, 100, 1)

        move_direction_right = bool(self.random.getrandbits(1))
        move_steps = self.random.randrange(1, 101)

        scaler0 = self.random.uniform(0.9, 1.1)
        scaler1 = self.random.uniform(0.9, 1.1)
        scaler2 = self.random.uniform(0.9, 1.1)
        scaler3 = self.random.uniform(0.9, 1.1)

        # manipulate
        channels[0]['values'] = self.add_sin(channels[0]['values'], a_sin1_channel_0, b_sin1_channel_0)
        channels[0]['values'] = self.add_sin(channels[0]['values'], a_sin2_channel_0, b_sin2_channel_0)

        channels[1]['values'] = self.add_sin(channels[1]['values'], a_sin1_channel_1, b_sin1_channel_1)
        channels[1]['values'] = self.add_sin(channels[1]['values'], a_sin2_channel_1, b_sin2_channel_1)

        channels[2]['values'] = self.add_sin(channels[2]['values'], a_sin1_channel_2, b_sin1_channel_2)
        channels[2]['values'] = self.add_sin(channels[2]['values'], a_sin2_channel_2, b_sin2_channel_2)

        channels[3]['values'] = self.add_sin(channels[3]['values'], a_sin1_channel_3, b_sin1_channel_3)
        channels[3]['values'] = self.add_sin(channels[3]['values'], a_sin2_channel_3, b_sin2_channel_3)

        if move_direction_right:
            channels = self.move_right(channels, move_steps)
        else:
            channels = self.move_left(channels, move_steps)

        channels[0]['values'] = self.scale(channels[0]['values'], scaler0)
        channels[1]['values'] = self.scale(channels[1]['values'], scaler1)
        channels[2]['values'] = self.scale(channels[2]['values'], scaler2)
        channels[3]['values'] = self.scale(channels[3]['values'], scaler3)

        return channels

    def generate_trajectories(self, amount, initial_value_path, output_folder_path):
        for i in range(amount):
            file = np.load(initial_value_path, allow_pickle=True)
            self.generate_trajectory(file)
            trajectory_name = f"RandomTrajectory-{i}.py"
            output_path = join(output_folder_path, trajectory_name)
            self.save_file_as_py(file, output_path)

    @staticmethod
    def save_file_as_py(file, path):
        with open(path, 'wb') as f:
            pickle.dump(file, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def add_sin(channel, a=1, b=1, c=0, d=0):
        normalizer = (2 * np.pi) / len(channel)
        for i in range(len(channel)):
            channel[i] += a * np.sin(b * (normalizer * i + c)) + d
        return channel

    @staticmethod
    def move_right(channels, steps=1):
        assert steps > 0
        results = channels
        for i, channel in enumerate(channels):
            left_side = np.full(steps, channel['values'][0])
            right_side = channel['values'][:len(channel['values']) - steps]
            results[i]['values'] = np.append(left_side, right_side)
            assert len(channel['values']) == len(results[i]['values'])
        return results

    @staticmethod
    def move_left(channels, steps=1):
        assert steps > 0
        results = channels
        for i, channel in enumerate(channels):
            right_side = np.full(steps, channel['values'][-1])
            left_side = channel['values'][steps:]
            results[i]['values'] = np.append(left_side, right_side)
            assert len(channel['values']) == len(results[i]['values'])
        return results

    @staticmethod
    def scale(channel, scaler=1):
        for i in range(len(channel)):
            channel[i] *= scaler
        return channel
