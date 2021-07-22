import tensorflow as tf
import logging
from os.path import join
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from DataLoader import DataLoader


class Engine:
    def __init__(self, model, data_loader: DataLoader, gpu_memory_limit: int = 8192):
        self.model = model
        self.data_loader = data_loader
        self.gpu_memory_limit = gpu_memory_limit

    def enable_gpu(self):
        """Limiting GPU memory growth. See: https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # Restrict TensorFlow to only allocate 8GB of memory on the first GPU
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=self.gpu_memory_limit)])
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                logging.info(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                logging.warning(e)


class ModelEvaluator(Engine):
    def __init__(self, model, data_loader: DataLoader, gpu_memory_limit: int = 8192):
        super().__init__(model, data_loader, gpu_memory_limit)

    def evaluate(self):
        self.enable_gpu()
        x, y = self.data_loader.get_data()
        self.model.evaluate(x, y)


class ModelTrainer(Engine):
    def __init__(self, model, data_loader: DataLoader, model_checkpoint_folder_path: str = None,
                 csv_logging_file_path: str = None, gpu_memory_limit: int = 8192):
        super().__init__(model, data_loader, gpu_memory_limit)
        self.model_checkpoint_folder_path = model_checkpoint_folder_path
        self.csv_logging_file_path = csv_logging_file_path

    def train(self, epochs: int, seed: int):
        self.enable_gpu()
        callbacks = self._get_callbacks()
        x_train, x_validation, y_train, y_validation = self.data_loader.get_training_and_validation_data(seed)
        self.model.fit(x=x_train, y=y_train, epochs=epochs, callbacks=callbacks,
                       validation_data=(x_validation, y_validation), use_multiprocessing=False)

    def _get_callbacks(self):
        callbacks = []
        if self.model_checkpoint_folder_path:
            checkpoint_filepath = join(self.model_checkpoint_folder_path, 'model.{epoch:02d}-{loss:.4f}.tf')
            model_checkpoint_callback = ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_weights_only=False,
                monitor='loss',
                mode='min',
                save_best_only=True)
            callbacks.append(model_checkpoint_callback)
        if self.csv_logging_file_path:
            csv_logger = CSVLogger(self.csv_logging_file_path, append=True, separator=';')
            callbacks.append(csv_logger)

        if len(callbacks) == 0:
            return None
        return callbacks

    def save_model(self, destination: str):
        """
        Convenience method to save the model.
        :param destination: Path where the model should be saved. Must end with .tf
        """
        self.model.save(destination, overwrite=True, include_optimizer=True, save_format='h5')
        