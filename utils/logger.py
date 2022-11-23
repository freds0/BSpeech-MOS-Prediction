import logging
from os.path import join

class logger():
    @staticmethod
    def config(folder=None):
        if not (folder is None):
            filepath = join(folder, 'training.log')
            logging.basicConfig(filename=filepath, level=logging.INFO)

    @staticmethod
    def info(message):
        print(message)
        logging.info(message)

    @staticmethod
    def error(message):
        print(message)
        logging.error(message)