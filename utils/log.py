import logging
import yaml
from pprint import pformat

def log_config(config):
    logging.info("==== Configurations ====")
    logging.info(pformat(config.__dict__)) 
    logging.info("========================")
    
    
def setup_logger(log_path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )