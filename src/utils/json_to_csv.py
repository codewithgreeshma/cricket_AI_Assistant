import pandas as pd
import os
from dotenv import load_dotenv
import logging
import json

load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DIRECTORY_PATH = os.getenv("DIRECTORY_PATH")
# List to store extracted data
data_list = []

for filename in os.listdir(DIRECTORY_PATH):
    logger.info(f'Processing...,{filename}')
    if filename.endswith(".json"):
        # print(filename)
        base_name = os.path.splitext(os.path.basename(filename))[0]
        # print(base_name)
        # csv_path = os.path.join(DIRECTORY_PATH, f"{base_name}.csv")

        # if not os.path.exists(csv_path):
        data = json.load(open(os.path.join(DIRECTORY_PATH, filename)))
        # with open(os.path.join(DIRECTORY_PATH, filename), encoding='utf-8') as jsonfile:
        # df = pd.read_json(jsonfile)
        # Flatten JSON using pandas json_normalize
        flattened_data = pd.json_normalize(data)
        data_list.append(flattened_data)
        # print(data_list)
        logger.info(f'Processed...,{filename}')
# Concatenate all DataFrames
df = pd.concat(data_list, ignore_index=True)
# df = pd.DataFrame(data["info"])
df.to_csv(os.path.join(DIRECTORY_PATH, f"ipl_data.csv"), encoding='utf-8', index=False)
# logger.info(f'Processed...,{filename}')
logger.info("json files converted to csv")