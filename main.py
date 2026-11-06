import os
from src.optimized_dataloader import clean_and_save_data

# this is to avoid running the clean_and_save_data() function if the processed data already exists
if not os.path.exists(os.path.join("data", "processed")):
    clean_and_save_data()

