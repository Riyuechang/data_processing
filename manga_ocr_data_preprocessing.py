import os
import json

import pandas as pd
from tqdm import tqdm


MANGA_OCR_DATA_PATH = "/media/ifw/GameFile/linux_cache/data_unprocessed/manga_image_ocr"

SAVE_PATH = f"{MANGA_OCR_DATA_PATH}.parquet"


ocr_data_path_list: list[str] = []

for path, folders, files in os.walk(MANGA_OCR_DATA_PATH):
    if not files:
        continue

    ocr_data_path_list.extend([
        f"{path}/{file}"
        for file in files
        if file.endswith(".json")
    ])

ocr_rec_data_list = []

for index, ocr_data_path in tqdm(enumerate(ocr_data_path_list), total=len(ocr_data_path_list)):
    with open(ocr_data_path, "r", encoding="utf-8") as file:
        dataset: dict[str, str | list] = json.load(file)
    
    ocr_rec_data_list.append({
        "image_path": ocr_data_path.lstrip(f"{MANGA_OCR_DATA_PATH}/"),
        "rec_texts": dataset["rec_texts"], 
        "rec_scores": dataset["rec_scores"], 
        "rec_polys": dataset["rec_polys"]
    })

ocr_rec_data = pd.DataFrame(ocr_rec_data_list)
ocr_rec_data.to_parquet(SAVE_PATH)