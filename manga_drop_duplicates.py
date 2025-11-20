import json

import numpy as np
import pandas as pd
from tqdm import tqdm


MANGA_OCR_DATA_PATH = "/media/ifw/GameFile/linux_cache/data_unprocessed/manga_image_ocr.parquet"

SAVE_PATH = "/media/ifw/GameFile/linux_cache/data_unprocessed/manga_image_ocr_duplicated.json"


def array_to_list(array):
    if isinstance(array, np.ndarray):
        return [array_to_list(i) for i in array.tolist()]

    if isinstance(array, (list, tuple)):
        return [array_to_list(i) for i in array]

    return array


dataset = pd.read_parquet(MANGA_OCR_DATA_PATH)

dataset_duplicated: dict[str, list[str]] = {}
for data in tqdm(dataset.itertuples(), total=len(dataset), desc="duplicated"):
    data_text = "".join(array_to_list(data.rec_texts))
    
    if data_text in dataset_duplicated:
        dataset_duplicated[data_text].append(data.image_path)
        continue

    dataset_duplicated[data_text] = [data.image_path]

duplicated_data = {
    key: duplicated 
    for key, duplicated in dataset_duplicated.items()
    if len(duplicated) > 1
}

with open(SAVE_PATH, 'w', encoding='utf-8') as file:
    json.dump(duplicated_data, file, indent=4, ensure_ascii=False)