import json

import numpy as np
import pandas as pd
from tqdm import tqdm


EXTRACTION_LENGTH = 8

DROP_KEYWORDS = [
    "拷貝", "拷贝", 
    "拷貝漫畫", "拷贝漫画",
    "拷貝漫画", "拷贝漫畫",

    "DMS", "dms",
    "DM5", "dm5",

    "ZEROBYW", "zerobyw",
    "搬運網", "搬运网",
    "運網", "运网",

    "Scan ", "scan ",
    "墮落的猴子",
    "堕落的猴子",

    "团子", "汉化", "漢化",
    "嵌字", "嵌字",
    "校對", "校对",
    "圖源", "图源",
    "合購", "合购"
]

MANGA_OCR_DATA_PATH = "/media/ifw/GameFile/linux_cache/data_unprocessed/manga_image_ocr.parquet"

SAVE_PATH = "/media/ifw/GameFile/linux_cache/data_unprocessed/manga_image_ocr_keywords.json"


def array_to_list(array):
    if isinstance(array, np.ndarray):
        return [array_to_list(i) for i in array.tolist()]

    if isinstance(array, (list, tuple)):
        return [array_to_list(i) for i in array]

    return array


dataset = pd.read_parquet(MANGA_OCR_DATA_PATH)

dataset_keywords: dict[str, list[str]] = {}
for data in tqdm(dataset.itertuples(), total=len(dataset), desc="duplicated"):
    data_text: str = "".join(array_to_list(data.rec_texts))

    for keyword in DROP_KEYWORDS:
        keyword_index = data_text.find(keyword)

        if keyword_index == -1:
            continue

        if keyword not in dataset_keywords:
            dataset_keywords[keyword] = []

        extraction_length_min_index = max(keyword_index - EXTRACTION_LENGTH, 0)
        extraction_length_max_index = keyword_index + EXTRACTION_LENGTH
        extraction_text = data_text[extraction_length_min_index:extraction_length_max_index]

        dataset_keywords[keyword].append(
            f"{data.image_path}___{extraction_text}"
        )

keywords_sort = sorted([
    (len(keyword), keyword) 
    for keyword in dataset_keywords.keys()
], reverse=True)

new_dataset_keywords: dict[str, list[str]] = {}
duplicated = set()
for _, keyword in keywords_sort:
    new_dataset_keywords[keyword] = []

    for text in dataset_keywords[keyword]:
        image_path = text.split("___")[0]

        if image_path in duplicated:
            continue

        duplicated.add(image_path)
        new_dataset_keywords[keyword].append(text)

for keyword in new_dataset_keywords.keys():
    new_dataset_keywords[keyword].sort()

total_count = 0
for keyword in new_dataset_keywords.keys():
    print(f"{keyword}_count:", len(new_dataset_keywords[keyword]))
    total_count += len(new_dataset_keywords[keyword])

print("total_count:", total_count)

"""book_info = {}
for value in dataset_keywords.values():
    for  in value:"""

with open(SAVE_PATH, 'w', encoding='utf-8') as file:
    json.dump(new_dataset_keywords, file, indent=4, ensure_ascii=False)