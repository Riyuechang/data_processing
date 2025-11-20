import opencc
import numpy as np
import pandas as pd
from tqdm import tqdm


MANGA_OCR_DATA_PATH = "/media/ifw/GameFile/linux_cache/data_unprocessed/manga_image_ocr.parquet"


def array_to_list(array):
    if isinstance(array, np.ndarray):
        return [array_to_list(i) for i in array.tolist()]

    if isinstance(array, (list, tuple)):
        return [array_to_list(i) for i in array]

    return array


opencc_converter = opencc.OpenCC('s2tw.json')

dataset = pd.read_parquet(MANGA_OCR_DATA_PATH)

word_set = set()
dataset_duplicated: dict[str, list[str]] = {}
dataset_duplicated_simplified_chinese = []
for data in tqdm(dataset.itertuples(), total=len(dataset), desc="duplicated"):
    data_text = "".join(array_to_list(data.rec_texts))
    
    for text in "".join(data_text):
        word_set.add(text)
    
    if data_text in dataset_duplicated:
        dataset_duplicated[data_text].append(data.image_path)

        if opencc_converter.convert(data_text) != data_text:
            dataset_duplicated_simplified_chinese.append(data.image_path)

        continue

    dataset_duplicated[data_text] = [data.image_path]

duplicated = [
    duplicated 
    for duplicated in dataset_duplicated.values()
    if len(duplicated) > 1
]

dataset_score_not_enough: dict[str, int | list] = {
    "all_score_count": 0,
    "not_enough": 0,
    "image_path": []
}
for data in tqdm(dataset.itertuples(), total=len(dataset), desc="score_not_enough"):
    data_rec_scores = array_to_list(data.rec_scores)
    data_score = [
        score 
        for score in data_rec_scores
        if score < 0.8
    ]

    dataset_score_not_enough["all_score_count"] += len(data_rec_scores)
    dataset_score_not_enough["not_enough"] += len(data_score)
    
    if data_score:
        dataset_score_not_enough["image_path"].append(data.image_path)

dataset_simplified_chinese: dict[str, int | list] = {
    "all_text_count": 0,
    "simplified_chinese": [],
    "image_path": []
}
for data in tqdm(dataset.itertuples(), total=len(dataset), desc="simplified_chinese"):
    data_rec_texts = array_to_list(data.rec_texts)
    data_simplified_chinese = [
        text 
        for text in data_rec_texts
        if opencc_converter.convert(text) != text
    ]

    dataset_simplified_chinese["all_text_count"] += len(data_rec_texts)
    dataset_simplified_chinese["simplified_chinese"].extend(data_simplified_chinese)

    if data_simplified_chinese:
        dataset_simplified_chinese["image_path"].append(data.image_path)

print("duplicated_count:", len(duplicated))
print("duplicated_sum:", sum([len(data) for data in duplicated]))
print("duplicated_simplified_chinese_count:", len(dataset_duplicated_simplified_chinese))
print("score_not_enough_count:", len(dataset_score_not_enough["image_path"]))
print("all_score_count:", dataset_score_not_enough["all_score_count"])
print("not_enough:", dataset_score_not_enough["not_enough"])
print("simplified_chinese_count:", len(dataset_simplified_chinese["image_path"]))
print("all_text_count:", dataset_simplified_chinese["all_text_count"])
print("simplified_chinese:", len(dataset_simplified_chinese["simplified_chinese"]))
print("simplified_chinese:", dataset_simplified_chinese["simplified_chinese"][:10])
print("word_set_count", len(word_set))
print("word_set_count", list(word_set)[:10])