import os
import re
import json
from collections import defaultdict
from multiprocessing import Pool, cpu_count

from tqdm import tqdm


MIN_LEN = 5
MAX_LEN = 64
REPEATED_COUNT = 3

EBOOK_NAME = "[北山結莉] 精霊幻想記 第27巻 ep"

TRANSLATION_PATH = f"/home/ifw/Python_project/data_processing/translation/{EBOOK_NAME}"


def find_repeated_content(
    content: str, 
    min_len: int = MIN_LEN, 
    max_len: int = MAX_LEN, 
    repeated_count: int = REPEATED_COUNT
):
    match_dict: dict[str, int] = {}

    index = 0
    while index < len(content):
        match_count = 0

        for pattern_len in range(min_len, max_len + 1):
            if index + pattern_len > len(content):
                return match_dict

            pattern = content[index:index + pattern_len]
            match_text = re.findall(pattern, content)

            if not match_text and not match_count:
                break

            if len(match_text) >= match_count:
                match_count = len(match_text)
                continue

            if match_count >= repeated_count:
                match_dict[content[index:index + pattern_len - 1]] = max(
                    match_count, 
                    match_dict.get(content[index:index + pattern_len - 1], 0)
                )

            index += pattern_len

            break
        else:
            index += 1

    return match_dict

def find_repeated_ngrams(content: str, n: int, repeated_count: int = 2):
    ngrams = defaultdict(list)

    for i in range(len(content) - n + 1):
        gram = content[i:i+n]
        ngrams[gram].append(i)

    return {gram: len(pos) for gram, pos in ngrams.items() if len(pos) >= repeated_count}


if __name__ == "__main__":
    translation_file_list = [dir for dir in os.listdir(TRANSLATION_PATH) if dir.endswith(".json")]

    repeated_content_dict = {}

    tqdm_progress = tqdm(translation_file_list)
    for translation_file in tqdm_progress:
        tqdm_progress.set_description(translation_file)
        with open(f"{TRANSLATION_PATH}/{translation_file}", "r", encoding="utf-8") as file:
            dataset: list[dict[str, str | list[dict[str, str]]]] = json.load(file)

        contents = []
        for data in dataset:
            for content in data["content"]:
                contents.append(content["translation"])
        
        with Pool(cpu_count()) as pool:
            response = list(tqdm(
                pool.imap(
                    find_repeated_content, 
                    contents
                ), 
                desc=f"cpu_count: {cpu_count()}",
                total=len(contents)
            ))
        
        for data in response:
            repeated_content_dict.update(data)

        #for content in tqdm(contents):
        #    repeated_content = find_repeated_content(content, MIN_LEN, MAX_LEN, REPEATED_COUNT)
        #    repeated_content_dict.update(repeated_content)
    
    repeated_content_sort = sorted([repeated_content for repeated_content in repeated_content_dict.items()], key=lambda x: x[1])

    print(repeated_content_sort)
