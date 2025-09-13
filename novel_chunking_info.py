import os
import json
from statistics import mean, median, mode

from tqdm import tqdm


#NOVEL_NAME = "test"
NOVEL_NAME = "Heru_modo_Yarikomizuki_no_gema_v01-06_epub"
NOVEL_PATH = f"./novel_chunking/{NOVEL_NAME}"

novel_file_list = [dir for dir in os.listdir(NOVEL_PATH) if dir.endswith(".json")]

tqdm_progress = tqdm(novel_file_list)
for novel_file in tqdm_progress:
    tqdm_progress.set_description(novel_file)

    with open(f"{NOVEL_PATH}/{novel_file}", "r", encoding="utf-8") as file:
        dataset: list[dict[str, str | list[str]]] = json.load(file)

    word_count_list = [(len(chunks), chunks_index, chapter_index) for chapter_index, chapter in enumerate(dataset) for chunks_index, chunks in enumerate(chapter["content"])]

    print("總字數:", sum([word_count for word_count, _, _ in word_count_list]))
    print("總塊數:", sum([len(chapter["content"]) for chapter in dataset]))

    max_word_count, chunks_index, chapter_index = max(word_count_list)
    print("最大字數:", max_word_count, "索引:", chunks_index, "章節:", dataset[chapter_index]["title"], "前16字:", [dataset[chapter_index]["content"][chunks_index][:16]])

    max_word_count, chunks_index, chapter_index = min(word_count_list)
    print("最小字數:", max_word_count, "索引:", chunks_index, "章節:", dataset[chapter_index]["title"], "前16字:", [dataset[chapter_index]["content"][chunks_index][:16]])

    print("平均數:", mean([word_count for word_count, _, _ in word_count_list]))
    print("中位數:", median([word_count for word_count, _, _ in word_count_list]))
    print("眾數:", mode([word_count for word_count, _, _ in word_count_list]))