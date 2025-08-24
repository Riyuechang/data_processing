import os
import json

from tqdm import tqdm


NOVEL_NAME = "test"
#NOVEL_NAME = "Heru_modo_Yarikomizuki_no_gema_v01-06_epub"
#NOVEL_NAME = "Heru_modo_Yarikomizuki_no_gema_v07-08_epub"
NOVEL_PATH = f"./translation/{NOVEL_NAME}"


novel_translation_file_list = [dir for dir in os.listdir(NOVEL_PATH) if dir.endswith(".json")]

tqdm_progress = tqdm(novel_translation_file_list)
for novel_translation_file in tqdm_progress:
    with open(f"{NOVEL_PATH}/{novel_translation_file}", "r", encoding="utf-8") as file:
        translation_data: list[dict[str, str | list[dict[str, str]]]] = json.load(file)
    
    count = 0
    for chapter in  translation_data:
        for content_index in range(len(chapter["content"])):
            jp_seg = chapter["content"][content_index]["jp"].strip("\n").split("\n")
            translation_seg = chapter["content"][content_index]["translation"].strip("\n").split("\n")

            if len(jp_seg) != len(translation_seg):
                print(f"格式錯誤：{chapter['title']}", f"索引：{content_index}", [chapter["content"][content_index]["jp"][-1]], jp_seg[0])

                if chapter["content"][content_index]["jp"][-1] == "\n":
                    count += 1

    print(count)