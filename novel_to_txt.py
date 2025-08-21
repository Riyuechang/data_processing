import os
import json


NOVEL_NAME = "Heru_modo_Yarikomizuki_no_gema_v01-06_epub"
#NOVEL_NAME = "Heru_modo_Yarikomizuki_no_gema_v07-08_epub"
NOVEL_PATH = f"./translation/{NOVEL_NAME}"

SAVE_DIR_PATH = f"./translation/{NOVEL_NAME}"


if not os.path.isdir(SAVE_DIR_PATH):
    os.mkdir(SAVE_DIR_PATH)

novel_file_list = [dir for dir in os.listdir(NOVEL_PATH) if dir.endswith(".json")]

for novel_file in novel_file_list:
    with open(f"{NOVEL_PATH}/{novel_file}", "r", encoding="utf-8") as file:
        dataset: list[dict[str, str]] = json.load(file)

    if not os.path.isdir(f"{SAVE_DIR_PATH}/{novel_file.rstrip('.json')}"):
        os.mkdir(f"{SAVE_DIR_PATH}/{novel_file.rstrip('.json')}")
    
    for index, chapter in enumerate(dataset):
        with open(f"{SAVE_DIR_PATH}/{novel_file.rstrip('.json')}/{index}.{chapter['title']}.txt", 'w', encoding='utf-8') as file:
            file.write(chapter["content"])