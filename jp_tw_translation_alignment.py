import json


NUM = 10

JP_NOVEL_NAME = f"Otonari_no_Tenshisama_ni_Itsu_v01-10_epub/お隣の天使様にいつの間にか駄目人間にされていた件_{NUM}.json"
JP_NOVEL_PATH = f"/home/ifw/Python_project/data_processing/translation/{JP_NOVEL_NAME}"

TW_NOVEL_NAME = f"關於我在無意間被隔壁的天使變成廢柴這件事_v1-10/關於我在無意間被隔壁的天使變成廢柴這件事_{NUM}.json"
TW_NOVEL_PATH = f"/home/ifw/Python_project/data_processing/epub_chapter_content/{TW_NOVEL_NAME}"

DATA_OUTPUT_PATH = f"/media/ifw/GameFile/linux_cache/data_processed/關於我在無意間被隔壁的天使變成廢柴這件事_v01-10_translation_alignment/{TW_NOVEL_NAME.split('/')[-1]}"


with open(JP_NOVEL_PATH, "r", encoding="utf-8") as file:
    jp_data: list[dict[str, str | list[dict[str, str]]]] = json.load(file)

with open(TW_NOVEL_PATH, "r", encoding="utf-8") as file:
    tw_data: list[dict[str, str]] = json.load(file)

jp_tw_alignment_list: list[dict[str, str | list]] = []
for jp_chapter, tw_chapter in zip(jp_data, tw_data):
    jp_tw_alignment_list.append(
        {
            "jp_title": jp_chapter["title"],
            "tw_title": tw_chapter["title"],
            "alignment": []
        }
    )

    tw_content_iter = iter(tw_chapter["content"].split("\n"))
    for content in jp_chapter["content"]:
        for jp_content, translation_content in zip(content["jp"].strip("\n").split("\n"), content["translation"].strip("\n").split("\n")):
            jp_tw_alignment_list[-1]["alignment"].append(
                {
                    "jp": jp_content,
                    "tw": next(tw_content_iter, ""),
                    "translation": translation_content
                }
            )
    
    for tw_content in tw_content_iter:
        jp_tw_alignment_list[-1]["alignment"].append(
            {
                "jp": "",
                "tw": tw_content,
                "translation": ""
            }
        )

with open(DATA_OUTPUT_PATH, 'w', encoding='utf-8') as file:
    json.dump(jp_tw_alignment_list, file, indent=4, ensure_ascii=False)