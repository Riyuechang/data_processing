import json


NOVEL_NAME = "關於我在無意間被隔壁的天使變成廢柴這件事_6"
NOVEL_PATH = f"/media/ifw/GameFile/linux_cache/data_unprocessed/jp_tw_novel_alignment/{NOVEL_NAME}"

DATA_OUTPUT_PATH = f"/media/ifw/GameFile/linux_cache/data_processed/jp_tw_novel_alignment_clean/{NOVEL_NAME}_clean.json"


with open(f"{NOVEL_PATH}/jp.json", "r", encoding="utf-8") as file:
    jp_data: list[dict[str, str]] = json.load(file)

with open(f"{NOVEL_PATH}/tw.json", "r", encoding="utf-8") as file:
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
    for jp_content in jp_chapter["content"].split("\n"):
        jp_tw_alignment_list[-1]["alignment"].append(
            {
                "jp": jp_content,
                "tw": next(tw_content_iter, "")
            }
        )
    
    for tw_content in tw_content_iter:
        jp_tw_alignment_list[-1]["alignment"].append(
            {
                "jp": "",
                "tw": tw_content
            }
        )

with open(DATA_OUTPUT_PATH, 'w', encoding='utf-8') as file:
    json.dump(jp_tw_alignment_list, file, indent=4, ensure_ascii=False)