import json


MAX_JP_VALUE = 0.2
MAX_TW_VALUE = 0.6
MAX_SUB_JP_VALUE = 0.05
MAX_SUB_TW_VALUE = 0.9

FIRST_MATCH_WORDS = ["※", "*", "【", "】"]
SECOND_TOP_MATCH_WORDS = [":", "："]
SECOND_MATCH_WORDS = ["注", "原", "译", "这句", "这条", "这行", "这段", "这边", "这个", "这里", "此处", "日文", "ps", "PS", "Ps", "pS", "(", "（", ")", "）"] # 第 卷 话

NOVEL_NAME = "關於我在無意間被隔壁的天使變成廢柴這件事_5"
JP_TW_NOVEL_SIMILARITY_PATH = f"/media/ifw/GameFile/linux_cache/data_processed/jp_tw_novel_similarity/{NOVEL_NAME}_similarity.json"

DATA_OUTPUT_PATH = f"/media/ifw/GameFile/linux_cache/data_processed/jp_tw_novel_examine/{NOVEL_NAME}_examine.json"


with open(JP_TW_NOVEL_SIMILARITY_PATH, "r", encoding="utf-8") as file:
    dataset: list[dict[str, str]] = json.load(file)

for chapter in dataset:
    for alignment in chapter["alignment"]:
        if any(word in alignment["tw"] for word in FIRST_MATCH_WORDS) or (
            any(word in alignment["tw"] for word in SECOND_TOP_MATCH_WORDS) 
            and any(word in alignment["tw"] for word in SECOND_MATCH_WORDS)
        ):
            alignment["examine"] = True
            continue

        if (alignment["jp"][-1] != alignment["tw"][-1] 
            and (
                alignment["similarity_to_jp"] > MAX_SUB_JP_VALUE
                or alignment["similarity_translation"] < MAX_SUB_TW_VALUE
            )
        ):
            alignment["examine"] = True
            continue

        if (alignment["similarity_to_jp"] > MAX_JP_VALUE 
            or alignment["similarity_translation"] < MAX_TW_VALUE
        ):
            alignment["examine"] = True
            continue

        alignment["examine"] = False

with open(DATA_OUTPUT_PATH, 'w', encoding='utf-8') as file:
    json.dump(dataset, file, indent=4, ensure_ascii=False)