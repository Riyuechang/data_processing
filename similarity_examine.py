import json


NUM = 1

MAX_JP_VALUE = 0.15
MAX_TW_VALUE = 0.75
MAX_SUB_JP_VALUE = 0.05
MAX_SUB_TW_VALUE = 0.95

FIRST_MATCH_WORDS = ["※", "*", "【", "】"]
SECOND_TOP_MATCH_WORDS = [":", "："]
SECOND_MATCH_WORDS = ["㊟", "注", "编", "原", "译", "这句", "这条", "这行", "这段", "这边", "这个", "这里", "此处", "日文", "ps", "PS", "Ps", "pS", "(", "（", ")", "）"] # 第 卷 话

NOVEL_NAME = f"關於我在無意間被隔壁的天使變成廢柴這件事_v01-10_similarity/關於我在無意間被隔壁的天使變成廢柴這件事_{NUM}.json"
JP_TW_NOVEL_SIMILARITY_PATH = f"/media/ifw/GameFile/linux_cache/data_processed/{NOVEL_NAME}"

DATA_OUTPUT_PATH = f"/media/ifw/GameFile/linux_cache/data_processed/關於我在無意間被隔壁的天使變成廢柴這件事_v01-10_examine/{NOVEL_NAME.split('/')[-1]}"

with open(JP_TW_NOVEL_SIMILARITY_PATH, "r", encoding="utf-8") as file:
    dataset: list[dict[str, str]] = json.load(file)

for chapter in dataset:
    for alignment in chapter["alignment"]:
        if any(word in alignment["tw"] for word in FIRST_MATCH_WORDS) or (
            any(word in alignment["tw"] for word in SECOND_TOP_MATCH_WORDS) 
            and any(word in alignment["tw"] for word in SECOND_MATCH_WORDS)
        ):
            alignment["examine"] = "True_匹配"
            continue

        if ((alignment["jp"][-1] != alignment["tw"][-1] 
             or alignment["translation"][-1] != alignment["tw"][-1]
            )and (
                alignment["similarity_to_jp"] > MAX_SUB_JP_VALUE
                or alignment["similarity_translation"] < MAX_SUB_TW_VALUE
            )
        ):
            alignment["examine"] = "True_尾字不同"
            continue

        if (alignment["similarity_to_jp"] > MAX_JP_VALUE 
            or alignment["similarity_translation"] < MAX_TW_VALUE
        ):
            alignment["examine"] = "True_小於"
            continue

        alignment["examine"] = "False"

with open(DATA_OUTPUT_PATH, 'w', encoding='utf-8') as file:
    json.dump(dataset, file, indent=4, ensure_ascii=False)