import os
import json
from itertools import islice

from tqdm import tqdm


JP_NOVEL_NAME = f"Otonari_no_Tenshisama_ni_Itsu_v01-10_epub"
JP_NOVEL_PATH = f"/home/ifw/Python_project/data_processing/novel_chunking/{JP_NOVEL_NAME}"

TW_NOVEL_NAME = f"關於我在無意間被隔壁的天使變成廢柴這件事_v1-10"
TW_NOVEL_PATH = f"/home/ifw/Python_project/data_processing/epub_chapter_content/{TW_NOVEL_NAME}"

DATA_OUTPUT_PATH = f"/media/ifw/GameFile/linux_cache/data_processed/關於我在無意間被隔壁的天使變成廢柴這件事_v01-10_alignment"


if not os.path.isdir(DATA_OUTPUT_PATH):
    os.mkdir(DATA_OUTPUT_PATH)

jp_novel_file_list = [dir for dir in os.listdir(JP_NOVEL_PATH) if dir.endswith(".json")]
tw_novel_file_list = [dir for dir in os.listdir(TW_NOVEL_PATH) if dir.endswith(".json")]

jp_novel_file_list.sort(key=lambda a: int(a.split("_")[-1].strip(".json")))
tw_novel_file_list.sort(key=lambda a: int(a.split("_")[-1].strip(".json")))

tqdm_progress = tqdm(zip(jp_novel_file_list, tw_novel_file_list), total=len(jp_novel_file_list))
for jp_novel_file, tw_novel_file in tqdm_progress:
    tqdm_progress.set_description(jp_novel_file)

    with open(f"{JP_NOVEL_PATH}/{jp_novel_file}", "r", encoding="utf-8") as file:
        jp_data: list[dict[str, str | list[dict[str, str]]]] = json.load(file)

    with open(f"{TW_NOVEL_PATH}/{tw_novel_file}", "r", encoding="utf-8") as file:
        tw_data: list[dict[str, str]] = json.load(file)

    jp_tw_alignment_list: list[dict[str, str | list]] = []
    for jp_chapter, tw_chapter in zip(jp_data, tw_data):
        jp_tw_alignment_list.append(
            {
                "jp_tw_count_alignment": None,
                "jp_title": jp_chapter["title"],
                "tw_title": tw_chapter["title"],
                "alignment": []
            }
        )

        tw_content_iter = iter(tw_chapter["content"].strip("\n").split("\n"))
        for jp_content in jp_chapter["content"]:
            jp_content_clean = jp_content.strip("\n")

            jp_tw_chunk = {
                "jp": jp_content_clean,
                "tw": "\n".join(islice(tw_content_iter, jp_content_clean.count("\n") + 1))
            }

            jp_tw_alignment_list[-1]["alignment"].append(jp_tw_chunk)
        
        tw_content_iter_remaining = list(tw_content_iter)

        if tw_content_iter_remaining:
            jp_tw_alignment_list[-1]["alignment"][-1]["tw"] += "\n" + "\n".join(tw_content_iter_remaining)
        
        jp_len = jp_tw_alignment_list[-1]["alignment"][-1]["jp"].count("\n")
        tw_len = jp_tw_alignment_list[-1]["alignment"][-1]["tw"].count("\n")

        jp_tw_alignment_list[-1]["jp_tw_count_alignment"] = (jp_len == tw_len)

    all_jp_tw_count_alignment_true = all([chapter["jp_tw_count_alignment"] for chapter in jp_tw_alignment_list])
    not_pass_tag = "" if all_jp_tw_count_alignment_true else "not_pass_"

    with open(f"{DATA_OUTPUT_PATH}/{not_pass_tag}{tw_novel_file}", 'w', encoding='utf-8') as file:
        json.dump(jp_tw_alignment_list, file, indent=4, ensure_ascii=False)