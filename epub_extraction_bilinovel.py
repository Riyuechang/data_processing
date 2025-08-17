import os
import re
import json

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from tqdm import tqdm


def extract_content(ebook: epub.EpubBook) -> list[dict[str,str]]:
    ebook_content = list(ebook.get_items_of_type(ebooklib.ITEM_DOCUMENT))

    contents: list[dict[str, str]] = []
    for toc, content in zip(ebook.toc, ebook_content):
        soup = BeautifulSoup(content.get_content().decode("utf-8"), 'html.parser')

        contents.append({
            "title": toc.title,
            "content": re.sub(
                pattern=r"\n+",
                repl="\n",
                string=soup.body.get_text().strip()
            )
        })
            
    return contents


if __name__ == "__main__":
    """
    #EBOOK_NAME = "无职转生 ～到了异世界就拿出真本事～/无职转生 ～到了异世界就拿出真本事～ 第一卷 幼年期.epub"
    EBOOK_NAME = "关于我在无意间被隔壁的天使变成废柴这件事/关于我在无意间被隔壁的天使变成废柴这件事 第一卷.epub"
    EBOOK_PATH = f"/home/ifw/bilinovel_download/{EBOOK_NAME}"


    ebook = epub.read_epub(EBOOK_PATH)

    data = extract_content(ebook)

    with open("test_data/bilinovel_test_2.json", 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
    """

    #EBOOK_NAME = "关于我在无意间被隔壁的天使变成废柴这件事"
    EBOOK_NAME = "无职转生 ～到了异世界就拿出真本事～"
    EBOOK_PATH = f"/home/ifw/bilinovel_download/{EBOOK_NAME}"
    SAVE_DIR_PATH = f"./epub_chapter_content/{EBOOK_NAME}"

    if not os.path.isdir(SAVE_DIR_PATH):
        os.mkdir(SAVE_DIR_PATH)

    epub_file_list = [dir for dir in os.listdir(EBOOK_PATH) if dir.endswith(".epub")]

    tqdm_progress = tqdm(epub_file_list)
    for epub_file in tqdm_progress:
        tqdm_progress.set_description(epub_file)

        ebook = epub.read_epub(f"{EBOOK_PATH}/{epub_file}")
        data = extract_content(ebook)

        with open(f"{SAVE_DIR_PATH}/{epub_file.rstrip('.epub')}.json", 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)