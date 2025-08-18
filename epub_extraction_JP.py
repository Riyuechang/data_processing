import os
import re
import json

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup, element
from tqdm import tqdm


def find_title_id(page_element: element.PageElement) -> str | None:
    if (type(page_element) is element.NavigableString
         or type(page_element) is element.Comment
         or type(page_element) is element.RubyTextString
        ):
        return None

    title_id = page_element.get("id")

    if title_id:
        return title_id

    page_element_contents = page_element.contents.copy()
    page_element_contents.reverse()
    for content in page_element_contents:
        title_id = find_title_id(content)
        
        if title_id:
            return title_id
    
    return None

def only_newline(content: str):
    for word in content:
        if word != "\n":
            return False
    
    return True

def only_numbers(text: str):
    numbers = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", "-"]

    number = ""
    for word in text:
        if word == "-" and number:
            continue

        if word == "." and "." in number:
            continue

        if word in numbers:
            number += word

    if number and number not in [".", "-", "-."]:
        return float(number)

def extract_content(ebook: epub.EpubBook) -> list[dict[str,str]]:
    title_data_list: list[dict[str, str | None]] = []
    for toc in ebook.toc:
        toc_href_split = toc.href.split("#")

        if not only_numbers(toc_href_split[0]):
            continue

        title_data_list.append(
            {
                "title": toc.title, 
                "title_id": None if len(toc_href_split) < 2 else toc_href_split[1], 
                "file_path": toc_href_split[0]
            }
        )
    
    ebook_content = [content for content in list(ebook.get_items_of_type(ebooklib.ITEM_DOCUMENT)) if only_numbers(content.file_name)]

    if not any(title_data["title_id"] for title_data in title_data_list):
        contents: list[dict[str, str]] = []

        toc_iterable = iter(ebook.toc)
        toc = next(toc_iterable)

        for content in ebook_content:
            if toc.href == content.file_name or abs(only_numbers(toc.href)) < abs(only_numbers(content.file_name)):
                contents.append({
                    "title": toc.title,
                    "content": ""
                })
                toc = next(toc_iterable, None)

            soup = BeautifulSoup(content.get_content().decode("utf-8"), 'html.parser')
            contents[-1]["content"] += f"\n{soup.body.get_text()}\n"

        for content in contents:
            content["content"] = re.sub(
                pattern=r"\n+",
                repl="\n",
                string=content["content"].strip()
            )

        return contents

    chapters: list[dict[str, str]] = []
    for content in ebook_content:
        soup = BeautifulSoup(content.get_content().decode("utf-8"), 'html.parser')

        chapter_content: dict[str, str] = {"file_path": content.file_name, "title_id": "", "content": ""}
        for page_element in soup.body.div.contents if soup.body.div else soup.body.contents:
            title_id = find_title_id(page_element)
            
            if title_id:
                chapters.append(chapter_content)
                chapter_content = {
                    "file_path": content.file_name, 
                    "title_id": title_id if [True for id in title_data_list if id["title_id"] == title_id] else "", 
                    "content": ""
                }

            chapter_content["content"] += f"\n{page_element.get_text()}\n"

        if chapter_content["content"]:
            chapters.append(chapter_content)

    chapters_clean = [chapter for chapter in chapters if not only_newline(chapter["content"]) or chapter["title_id"]]

    chapters_iterable = iter(chapters_clean)
    chapter_data = next(chapters_iterable)

    contents: list[dict[str, str]] = []
    for title_data in title_data_list:
        content_data: dict[str, str] = {
            "title": title_data["title"],
            "content": ""
        }

        while True:
            if not chapter_data:
                break

            if abs(only_numbers(chapter_data["file_path"])) < abs(only_numbers(title_data["file_path"])):
                chapter_data = next(chapters_iterable, None)
                continue

            if chapter_data["title_id"] == "" or chapter_data["title_id"] == title_data["title_id"]:
                content_data["content"] += chapter_data["content"]
                chapter_data = next(chapters_iterable, None)
            else:
                break
        
        content_data["content"] = re.sub(
            pattern=r"\n+",
            repl="\n",
            string=content_data["content"].strip()
        )
        contents.append(content_data)
            
    return contents


if __name__ == "__main__":
    """
    EBOOK_NAME = "Otonari_no_Tenshisama_ni_Itsu_v01-08_epub/[佐伯さん,はねこと]お隣の天使様にいつの間にか駄目人間にされていた件 (GA文庫)(SBクリエイティブ株式会社)(2019).epub"
    #EBOOK_NAME = "Mushoku_Isekai_Dasu_v01-26_epub/無職転生 ～異世界行ったら本気だす～ 01 (MFブックス).epub"
    EBOOK_PATH = f"/home/ifw/epub/{EBOOK_NAME}"


    ebook = epub.read_epub(EBOOK_PATH)

    data = extract_content(ebook)

    with open("test_data/test_2.json", 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
    """

    EBOOK_NAME = "Heru_modo_Yarikomizuki_no_gema_v01-06_epub"
    #EBOOK_NAME = "test"
    #EBOOK_NAME = "Otonari_no_Tenshisama_ni_Itsu_v01-08_epub"
    #EBOOK_NAME = "Otonari_no_Tenshisama_ni_Itsu_v05.5_08.5_epub"
    #EBOOK_NAME = "Otonari_no_Tenshisama_ni_Itsu_v09-10_epub"
    #EBOOK_NAME = "Mushoku_Isekai_Dasu_v01-26_epub"
    EBOOK_PATH = f"/home/ifw/epub/{EBOOK_NAME}"
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
