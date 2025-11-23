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

    if not title_id and page_element.a:
        title_id = page_element.a.get("id")

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
    ebook_content = [content for content in enumerate(list(ebook.get_items_of_type(ebooklib.ITEM_DOCUMENT)))]

    catalog_page_index = 0
    catalog_page_path = ebook.toc[catalog_page_index].href.split("#")[0]

    if type(only_numbers(catalog_page_path)) is not float and "toc" not in catalog_page_path:
        catalog_page_index = 1
        catalog_page_path = ebook.toc[catalog_page_index].href.split("#")[0]

    catalog_page_content = [content for _, content in ebook_content if content.file_name == catalog_page_path][0]
    soup = BeautifulSoup(catalog_page_content.get_content().decode("utf-8"), 'html.parser')
    a_tag = soup.body.find_all("a")
    catalog_href = [page_element["href"] for page_element in a_tag if page_element.get("href")]

    title_data_list: list[dict[str, str | None]] = []
    for href, toc in zip(catalog_href, ebook.toc[catalog_page_index + 1:]):
        href_split = href.split("#")

        title_data_list.append(
            {
                "title": toc.title, 
                "title_id": None if len(href_split) < 2 else href_split[1], 
                "file_path": href_split[0],
                "file_index": [
                    content_index 
                    for content_index, content in ebook_content 
                    if href_split[0] in content.file_name
                ][0]
            }
        )

    print([(toc.title, toc.href) for toc in ebook.toc])
    #assert title_data_list and any(title_data["title_id"] for title_data in title_data_list), "title_id為空"
    if not any(title_data["title_id"] for title_data in title_data_list):
        toc_iterable = iter(ebook.toc)
        toc = next(toc_iterable)

        contents: list[dict[str, str]] = []
        for content_index, content in ebook_content:
            if toc.href in content.file_name:
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

        contents: list[dict[str, str]] = [
            {
                "title": ebook.toc[catalog_page_index].title,
                "content": "\n".join(
                    [
                        toc.title 
                        for toc in ebook.toc[catalog_page_index + 1:]
                    ]
                )
            }
        ]

        content_iterable = iter(ebook_content)
        content_index, content = next(content_iterable)

        print(content.file_name)
        print(ebook.toc[0].href)
        for toc in ebook.toc:
            if toc.href in content.file_name:
                contents.append({
                    "title": toc.title,
                    "content": ""
                })

            soup = BeautifulSoup(content.get_content().decode("utf-8"), 'html.parser')
            contents[-1]["content"] += f"\n{soup.body.get_text()}\n"
            content_index, content = next(content_iterable, None)

        for content in contents:
            content["content"] = re.sub(
                pattern=r"\n+",
                repl="\n",
                string=content["content"].strip()
            )

        return contents

    chapters: list[dict[str, str]] = []
    for content_index, content in ebook_content:
        soup = BeautifulSoup(content.get_content().decode("utf-8"), 'html.parser')

        if soup.body.div:
            main_div = [
                (len(find_main_div.find_all("p")), index, find_main_div) 
                for index, find_main_div in enumerate(soup.body.find_all("div"))
            ]
            _, _, content_main = max(main_div)
        else:
            content_main = soup.body

        chapter_content: dict[str, str] = {"file_path": content.file_name, "file_index": content_index, "title_id": "", "content": ""}
        for page_element in content_main.contents:
            title_id = find_title_id(page_element)
            
            if title_id:
                chapters.append(chapter_content)
                chapter_content = {
                    "file_path": content.file_name, 
                    "file_index": content_index, 
                    "title_id": title_id if [True for id in title_data_list if id["title_id"] == title_id] else "", 
                    "content": ""
                }

            chapter_content["content"] += f"\n{page_element.get_text()}\n"

        if chapter_content["content"]:
            chapters.append(chapter_content)

    chapters_clean = [chapter for chapter in chapters if not only_newline(chapter["content"]) or chapter["title_id"]]

    chapters_iterable = iter(chapters_clean)
    chapter_data = next(chapters_iterable)

    contents: list[dict[str, str]] = [
        {
            "title": ebook.toc[catalog_page_index].title,
            "content": "\n".join(
                [
                    toc.title 
                    for toc in ebook.toc[catalog_page_index + 1:] 
                    if toc.title in [title_data["title"] for title_data in title_data_list]
                ]
            )
        }
    ]
    for title_data in title_data_list:
        content_data: dict[str, str] = {
            "title": title_data["title"],
            "content": ""
        }

        while True:
            if not chapter_data:
                break

            if chapter_data["file_index"] < title_data["file_index"]:
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

    #EBOOK_NAME = "[依空まつり]_サイレント・ウィッチ_沈黙の魔女の隠しごと_第09巻_epub"
    #EBOOK_NAME = "Heru_modo_Yarikomizuki_no_gema_v01-06_epub"
    #EBOOK_NAME = "Heru_modo_Yarikomizuki_no_gema_v07-08_epub"
    #EBOOK_NAME = "test"
    EBOOK_NAME = "[北山結莉] 精霊幻想記 第27巻 ep"
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
        data_clean = [chapter for chapter in data if chapter["content"]]

        with open(f"{SAVE_DIR_PATH}/{epub_file.rstrip('.epub')}.json", 'w', encoding='utf-8') as file:
            json.dump(data_clean, file, indent=4, ensure_ascii=False)
