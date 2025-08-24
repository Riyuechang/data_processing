import os
import re
import json

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup, element
from tqdm import tqdm


#EBOOK_NAME = "Heru_modo_Yarikomizuki_no_gema_v01-06_epub"
#EBOOK_NAME = "Heru_modo_Yarikomizuki_no_gema_v07-08_epub"
EBOOK_NAME = "test"
#EBOOK_NAME = "Otonari_no_Tenshisama_ni_Itsu_v01-08_epub"
#EBOOK_NAME = "Otonari_no_Tenshisama_ni_Itsu_v05.5_08.5_epub"
#EBOOK_NAME = "Otonari_no_Tenshisama_ni_Itsu_v09-10_epub"
#EBOOK_NAME = "Mushoku_Isekai_Dasu_v01-26_epub"
EBOOK_PATH = f"/home/ifw/epub/{EBOOK_NAME}"

TRANSLATION_PATH = f"/home/ifw/Python_project/data_processing/translation/{EBOOK_NAME}"

SAVE_DIR_PATH = f"/home/ifw/epub/{EBOOK_NAME}_translation"


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

def replace_translation(ebook: epub.EpubBook, translation_dataaet: list[dict[str, str | list[dict[str, str]]]]):
    ebook_content = [content for content in list(ebook.get_items_of_type(ebooklib.ITEM_DOCUMENT)) if only_numbers(content.file_name)]

    catalog_page_index = 0
    catalog_page_path = ebook.toc[catalog_page_index].href.split("#")[0]

    if not only_numbers(catalog_page_path):
        catalog_page_index = 1
        catalog_page_path = ebook.toc[catalog_page_index].href.split("#")[0]

    catalog_page_content = [content for content in ebook_content if content.file_name == catalog_page_path][0]
    soup = BeautifulSoup(catalog_page_content.get_content().decode("utf-8"), 'html.parser')
    a_tag = soup.body.div.find_all("a")
    catalog_href = [page_element["href"] for page_element in a_tag]

    title_data_list: list[dict[str, str]] = [
            {
                "title": ebook.toc[catalog_page_index].title, 
                "file_path": catalog_page_path
            }
        ]
    for href, toc in zip(catalog_href, ebook.toc[catalog_page_index + 1:]):
        href_split = href.split("#")

        if not only_numbers(href_split[0]):
            continue

        title_data_list.append(
            {
                "title": toc.title, 
                "file_path": href_split[0]
            }
        )

    for title_data_index, chapter in zip(range(len(title_data_list)), translation_dataaet):
        chapter_ebook_content = [
            content 
            for content in ebook_content 
            if abs(only_numbers(content.file_name)) >= abs(only_numbers(title_data_list[title_data_index]["file_path"])) and (
                title_data_index + 1 == len(title_data_list)
                or abs(only_numbers(content.file_name)) <= abs(only_numbers(title_data_list[title_data_index + 1]["file_path"]))
            )
        ]

        p_tag_dict = {
            "soup": [],
            "p_tag": []
        }
        for chapter_content in chapter_ebook_content:
            p_tag_dict["soup"].append(BeautifulSoup(chapter_content.get_body_content().decode("utf-8"), 'html.parser'))
            p_tag_dict["p_tag"].extend(p_tag_dict["soup"][-1].find_all("p"))
        
        p_tag_dict["p_tag_text"] = [p_tag.get_text() for p_tag in p_tag_dict["p_tag"]]

        for content in range(len(chapter["content"])):
            jp_seg = chapter["content"][content]["jp"].strip("\n").split("\n")
            translation_seg = chapter["content"][content]["translation"].strip("\n").split("\n")

            for jp, translation in zip(jp_seg, translation_seg):
                for p_tag_index, p_tag_text in enumerate(p_tag_dict["p_tag_text"]):
                    if p_tag_text == jp:
                        a_tag = p_tag_dict["p_tag"][p_tag_index].a

                        if a_tag:
                            a_tag.string = translation
                            new_a_tag = BeautifulSoup(str(a_tag), 'html.parser').a
                            p_tag_dict["p_tag"][p_tag_index].clear()
                            p_tag_dict["p_tag"][p_tag_index].append(new_a_tag)
                        else:
                            p_tag_dict["p_tag"][p_tag_index].string = translation

                        break
                else:
                    assert True, "對齊錯誤"
        
        for index, chapter_content in enumerate(chapter_ebook_content):
            chapter_content.set_content(str(p_tag_dict["soup"][index]))

    return ebook


if not os.path.isdir(SAVE_DIR_PATH):
    os.mkdir(SAVE_DIR_PATH)

epub_file_list = [dir for dir in os.listdir(EBOOK_PATH) if dir.endswith(".epub")]

tqdm_progress = tqdm(epub_file_list)
for epub_file in tqdm_progress:
    tqdm_progress.set_description(epub_file)

    ebook = epub.read_epub(f"{EBOOK_PATH}/{epub_file}")

    with open(f"{TRANSLATION_PATH}/{epub_file.rstrip('.epub')}.json", "r", encoding="utf-8") as file:
        translation_dataaet: list[dict[str, str | list[dict[str, str]]]] = json.load(file)

    epub.write_epub(f"{SAVE_DIR_PATH}/{epub_file}", replace_translation(ebook, translation_dataaet))