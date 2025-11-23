import os
import json
from zipfile import ZipFile

import warnings
from bs4 import XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
from bs4 import BeautifulSoup, PageElement
from tqdm import tqdm


WORD_COUNT_DIFFERENCE_LIMIT = 1

EBOOK_NAME = "[北山結莉] 精霊幻想記 第27巻 ep"
#EBOOK_NAME = "[依空まつり]_サイレント・ウィッチ_沈黙の魔女の隠しごと_第09巻_epub"
#EBOOK_NAME = "Heru_modo_Yarikomizuki_no_gema_v01-06_epub"
#EBOOK_NAME = "Heru_modo_Yarikomizuki_no_gema_v07-08_epub"
#EBOOK_NAME = "test"
#EBOOK_NAME = "Otonari_no_Tenshisama_ni_Itsu_v01-08_epub"
#EBOOK_NAME = "Otonari_no_Tenshisama_ni_Itsu_v05.5_08.5_epub"
#EBOOK_NAME = "Otonari_no_Tenshisama_ni_Itsu_v09-10_epub"
#EBOOK_NAME = "Mushoku_Isekai_Dasu_v01-26_epub"
EBOOK_PATH = f"/home/ifw/epub/{EBOOK_NAME}"

TRANSLATION_PATH = f"/home/ifw/Python_project/data_processing/translation/{EBOOK_NAME}"

SAVE_DIR_PATH = f"/home/ifw/epub/{EBOOK_NAME}_translation"


def replace_translation(
    epub_path: str, 
    save_path: str,
    translation_dataaet: list[dict[str, str | list[dict[str, str]]]]
):
    with ZipFile(epub_path, mode="r") as zip_file:
        container_xml = zip_file.read("META-INF/container.xml").decode()

        container_soup = BeautifulSoup(container_xml, 'html.parser')
        rootfile_element = container_soup.find("rootfile", attrs={"full-path": True})
        full_path = rootfile_element.get("full-path")
        opf_flie = zip_file.read(full_path).decode()

        opf_soup = BeautifulSoup(opf_flie, 'html.parser')
        spine_element = opf_soup.find("spine")
        itemref_element = spine_element.find_all("itemref")
        idref_list = [element.get("idref") for element in itemref_element]

        manifest_element = opf_soup.find("manifest")
        content_href = [manifest_element.find("item", id=idref).get("href") for idref in idref_list]

        content_path: list[str] = []
        for href in content_href:
            for file_name in zip_file.namelist():
                if href in file_name:
                    content_path.append(file_name)
                    break
        
        epub_content = [(file_path, BeautifulSoup(zip_file.read(file_path).decode(), 'html.parser')) for file_path in content_path]
        epub_other_data = [(file_path, zip_file.read(file_path)) for file_path in zip_file.namelist() if file_path not in content_path]
    
    all_text_element: list[PageElement] = []
    for _, content in epub_content:
        all_text_element.extend(content.find_all("p"))
        all_text_element.extend(content.find_all("h1"))
        all_text_element.extend(content.find_all("h2"))
        all_text_element.extend(content.find_all("h3"))
        all_text_element.extend(content.find_all("h4"))
        all_text_element.extend(content.find_all("h5"))
        all_text_element.extend(content.find_all("h6"))
    
    all_text_element_content = [p_tag.get_text().replace("　", " ") for p_tag in all_text_element]

    for chapter in translation_dataaet:
        for content in chapter["content"]:
            jp_seg = content["jp"].strip("\n").split("\n")
            translation_seg = content["translation"].strip("\n").split("\n")

            for jp, translation in zip(jp_seg, translation_seg):
                for index in range(len(all_text_element)):
                    jp = jp.replace("　", " ")

                    if (
                        jp not in all_text_element_content[index] 
                        or abs(len(jp) - len(all_text_element_content[index])) > WORD_COUNT_DIFFERENCE_LIMIT
                    ):
                        continue
                    
                    all_text_element_content[index] = all_text_element_content[index].replace(jp, translation)
                    a_tag = all_text_element[index].a

                    if a_tag:
                        a_tag.string = all_text_element_content[index]
                        new_a_tag = BeautifulSoup(str(a_tag), 'html.parser').a
                        all_text_element[index].clear()
                        all_text_element[index].append(new_a_tag)
                        break

                    all_text_element[index].string = all_text_element_content[index]

                    break
                else:
                    assert False, f"對齊錯誤: {[jp]}"
    
    with ZipFile(save_path, mode="w") as save_zip_file:
        for file_path, content in epub_content:
            save_zip_file.writestr(file_path, str(content))

        for file_path, content in epub_other_data:
            save_zip_file.writestr(file_path, content)


if not os.path.isdir(SAVE_DIR_PATH):
    os.mkdir(SAVE_DIR_PATH)

epub_file_list = [dir for dir in os.listdir(EBOOK_PATH) if dir.endswith(".epub")]

tqdm_progress = tqdm(epub_file_list)
for epub_file in tqdm_progress:
    tqdm_progress.set_description(epub_file)

    with open(f"{TRANSLATION_PATH}/{epub_file.rstrip('.epub')}.json", "r", encoding="utf-8") as file:
        translation_dataaet: list[dict[str, str | list[dict[str, str]]]] = json.load(file)

    replace_translation(
        f"{EBOOK_PATH}/{epub_file}", 
        f"{SAVE_DIR_PATH}/{epub_file}", 
        translation_dataaet
    )