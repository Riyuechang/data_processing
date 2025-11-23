import os
import re
import json
from zipfile import ZipFile

from tqdm import tqdm
from bs4 import BeautifulSoup, element


#EBOOK_NAME = "Mushoku_Isekai_Dasu_v01-26_epub"
EBOOK_NAME = "[北山結莉] 精霊幻想記 第27巻 ep"

EBOOK_PATH = f"/home/ifw/epub/{EBOOK_NAME}"
SAVE_DIR_PATH = f"./epub_chapter_content/{EBOOK_NAME}"


def extract_content(epub_path: str):
    with ZipFile(epub_path, mode="r") as zip_file:
        container_xml = zip_file.read("META-INF/container.xml").decode()

        container_soup = BeautifulSoup(container_xml, 'xml')
        rootfile_element = container_soup.find("rootfile", attrs={"full-path": True})
        full_path = rootfile_element.get("full-path")
        opf_flie = zip_file.read(full_path).decode()

        opf_soup = BeautifulSoup(opf_flie, 'xml')
        spine_element = opf_soup.find("spine")
        itemref_element = spine_element.find_all("itemref")
        idref_list = [element.get("idref") for element in itemref_element]

        manifest_element = opf_soup.find("manifest")
        content_href = [manifest_element.find("item", id=idref).get("href") for idref in idref_list]

        guide_element = opf_soup.find("guide")

        if guide_element:
            toc_href = guide_element.find("reference", type="toc").get("href")
        else:
            toc_element = manifest_element.find("item", id=re.compile(r"toc"))

            if not toc_element:
                toc_element = manifest_element.find("item", id=re.compile(r"ncx"))

            if not toc_element:
                toc_element = manifest_element.find("item", id=re.compile(r"nav"))

            toc_href = toc_element.get("href")
        
        for file_name in zip_file.namelist():
            if toc_href.split("#")[0] in file_name:
                toc_path = file_name
                break

        content_path: list[str] = []
        for href in content_href:
            for file_name in zip_file.namelist():
                if href in file_name:
                    content_path.append(file_name)
                    break

        epub_content = [(file_path, zip_file.read(file_path).decode()) for file_path in content_path]
        epub_toc = zip_file.read(toc_path).decode()
    
    toc_soup = BeautifulSoup(epub_toc, 'xml')
    toc_a_tag = toc_soup.body.find_all("a", href=True)
    toc_data = {a_tag.get_text(): a_tag.get("href").split("#") for a_tag in toc_a_tag}
    toc_id = [data[-1] for data in toc_data.values() if len(data) == 2]
    #toc_not_id_path = [data[0] for data in toc_data.values() if len(data) != 2]
    
    if toc_id:
        id_pattern = "|".join(toc_id)
        id_pattern = re.compile(f"({id_pattern})")

        page_text: list[str] = []
        for _, content in epub_content:
            content_soup = BeautifulSoup(content, 'xml')
            find_id_element = content_soup.body.find_all(id=id_pattern)

            if find_id_element:
                for id_element in find_id_element:
                    id_element.string = f"### TOC_ID ###\n{id_element.get_text()}"
                
                page_text.append(content_soup.body.get_text())
                continue

            if page_text:
                page_text[-1] += f"\n{content_soup.body.get_text()}\n"
        
        all_content = "".join(page_text)
        all_content_id_split = all_content.split("### TOC_ID ###")[1:]

        assert len(all_content_id_split) == len(toc_id), "目錄ID與內容ID不對齊"

        id_content_alignment = {
            id: re.sub(
                pattern=r"\n+",
                repl="\n",
                string=content.strip()
            ) 
            for id, content in zip(toc_id, all_content_id_split)
        }

        contents = [
            {
                "title": key,
                "content": id_content_alignment.get(value[-1])
            } 
            for key, value in toc_data.items()
        ]
        contents_clean = [content for content in contents if content["content"]]
        contents_clean.insert(0, {
            "title": "contents",
            "content": "\n".join([content["title"] for content in contents_clean])
        })

        return contents_clean

    toc_data_iter = iter(toc_data.items())
    title, title_path = next(toc_data_iter)

    contents: list[dict[str, str]] = []
    for file_path, content in epub_content:
        if title_path and title_path[0] in file_path:
            contents.append({
                "title": title,
                "content": ""
            })
            title, title_path = next(toc_data_iter, (None, None))
        
        if contents:
            content_soup = BeautifulSoup(content, 'xml')
            contents[-1]["content"] += f"\n{content_soup.body.get_text()}\n"
    
    for content in contents:
        content["content"] = re.sub(
            pattern=r"\n+",
            repl="\n",
            string=content["content"].strip()
        )
    
    contents.insert(0, {
        "title": "contents",
        "content": "\n".join([content["title"] for content in contents])
    })

    return contents


if not os.path.isdir(SAVE_DIR_PATH):
    os.mkdir(SAVE_DIR_PATH)

epub_file_list = [dir for dir in os.listdir(EBOOK_PATH) if dir.endswith(".epub")]

tqdm_progress = tqdm(epub_file_list)
for epub_file in tqdm_progress:
    tqdm_progress.set_description(epub_file)

    epub_content = extract_content(f"{EBOOK_PATH}/{epub_file}")

    with open(f"{SAVE_DIR_PATH}/{epub_file.rstrip('.epub')}.json", 'w', encoding='utf-8') as file:
        json.dump(epub_content, file, indent=4, ensure_ascii=False)