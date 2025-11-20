import os
from zipfile import ZipFile
from multiprocessing import Pool, cpu_count

from tqdm import tqdm
from PIL import Image


MANGA_PATH = "/media/ifw/GameFile/linux_cache/data_unprocessed/manga"

WEBP_TO_JPG = True

def processing(cbz_path: str):
    save_path = f"{MANGA_PATH}_image{cbz_path.split(MANGA_PATH)[-1].rstrip('.cbz')}"

    with ZipFile(cbz_path, mode="r") as zip_file:
        zip_file.extractall(save_path)
    
    not_image_file = [
        dir 
        for dir in os.listdir(save_path) 
        if not (
            dir.endswith(".webp")
            or dir.endswith(".png")
            or dir.endswith(".jpg")
            or dir.endswith(".jpeg")
        )
    ]

    for file in not_image_file:
        os.remove(f"{save_path}/{file}")
    
    if not WEBP_TO_JPG:
        return

    webp_file = [
        dir 
        for dir in os.listdir(save_path) 
        if dir.endswith(".webp")
    ]

    for file in webp_file:
        image_webp = Image.open(f"{save_path}/{file}")
        image_webp.save(f"{save_path}/{file.rstrip('.webp')}.jpg", "jpeg")

    for file in webp_file:
        os.remove(f"{save_path}/{file}")

if __name__ == "__main__":
    cbz_path_list: list[str] = []

    for path, folders, files in os.walk(MANGA_PATH):
        if not files:
            continue

        cbz_path_list.extend([
            f"{path}/{file}"
            for file in files
            if file.endswith(".cbz")
        ])

    with Pool(cpu_count()) as pool:
        response = list(tqdm(
            pool.imap(
                processing, 
                cbz_path_list
            ), 
            desc=f"cpu_count: {cpu_count()}",
            total=len(cbz_path_list)
        ))