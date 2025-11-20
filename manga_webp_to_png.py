import os
from io import BytesIO
from zipfile import ZipFile
from multiprocessing import Pool, cpu_count

from PIL import Image
from tqdm import tqdm


def processing(file_name: str):
    output_path = f"{MANGA_PATH.rstrip('.zip')}/{file_name.split('/')[0]}"
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    output_path = f"{MANGA_PATH.rstrip('.zip')}/{file_name.rstrip('.cbz')}"
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    with ZipFile(MANGA_PATH, mode="r") as zip_file:
       file_data = zip_file.read(file_name)

    with ZipFile(BytesIO(file_data), strict_timestamps=False) as inner_zip_file:
        all_inner_file_name = inner_zip_file.namelist()
        all_webp_file_name = [inner_file_name for inner_file_name in all_inner_file_name if inner_file_name.endswith(".webp")]

        for webp_file_name in all_webp_file_name:
            image_webp = Image.open(BytesIO(inner_zip_file.read(webp_file_name)))
            image_webp.save(f"{output_path}/{webp_file_name.rstrip('.webp')}.png", "png")


MANGA_PATH = "/media/ifw/GameFile/linux_cache/data_unprocessed/manga.zip"

output_path = MANGA_PATH.rstrip('.zip')
if not os.path.isdir(output_path):
    os.mkdir(output_path)

if __name__ == "__main__":
    with ZipFile(MANGA_PATH, mode="r") as zip_file:
        all_file_name = zip_file.namelist()

        with Pool(cpu_count()) as pool:
            response = list(tqdm(
                pool.imap(
                    processing, 
                    all_file_name
                ), 
                desc=f"cpu_count: {cpu_count()}",
                total=len(all_file_name)
            ))