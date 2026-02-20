import os

from tqdm import tqdm


NEW_CONDA_PATH = "/media/ifw/GameFile/linux_cache/anaconda3"

OLD_CONDA_PATH = "/home/ifw/anaconda3"


for path, folders, files in tqdm(os.walk(NEW_CONDA_PATH)):
    if not files:
        continue

    for file in files:
        file_path = f"{path}/{file}"

        try:
            with open(file_path, "r", encoding="utf-8") as text_file:
                text = text_file.read()

        except (UnicodeDecodeError, FileNotFoundError):
            continue

        new_text = text.replace(OLD_CONDA_PATH, NEW_CONDA_PATH)

        with open(file_path, 'w', encoding='utf-8') as text_file:
            text_file.write(new_text)
