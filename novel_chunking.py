import os
import json

from tqdm import tqdm


MAX_CHUNK_SIZE = 1280

NOVEL_NAME = "test"
#NOVEL_NAME = "Heru_modo_Yarikomizuki_no_gema_v01-06_epub"
#NOVEL_NAME = "Heru_modo_Yarikomizuki_no_gema_v07-08_epub"
#NOVEL_NAME = "[依空まつり]_サイレント・ウィッチ_沈黙の魔女の隠しごと_第09巻_epub"
NOVEL_PATH = f"./novel_perplexity/{NOVEL_NAME}"

SAVE_DIR_PATH = f"./novel_chunking/{NOVEL_NAME}"


def text_chunking(
    chunks: list[list[str]], 
    chunks_perplexitys: list[list[float]]
) -> list[str]:
    not_exceeding_limits = True
    for chunk in  chunks:
        if sum([len(sentence) for sentence in chunk]) > MAX_CHUNK_SIZE:
            not_exceeding_limits = False
    
    if not_exceeding_limits:
        return ["".join(chunk) for chunk in chunks]
    
    new_chunks: list[list[str]] = []
    new_chunks_perplexitys: list[list[float]] = []
    for chunk, perplexitys in zip(chunks, chunks_perplexitys):
        if sum([len(sentence) for sentence in chunk]) <= MAX_CHUNK_SIZE:
            new_chunks.append(chunk)
            new_chunks_perplexitys.append(perplexitys)
            continue

        _, perplexity_index = max([(perplexity, index) for index, perplexity in enumerate(perplexitys)])

        new_chunks.extend([chunk[:perplexity_index], chunk[perplexity_index:]])
        new_perplexitys_1 = perplexitys[:perplexity_index]
        new_perplexitys_2 = perplexitys[perplexity_index:]
        new_perplexitys_1[0] = 0
        new_perplexitys_2[0] = 0
        new_chunks_perplexitys.extend([new_perplexitys_1, new_perplexitys_2])
    
    return text_chunking(new_chunks, new_chunks_perplexitys)

def small_chunk_merging(chunks: list[str]) -> list[str]:
    if len(chunks) == 1:
        return chunks

    chunks_sorted = sorted([(len(chunk), index) for index, chunk in enumerate(chunks)])

    for _, chunk_index in chunks_sorted:
        if chunk_index == 0:
            if len(chunks[chunk_index]) + len(chunks[chunk_index + 1]) > MAX_CHUNK_SIZE:
                continue

            chunks[chunk_index:chunk_index + 2] = [chunks[chunk_index] + chunks[chunk_index + 1]]
            return small_chunk_merging(chunks)
        
        if chunk_index + 1 == len(chunks):
            if len(chunks[chunk_index]) + len(chunks[chunk_index - 1]) > MAX_CHUNK_SIZE:
                continue

            chunks[chunk_index - 1:chunk_index + 1] = [chunks[chunk_index - 1] + chunks[chunk_index]]
            return small_chunk_merging(chunks)
        
        _, index_offset = min([
            (len(chunks[chunk_index - 1]), -1),
            (len(chunks[chunk_index + 1]), 1),
        ])

        if len(chunks[chunk_index]) + len(chunks[chunk_index + index_offset]) > MAX_CHUNK_SIZE:
            continue
        
        chunks[chunk_index + min(0, index_offset):chunk_index + 1 + max(0, index_offset)] = [
            chunks[chunk_index + min(0, index_offset)] + chunks[chunk_index + max(0, index_offset)]
        ]
        return small_chunk_merging(chunks)

    return chunks


if not os.path.isdir(SAVE_DIR_PATH):
    os.mkdir(SAVE_DIR_PATH)

novel_file_list = [dir for dir in os.listdir(NOVEL_PATH) if dir.endswith(".json")]

tqdm_progress = tqdm(novel_file_list)
for novel_file in tqdm_progress:
    tqdm_progress.set_description(novel_file)

    with open(f"{NOVEL_PATH}/{novel_file}", "r", encoding="utf-8") as file:
        dataset: list[dict[str, list[str] | list[float]]] = json.load(file)
    
    for data in dataset:
        data["content"] = small_chunk_merging(text_chunking([data["content"]], [data["loss"]]))
        data.pop("loss")
        data.pop("perplexity")
    
    with open(f"{SAVE_DIR_PATH}/{novel_file}", 'w', encoding='utf-8') as file:
        json.dump(dataset, file, indent=4, ensure_ascii=False)
