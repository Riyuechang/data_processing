import os
import json
from typing import Literal
from statistics import mean

from tqdm import tqdm
from sentence_transformers import SentenceTransformer


MAX_CHUNK_SIZE = 1280
BACKWARD_WINDOW_SIZE = 3

#NOVEL_NAME = "test"
NOVEL_NAME = "Otonari_no_Tenshisama_ni_Itsu_v01-10_epub"
#NOVEL_NAME = "Heru_modo_Yarikomizuki_no_gema_v01-06_epub"
#NOVEL_NAME = "Heru_modo_Yarikomizuki_no_gema_v07-08_epub"
#NOVEL_NAME = "[依空まつり]_サイレント・ウィッチ_沈黙の魔女の隠しごと_第09巻_epub"
#NOVEL_PATH = f"./novel_perplexity/{NOVEL_NAME}"
NOVEL_PATH = f"./novel_similarity/{NOVEL_NAME}"

SAVE_DIR_PATH = f"./novel_chunking/{NOVEL_NAME}"

#MODEL_NAME = "Qwen3-Embedding-0.6B"
MODEL_NAME = "jina-embeddings-v3"
MODEL_PATH = f"/media/ifw/GameFile/linux_cache/embedding_model/{MODEL_NAME}"

def text_chunking(
    chunks: list[list[str]], 
    chunks_values: list[list[float]],
    mode: Literal["max", "min"]
) -> list[str]:
    not_exceeding_limits = True
    for chunk in  chunks:
        if sum([len(sentence) for sentence in chunk]) > MAX_CHUNK_SIZE:
            not_exceeding_limits = False
    
    if not_exceeding_limits:
        return ["".join(chunk) for chunk in chunks]
    
    new_chunks: list[list[str]] = []
    new_chunks_values: list[list[float]] = []
    for chunk, values in zip(chunks, chunks_values):
        if sum([len(sentence) for sentence in chunk]) <= MAX_CHUNK_SIZE:
            new_chunks.append(chunk)
            new_chunks_values.append(values)
            continue

        new_values = [mean(values[i:min(i + BACKWARD_WINDOW_SIZE, len(values))]) for i in range(len(values))]

        if mode == "max":
            _, value_index = max([(value, index) for index, value in enumerate(new_values)])
        else:
            _, value_index = min([(value, index) for index, value in enumerate(new_values)])

        new_chunks.extend([chunk[:value_index], chunk[value_index:]])
        new_values_1 = values[:value_index]
        new_values_2 = values[value_index:]
        new_values_1[0] = 0 if mode == "max" else 1
        new_values_2[0] = 0 if mode == "max" else 1
        new_chunks_values.extend([new_values_1, new_values_2])
    
    return text_chunking(new_chunks, new_chunks_values, mode=mode)

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
        
        embeddings = model.encode(
            [
                chunks[chunk_index],
                chunks[chunk_index - 1],
                chunks[chunk_index + 1]
            ],
            convert_to_tensor=True
        )

        similarity_1 = float(model.similarity(embeddings[0], embeddings[1])[0][0])
        similarity_2 = float(model.similarity(embeddings[0], embeddings[2])[0][0])
        
        _, index_offset = max([
            (similarity_1, -1),
            (similarity_2, 1),
        ])

        if len(chunks[chunk_index]) + len(chunks[chunk_index + index_offset]) > MAX_CHUNK_SIZE:
            continue
        
        chunks[chunk_index + min(0, index_offset):chunk_index + 1 + max(0, index_offset)] = [
            chunks[chunk_index + min(0, index_offset)] + chunks[chunk_index + max(0, index_offset)]
        ]
        return small_chunk_merging(chunks)

    return chunks


model = SentenceTransformer(MODEL_PATH, trust_remote_code=True)

if not os.path.isdir(SAVE_DIR_PATH):
    os.mkdir(SAVE_DIR_PATH)

novel_file_list = [dir for dir in os.listdir(NOVEL_PATH) if dir.endswith(".json")]

tqdm_progress = tqdm(novel_file_list)
for novel_file in tqdm_progress:
    tqdm_progress.set_description(novel_file)

    with open(f"{NOVEL_PATH}/{novel_file}", "r", encoding="utf-8") as file:
        dataset: list[dict[str, list[str] | list[float]]] = json.load(file)
    
    mode: Literal["max", "min"] = "max"
    if "similarity" in dataset[0].keys():
        mode = "min"
    
    for data in dataset:
        data["content"] = small_chunk_merging(text_chunking(
            [data["content"]], 
            [data["loss"]] if mode == "max" else [data["similarity"]],
            mode=mode
        ))

        if mode == "max":
            data.pop("loss")
            data.pop("perplexity")
        else:
            data.pop("similarity")
    
    with open(f"{SAVE_DIR_PATH}/{novel_file}", 'w', encoding='utf-8') as file:
        json.dump(dataset, file, indent=4, ensure_ascii=False)
