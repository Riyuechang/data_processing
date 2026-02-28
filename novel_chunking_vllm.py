import os
import json
from statistics import mean

from torch.nn.functional import cosine_similarity

from tqdm import tqdm
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer


MAX_CHUNK_SIZE = 1280
BACKWARD_WINDOW_SIZE = 1

NOVEL_NAME = "test"
#NOVEL_NAME = "[北山結莉] 精霊幻想記 第27巻 ep"
#NOVEL_NAME = "Otonari_no_Tenshisama_ni_Itsu_v01-10_epub"
#NOVEL_NAME = "Heru_modo_Yarikomizuki_no_gema_v01-06_epub"
#NOVEL_NAME = "Heru_modo_Yarikomizuki_no_gema_v07-08_epub"
#NOVEL_NAME = "[依空まつり]_サイレント・ウィッチ_沈黙の魔女の隠しごと_第09巻_epub"
#NOVEL_PATH = f"./novel_perplexity/{NOVEL_NAME}"
NOVEL_PATH = f"./novel_similarity/{NOVEL_NAME}"

SAVE_DIR_PATH = f"./novel_chunking/{NOVEL_NAME}"

#MODEL_NAME = "jina-embeddings-v3"
#MODEL_NAME = "Qwen3-Embedding-0.6B"
MODEL_NAME = "Qwen3-Embedding-8B"
MODEL_PATH = f"/media/ifw/GameFile/linux_cache/embedding_model/{MODEL_NAME}"

def text_chunking(
    chunks: list[list[str]], 
    chunks_values: list[list[float]],
    chunks_token_counts: list[list[int]]
) -> tuple[list[str], list[int]]:
    if all(
        sum(chunk_token_counts) <= MAX_CHUNK_SIZE 
        for chunk_token_counts in chunks_token_counts
    ):
        return (
            ["".join(chunk) for chunk in chunks], 
            [sum(chunk_token_counts) for chunk_token_counts in chunks_token_counts]
        )

    new_chunks: list[list[str]] = []
    new_chunks_values: list[list[float]] = []
    new_chunks_token_counts: list[list[int]] = []
    for chunk, values, chunk_token_counts in zip(chunks, chunks_values, chunks_token_counts):
        if sum(chunk_token_counts) <= MAX_CHUNK_SIZE:
            new_chunks.append(chunk)
            new_chunks_values.append(values)
            new_chunks_token_counts.append(chunk_token_counts)
            continue

        new_values = [mean(values[i:min(i + BACKWARD_WINDOW_SIZE, len(values))]) for i in range(len(values))]

        _, value_index = min([(value, index) for index, value in enumerate(new_values)])

        new_chunks.extend([chunk[:value_index], chunk[value_index:]])
        new_values_1 = values[:value_index]
        new_values_2 = values[value_index:]
        new_values_1[0] = 1
        new_values_2[0] = 1
        new_chunks_values.extend([new_values_1, new_values_2])
        new_chunks_token_counts.extend([chunk_token_counts[:value_index], chunk_token_counts[value_index:]])
    
    return text_chunking(new_chunks, new_chunks_values, new_chunks_token_counts)

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

        similarity_1 = float(cosine_similarity(
            embeddings[0].unsqueeze(0), 
            embeddings[1].unsqueeze(0)
        )[0])
        similarity_2 = float(cosine_similarity(
            embeddings[0].unsqueeze(0), 
            embeddings[2].unsqueeze(0)
        )[0])
        
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


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = SentenceTransformer(MODEL_PATH, trust_remote_code=True)

if not os.path.isdir(SAVE_DIR_PATH):
    os.mkdir(SAVE_DIR_PATH)

novel_file_list = [dir for dir in os.listdir(NOVEL_PATH) if dir.endswith(".json")]

tqdm_progress = tqdm(novel_file_list)
for novel_file in tqdm_progress:
    tqdm_progress.set_description(novel_file)

    with open(f"{NOVEL_PATH}/{novel_file}", "r", encoding="utf-8") as file:
        dataset: list[dict[str, list[str] | list[float]]] = json.load(file)
    
    text_chunks: list[tuple[list[str], list[int]]] = []
    for data in dataset:
        sentences_token_count = [len(tokenizer(sentence, add_special_tokens=False)) for sentence in data["content"]]
        text_chunks.append(text_chunking(
            [data["content"]], 
            [data["similarity"]],
            [sentences_token_count]
        ))

    for chunks, chunks_token_count in text_chunks:
        pass

    for data in dataset:
        data["content"] = small_chunk_merging(text_chunking(
            [data["content"]], 
            [data["similarity"]]
        ))

        data.pop("similarity")
    
    with open(f"{SAVE_DIR_PATH}/{novel_file}", 'w', encoding='utf-8') as file:
        json.dump(dataset, file, indent=4, ensure_ascii=False)
