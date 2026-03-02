import os
import json
from statistics import mean

import torch
from torch.nn.functional import cosine_similarity

from vllm import LLM
from tqdm import tqdm
from transformers import AutoTokenizer


USE_SMALL_CHUNK_MERGING = True
USE_BACKWARD_WINDOW = False
VLLM_PROGRESS_BAR = False

DISPLAY_FILE_NAME_LENGTH_LIMIT = 32
BACKWARD_WINDOW_SIZE = 3

MAX_CHUNK_SIZE = 896
MAX_TOKENS = MAX_CHUNK_SIZE + 2 

MAX_REQUESTS = 32 #32 96 256
MAX_BATCHED_TOKENS = MAX_REQUESTS * MAX_TOKENS #8192 16384 32768 65536 131072
VRAM_UTILIZATION = 0.9

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


def truncate_middle(text, max_length, ellipsis="..."):
    if len(text) <= max_length:
        return text

    remaining_len = max_length - len(ellipsis)

    front_len = remaining_len // 2
    back_len = remaining_len - front_len

    return text[:front_len] + ellipsis + text[-back_len:]

def text_chunking(
    chunks: list[list[str]], 
    chunks_values: list[list[float]],
    chunks_token_counts: list[list[int]]
) -> tuple[list[str], list[int]]:
    while True:
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

            if USE_BACKWARD_WINDOW:
                new_values = [mean(values[i:min(i + BACKWARD_WINDOW_SIZE, len(values))]) for i in range(len(values))]

            _, value_index = min([(value, index) for index, value in enumerate(new_values if USE_BACKWARD_WINDOW else values)])

            new_chunks.extend([chunk[:value_index], chunk[value_index:]])
            new_values_1 = values[:value_index]
            new_values_2 = values[value_index:]
            new_values_1[0] = 1
            new_values_2[0] = 1
            new_chunks_values.extend([new_values_1, new_values_2])
            new_chunks_token_counts.extend([chunk_token_counts[:value_index], chunk_token_counts[value_index:]])

        chunks = new_chunks
        chunks_values = new_chunks_values
        chunks_token_counts = new_chunks_token_counts

def update_chunks(
    chunks: list[str],
    chunks_token_count: list[int],
    index: int
):
    chunks[index:index + 2] = [chunks[index] + chunks[index + 1]]
    chunks_token_count[index:index + 2] = [chunks_token_count[index] + chunks_token_count[index + 1]]

def small_chunk_merging(
    chunks: list[str],
    chunks_token_count: list[int]
):
    def total_token_count(index):
        return chunks_token_count[index] + chunks_token_count[index + 1]


    while True:
        if len(chunks) == 1:
            return False, None

        chunks_token_count_sorted = sorted([(token_count, index) for index, token_count in enumerate(chunks_token_count)])

        for _, chunk_index in chunks_token_count_sorted:
            if chunk_index == 0:
                if total_token_count(chunk_index) > MAX_CHUNK_SIZE:
                    continue

                update_chunks(chunks, chunks_token_count, chunk_index)
                break
            
            if chunk_index == len(chunks) - 1:
                if total_token_count(chunk_index - 1) > MAX_CHUNK_SIZE:
                    continue

                update_chunks(chunks, chunks_token_count, chunk_index - 1)
                break

            if (
                total_token_count(chunk_index) <= MAX_CHUNK_SIZE
                and total_token_count(chunk_index - 1) <= MAX_CHUNK_SIZE
            ):
                return True, chunk_index

            if total_token_count(chunk_index) <= MAX_CHUNK_SIZE:
                update_chunks(chunks, chunks_token_count, chunk_index)
                break

            if total_token_count(chunk_index - 1) <= MAX_CHUNK_SIZE:
                update_chunks(chunks, chunks_token_count, chunk_index - 1)
                break
        else:
            return False, None


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

if USE_SMALL_CHUNK_MERGING:
    model = LLM(
        model=MODEL_PATH, 

        dtype="bfloat16",
        quantization="bitsandbytes", 
        #load_format="bitsandbytes",

        gpu_memory_utilization=VRAM_UTILIZATION,
        max_model_len=MAX_TOKENS,
        max_num_batched_tokens=MAX_BATCHED_TOKENS,
        max_num_seqs=MAX_REQUESTS,

        trust_remote_code=True,
        enable_prefix_caching=True,
        enforce_eager=True,
        task="embed"
    )

if not os.path.isdir(SAVE_DIR_PATH):
    os.mkdir(SAVE_DIR_PATH)

novel_file_list = [dir for dir in os.listdir(NOVEL_PATH) if dir.endswith(".json")]

tqdm_progress = tqdm(novel_file_list)
for novel_file in tqdm_progress:
    with open(f"{NOVEL_PATH}/{novel_file}", "r", encoding="utf-8") as file:
        dataset: list[dict[str, list[str] | list[float]]] = json.load(file)
    
    text_chunks: list[tuple[list[str], list[int], bool, int]] = []
    for data in dataset:
        sentences_token_count = [len(tokenizer(sentence, add_special_tokens=False)["input_ids"]) for sentence in data["content"]]
        chunks, chunks_token_count = text_chunking(
            [data["content"]], 
            [data["similarity"]],
            [sentences_token_count]
        )
        text_chunks.append((
            chunks, 
            chunks_token_count,
            True,
            0
        ))

    while USE_SMALL_CHUNK_MERGING and any(to_be_processed for _, _, to_be_processed, _ in text_chunks):
        completed_count = len([to_be_processed for _, _, to_be_processed, _ in text_chunks if not to_be_processed])
        tqdm_progress.set_description(f"{completed_count}/{len(dataset)}|{truncate_middle(novel_file, DISPLAY_FILE_NAME_LENGTH_LIMIT)}")

        chunks_to_be_processed: list[str] = []
        for index, (chunks, chunks_token_count, to_be_processed, chunk_index) in enumerate(text_chunks):
            if to_be_processed:
                to_be_processed, chunk_index = small_chunk_merging(chunks, chunks_token_count)

            text_chunks[index] = (chunks, chunks_token_count, to_be_processed, chunk_index)

            if not to_be_processed:
                continue

            chunks_to_be_processed.extend([
                chunks[chunk_index],
                chunks[chunk_index - 1],
                chunks[chunk_index + 1]
            ])
        
        if not chunks_to_be_processed:
            continue

        embeddings = [
            torch.tensor(output.outputs.embedding) 
            for output in model.embed(
                chunks_to_be_processed,
                use_tqdm=VLLM_PROGRESS_BAR
            )
        ]
        embeddings_iter = iter(embeddings)

        for chunks, chunks_token_count, to_be_processed, chunk_index in text_chunks:
            if not to_be_processed:
                continue

            target_embedding = next(embeddings_iter)
            previous_one_embedding = next(embeddings_iter)
            next_one_embedding = next(embeddings_iter)

            previous_one_similarity = float(cosine_similarity(
                target_embedding.unsqueeze(0), 
                previous_one_embedding.unsqueeze(0)
            )[0])
            next_one_similarity = float(cosine_similarity(
                target_embedding.unsqueeze(0), 
                next_one_embedding.unsqueeze(0)
            )[0])

            _, index_offset = max([
                (previous_one_similarity, -1),
                (next_one_similarity, 0),
            ])

            update_chunks(chunks, chunks_token_count, chunk_index + index_offset)

    for data, (chunks, _, _, _) in zip(dataset, text_chunks):
        data["content"] = chunks
        data.pop("similarity")
    
    with open(f"{SAVE_DIR_PATH}/{novel_file}", 'w', encoding='utf-8') as file:
        json.dump(dataset, file, indent=4, ensure_ascii=False)
