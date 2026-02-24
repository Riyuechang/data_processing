import os
import json

import torch
from torch.nn.functional import cosine_similarity

from vllm import LLM
from tqdm import tqdm
from transformers import AutoTokenizer


VLLM_PROGRESS_BAR = True

MAX_SENTENCE_LEN = 256
SLIDING_WINDOW_SIZE = 512
BACKWARD_WINDOW_SIZE = 64
MAX_TOKENS = SLIDING_WINDOW_SIZE + 2 

MAX_REQUESTS = 32 #32 96 256
MAX_BATCHED_TOKENS = MAX_REQUESTS * MAX_TOKENS #8192 16384 32768 65536 131072
VRAM_UTILIZATION = 0.9

#MODEL_NAME = "jina-embeddings-v3"
#MODEL_NAME = "Qwen3-Embedding-0.6B"
MODEL_NAME = "Qwen3-Embedding-8B"
MODEL_PATH = f"/media/ifw/GameFile/linux_cache/embedding_model/{MODEL_NAME}"

NOVEL_NAME = "test"
#NOVEL_NAME = "[北山結莉] 精霊幻想記 第27巻 ep"
#NOVEL_NAME = "Otonari_no_Tenshisama_ni_Itsu_v01-10_epub"
#NOVEL_NAME = "Heru_modo_Yarikomizuki_no_gema_v01-06_epub"
#NOVEL_NAME = "Heru_modo_Yarikomizuki_no_gema_v07-08_epub"
#NOVEL_NAME = "[依空まつり]_サイレント・ウィッチ_沈黙の魔女の隠しごと_第09巻_epub"
NOVEL_PATH = f"./epub_chapter_content/{NOVEL_NAME}"

SAVE_DIR_PATH = f"./novel_similarity/{NOVEL_NAME}"


def pre_segmentation(
    text: str, 
    seg_symbol: list[str] = ["\n"]
):
    text_seg: list[str] = []
    new_text = text[0]

    for word in text[1:]:
        if word not in seg_symbol and new_text[-1] in seg_symbol:
            text_seg.append(new_text)
            new_text = ""

        new_text += word
    
    if new_text:
        text_seg.append(new_text)

    return text_seg

def sentence_segmentation(text: str):
    SEGMENTATION_SYMBOL_1 = ["\n"]
    SEGMENTATION_SYMBOL_2 = ["\n", "」", "』", "）", ")"]
    SEGMENTATION_SYMBOL_3 = ["\n", "。", ".", "！", "!", "？", "?", "；", ";"]
    SEGMENTATION_SYMBOL_4 = ["\n", "，", ",", "：", ":", "、"]

    sentences = pre_segmentation(text, SEGMENTATION_SYMBOL_1)

    def sentence_len_check(
        sentences: list[str], 
        seg_symbol: list[str]
    ):
        new_sentences: list[str] = []

        for sentence in sentences:
            if len(sentence) <= MAX_SENTENCE_LEN:
                new_sentences.append(sentence)
                continue

            if seg_symbol is None:
                new_sentences.extend([
                    sentence[i:i + MAX_SENTENCE_LEN] 
                    for i in range(0, len(sentence), MAX_SENTENCE_LEN)
                ])
            else:
                new_sentences.extend(pre_segmentation(sentence, seg_symbol))
        
        return new_sentences

    sentences = sentence_len_check(sentences, SEGMENTATION_SYMBOL_2)
    sentences = sentence_len_check(sentences, SEGMENTATION_SYMBOL_3)
    sentences = sentence_len_check(sentences, SEGMENTATION_SYMBOL_4)
    sentences = sentence_len_check(sentences, None)

    return sentences

def similarity_distribution(sentences: list[str]):
    sentences_token_count = [len(tokenizer.tokenize(sentence, add_special_tokens=False)) for sentence in sentences]

    contexts: list[str] = []
    for sentence_index in range(1, len(sentences)):
        context = ""
        context_token_count = 0
        for context_sentence, context_sentence_token_count in reversed(list(zip(sentences[:sentence_index], sentences_token_count[:sentence_index]))):
            context_token_count += context_sentence_token_count

            if context_token_count > SLIDING_WINDOW_SIZE:
                break

            context = context_sentence + context
        
        contexts.append(context)
    
    contexts_embeddings, sentences_embeddings = [
        torch.tensor([
            output.outputs.embedding 
            for output in model.embed(
                texts,
                use_tqdm=VLLM_PROGRESS_BAR
            )
        ])
        for texts in [contexts, sentences[1:]]
    ]

    distributions: list[float] = [1]
    for context_embedding, sentence_embedding in zip(contexts_embeddings, sentences_embeddings):
        distributions.append(float(cosine_similarity(
            context_embedding.unsqueeze(0), 
            sentence_embedding.unsqueeze(0)
        )[0]))

    return distributions


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
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
        tqdm_progress.set_description(novel_file)

        with open(f"{NOVEL_PATH}/{novel_file}", "r", encoding="utf-8") as file:
            dataset: list[dict[str, str]] = json.load(file)

        for data in dataset:
            sentences = sentence_segmentation(data["content"])
            similarity = similarity_distribution(sentences)
            data["content"] = sentences
            data["similarity"] = similarity

        with open(f"{SAVE_DIR_PATH}/{novel_file}", 'w', encoding='utf-8') as file:
            json.dump(dataset, file, indent=4, ensure_ascii=False)