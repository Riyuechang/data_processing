import os
import json

from tqdm import tqdm
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer


BATCH_SIZE = 24
SLIDING_WINDOW_SIZE = 2048
MAX_SENTENCE_LEN = 256

MODEL_NAME = "Qwen3-Embedding-0.6B"
#MODEL_NAME = "jina-embeddings-v3"
MODEL_PATH = f"/media/ifw/GameFile/linux_cache/embedding_model/{MODEL_NAME}"

NOVEL_NAME = "test"
#NOVEL_NAME = "Heru_modo_Yarikomizuki_no_gema_v01-06_epub"
#NOVEL_NAME = "Heru_modo_Yarikomizuki_no_gema_v07-08_epub"
#NOVEL_NAME = "[依空まつり]_サイレント・ウィッチ_沈黙の魔女の隠しごと_第09巻_epub"
NOVEL_PATH = f"./epub_chapter_content/{NOVEL_NAME}"

SAVE_DIR_PATH = f"./novel_similarity/{NOVEL_NAME}"


def total_proportion(numbers: list[int | float]):
    number_sum = sum(numbers)
    return [number / number_sum for number in numbers]

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
        model.encode(
            texts, 
            batch_size=BATCH_SIZE, 
            convert_to_tensor=True, 
            #show_progress_bar=True
        )
        for texts in [contexts, sentences[1:]]
    ]

    distributions: list[float] = [1]
    for context_embedding, sentence_embedding in zip(contexts_embeddings, sentences_embeddings):
        distributions.append(float(model.similarity(context_embedding, sentence_embedding)[0][0]))

    return distributions


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = SentenceTransformer(MODEL_PATH, trust_remote_code=True).bfloat16()

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