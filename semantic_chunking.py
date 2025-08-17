import json

from sentence_transformers import SentenceTransformer


CHUNKING_THRESHOLD = 0.4
#MAX_CHUNKING_TOKEN_SIZE = 8192
#MAX_CHUNKING_SIZE = 3

#DATA_PATH = "./text.txt"
DATA_PATH = "test_data/無職転生　ルーデウスが嫁たちに前世のことを告白する話.txt"
DATA_OUTPUT_PATH = "./output/無職転生　ルーデウスが嫁たちに前世のことを告白する話.json"

MODEL_PATH = "/media/ifw/GameFile/linux_cache/embedding_model/jina-embeddings-v3"
#MODEL_PATH = "/media/ifw/GameFile/linux_cache/embedding_model/multilingual-e5-large"
#MODEL_PATH = "/media/ifw/GameFile/linux_cache/embedding_model/multilingual-e5-small"


def pre_segmentation(text: str):
    SEGMENTATION_SYMBOL = ["\n"]#["\n", "。", "！", "？", "；", ".", "!", "?", ";"]

    text_seg: list[str] = []
    new_text = text[0]

    for word in text[1:]:
        if word not in SEGMENTATION_SYMBOL and new_text[-1] in SEGMENTATION_SYMBOL:
            text_seg.append(new_text)
            new_text = ""

        new_text += word
    
    if new_text:
        text_seg.append(new_text)

    return text_seg

def text_chunking(text: str):
    text_seg = pre_segmentation(text)

    chunking_list: list[str] = []
    new_chunk: str = text_seg[0]

    for new_text in text_seg[1:]:
        new_chunk_embedding = model.encode(new_chunk)
        new_text_embedding = model.encode(new_text)

        similarity = float(model.similarity(new_chunk_embedding, new_text_embedding)[0][0])

        if similarity < CHUNKING_THRESHOLD:
            chunking_list.append(new_chunk)
            new_chunk = ""

        new_chunk += new_text
    
    if new_chunk:
        chunking_list.append(new_chunk)

    return chunking_list


if __name__ == "__main__":
    model = SentenceTransformer(MODEL_PATH, trust_remote_code=True)#.bfloat16()#.half()

    with open(DATA_PATH, "r", encoding="utf-8") as file:
        #data: list[dict[str, str]] = json.load(file)
        data: str = file.read()

    #data_chunking = text_chunking(data[0]["content"].strip())
    data_chunking = text_chunking(data.strip())

    with open(DATA_OUTPUT_PATH, 'w', encoding='utf-8') as file:
        json.dump(data_chunking, file, indent=4, ensure_ascii=False)