import json

from tqdm import tqdm
from sentence_transformers import SentenceTransformer


BATCH_SIZE = 64

MODEL_PATH = "/media/ifw/GameFile/linux_cache/embedding_model/jina-embeddings-v3"

NOVEL_NAME = "關於我在無意間被隔壁的天使變成廢柴這件事_6"
JP_TW_NOVEL_TRANSLATION_ALIGNMENT_PATH = f"/media/ifw/GameFile/linux_cache/data_processed/jp_tw_novel_translation_alignment/{NOVEL_NAME}_translation.json"

DATA_OUTPUT_PATH = f"/media/ifw/GameFile/linux_cache/data_processed/jp_tw_novel_similarity/{NOVEL_NAME}_similarity.json"


model = SentenceTransformer(MODEL_PATH, trust_remote_code=True)

with open(JP_TW_NOVEL_TRANSLATION_ALIGNMENT_PATH, "r", encoding="utf-8") as file:
    dataset: list[dict[str, str]] = json.load(file)

for chapter in tqdm(dataset, desc="正在計算相似度..."):
    sentences: list[str] = []
    for alignment in chapter["alignment"]:
        sentences.append(alignment["jp"])
        sentences.append(alignment["tw"])
        sentences.append(alignment["tw_translation"])

    embeddings = model.encode(sentences, batch_size=BATCH_SIZE, convert_to_tensor=True)

    embeddings_iter = iter(embeddings)
    for alignment in chapter["alignment"]:
        jp_embedding = next(embeddings_iter)
        tw_embedding = next(embeddings_iter)
        tw_translation_embedding = next(embeddings_iter)

        alignment["similarity_to_jp"] = abs(float(model.similarity(jp_embedding, tw_embedding)[0][0]) - float(model.similarity(jp_embedding, tw_translation_embedding)[0][0]))
        alignment["similarity_translation"] = float(model.similarity(tw_embedding, tw_translation_embedding)[0][0])

with open(DATA_OUTPUT_PATH, 'w', encoding='utf-8') as file:
    json.dump(dataset, file, indent=4, ensure_ascii=False)