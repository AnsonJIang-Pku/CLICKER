import os
import json
import numpy as np
from dataset_loader import CounterfactDataset, ZsreDataset, WikiFactDiffDataset
from knowledge_editor import DataEncoder, KNNRetriever
from context_builder import format_label, LANG_LABELS

def precompute(test_path: str,
               cache_prefix: str,
               lang: str = "en",
               dataset_type: str = "counterfact",
               k: int = 16):

    DatasetClass = CounterfactDataset if dataset_type == "counterfact" else ZsreDataset
    test_dataset = DatasetClass(test_path, lang=lang)
    test_records = test_dataset.load_data()

    test_queries: list[str] = []
    sep = LANG_LABELS.get(lang, LANG_LABELS["en"])["sep"]
    for rec in test_records:
        req = rec["requested_rewrite"]
        subj = req["subject"]
        prompt_txt = req["prompt"].format(subj)
        target_new = req["target_new"]["str"]

        new_line = format_label(lang, "new_fact", f"{prompt_txt}{sep}{target_new}")
        prompt_line = format_label(lang, "prompt", prompt_txt)
        answer_line = format_label(lang, "answer", "")
        test_queries.append(f"{new_line}\n{prompt_line}\n{answer_line}")

    encoder = DataEncoder()
    train_embs, train_texts = encoder.load_embeddings(cache_prefix)

    test_embs = encoder.model.encode(
        test_queries,
        convert_to_numpy=True,
        show_progress_bar=True
    ).astype("float32")

    retriever = KNNRetriever(train_embs, train_texts)
    distances, indices = retriever.index.search(test_embs, k)

    np.save(f"{cache_prefix}_test_indices.npy", indices)
    np.save(f"{cache_prefix}_test_distances.npy", distances)
    with open(f"{cache_prefix}_test_queries.json", "w", encoding="utf-8") as f:
        json.dump(test_queries, f, ensure_ascii=False, indent=2)

    print(f"[Precompute] Saved indices & distances under prefix '{cache_prefix}_test_*.npy' and queries json.")


if __name__ == "__main__":
    src_lang, tgt_lang = "en", "zh"
    # Counterfact
    precompute(
        test_path=f"/datasets/Multi-CounterFact/{tgt_lang}/counterfact_test_{tgt_lang}_QA_200.json",
        cache_prefix=f"counterfact_train_cache_{src_lang}",
        lang=tgt_lang,
        dataset_type="counterfact",
        k=16
    )