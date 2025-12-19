import os
import random
from dataset_loader import CounterfactDataset, ZsreDataset, WikiFactDiffDataset
from knowledge_editor import KnowledgeEditor, DataEncoder
from precompute_retrieval import precompute

random.seed(42)

def generate_cache_if_not_exist(dataset_path: str,
                                cache_prefix: str,
                                prefix_dir: str,
                                src_lang: str,
                                tgt_lang: str,
                                DatasetClass=CounterfactDataset):
    emb_file = cache_prefix + "_embeddings.npy"
    txt_file = cache_prefix + "_texts.json"
    if not (os.path.exists(emb_file) and os.path.exists(txt_file)):
        print("Cache not found; starting to load and preprocess the training dataset...")
        dataset = DatasetClass(dataset_path, lang=src_lang)
        demo_texts = dataset.prepare_demo_texts()
        encoder = DataEncoder('BAAI/bge-m3')
        embeddings = encoder.encode_texts(demo_texts)
        encoder.save_embeddings(embeddings, demo_texts, cache_prefix)
        print(f"Save cache {cache_prefix}_embeddings.npy and {cache_prefix}_texts.json")
    else:
        print("Cache file found; loading directly.")

def generate_test_cache_if_not_exist(test_path: str,
                                     cache_prefix: str,
                                     lang: str,
                                     dataset_type: str,
                                     k: int = 16):
    idx_file = cache_prefix + "_test_indices.npy"
    dist_file = cache_prefix + "_test_distances.npy"
    query_file = cache_prefix + "_test_queries.json"

    if not (os.path.exists(idx_file) and os.path.exists(dist_file) and os.path.exists(query_file)):
        print("Test cache not found; precomputing retrieval indices...")
        precompute(
            test_path=test_path,
            cache_prefix=cache_prefix,
            lang=lang,
            dataset_type=dataset_type,
            k=k
        )
    else:
        print("Test cache files found; loading directly.")

def main(dataset_type: str = "counterfact",
         src_lang: str = "en",
         tgt_lang: str = "zh",
         threshold: float = 0.60,
         use_cot: bool = False,
         use_remake: bool = False,
         use_open_model: bool = False):

    device = 'cuda'
    model_path = "/models/Qwen/Qwen2.5-7B-Instruct"
    # model_path = "/models/llama-hf/Llama-3.1-8B-Instruct"

    if not use_open_model:
        model_path = "gpt-4o-mini"
        # model_path = "gpt-4.1-mini"
        # model_path = "gpt-4.1-nano"
    else:
        # For local models, ensure the model_path is set correctly
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path {model_path} does not exist. Please check the path.")

    if dataset_type == "zsre":
        DatasetClass = ZsreDataset
        cache_prefix = f"zsre_train_cache_{src_lang}"
        # Mzsre
        src_train_path = f"/datasets/MzsRE/{src_lang}/mzsre_train_{src_lang}.json"
        tgt_train_path = f"/datasets/MzsRE/{tgt_lang}/mzsre_train_{tgt_lang}.json"
        src_test_path  = f"/datasets/MzsRE/{src_lang}/mzsre_test_{src_lang}_200.json"
        tgt_test_path  = f"/datasets/MzsRE/{tgt_lang}/mzsre_test_{tgt_lang}_200.json"

    elif dataset_type == "counterfact":
        DatasetClass = CounterfactDataset
        cache_prefix = f"counterfact_train_cache_{src_lang}"

        # mid_QA(200): src-tgt
        src_train_path = f"/datasets/Multi-CounterFact/{src_lang}/counterfact_train_{src_lang}_QA.json"
        tgt_train_path = f"/datasets/Multi-CounterFact/{tgt_lang}/counterfact_train_{tgt_lang}_QA.json"
        src_test_path  = f"/datasets/Multi-CounterFact/{src_lang}/counterfact_test_{src_lang}_QA_200.json"
        tgt_test_path  = f"/datasets/Multi-CounterFact/{tgt_lang}/counterfact_test_{tgt_lang}_QA_200.json"
    
    else:
        raise NotImplementedError("Please implement the dataset interface.")

    generate_cache_if_not_exist(
        src_train_path,
        cache_prefix,
        prefix_dir='',
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        DatasetClass=DatasetClass
    )

    generate_test_cache_if_not_exist(
        test_path=tgt_test_path,
        cache_prefix=cache_prefix,
        lang=tgt_lang,
        dataset_type=dataset_type,
        k=16
    )

    src_dataset = DatasetClass(src_train_path, lang=src_lang)
    tgt_dataset = DatasetClass(tgt_train_path, lang=tgt_lang)

    ke = KnowledgeEditor(
        embed_cache_prefix=cache_prefix,
        lm_model_name=model_path,
        device=device,
        src_dataset=src_dataset,
        tgt_dataset=tgt_dataset,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        use_cot=use_cot,
        use_remake=use_remake,
        threshold=threshold,
        use_open_model=use_open_model
    )

    return ke, src_test_path, tgt_test_path, dataset_type

if __name__ == "__main__":
    use_open_model = True # True：不用GPT
    use_cot_flag = True
    use_remake_flag = False

    announcement = "####### CounterFact-200, CLICKER, en-zh, Qwen2.5-7b-instruct #######"
    # announcement = "####### Mzsre, CLICKER, en-zh, Qwen2.5-7B #######"
    print(announcement)
    # Counterfact
    ke, src_test, tgt_test, dt = main(
        dataset_type="counterfact",
        src_lang="en",
        tgt_lang="zh",
        threshold=0.60,
        use_cot=use_cot_flag,
        use_remake=use_remake_flag,
        use_open_model=use_open_model
    )

    # Mzsre
    # ke, src_test, tgt_test, dt = main(
    #     dataset_type="zsre",
    #     src_lang="en",
    #     tgt_lang="zh",
    #     threshold=0.60,
    #     use_cot=use_cot_flag,
    #     use_remake=use_remake_flag,
    #     use_open_model=use_open_model
    # )

    from evaluate import evaluate_all
    evaluate_all(ke, src_test, tgt_test, dt, use_generate=False)
    print(announcement)
