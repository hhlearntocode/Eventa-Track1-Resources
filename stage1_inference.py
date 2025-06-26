import argparse
import os
import sys
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import json
import numpy as np
import faiss
from transformers import AutoModel, SiglipImageProcessor, LlavaNextForConditionalGeneration, LlavaNextProcessor, AutoProcessor
from PIL import Image
from transformers.image_utils import load_image
import sqlite3
import csv
import time
import re

# --- Constants ---
LLAVA_MODEL_ID_DEFAULT = "llava-hf/llava-v1.6-mistral-7b-hf"
EMBEDDING_DIM = 1536
DATABASE_FILE = "/root/data/root/23tnt/Track1/retrieval/image_embedding_store_all.db"
FAISS_INDEX_FILE = "/root/data/root/23tnt/Track1/retrieval/image_faiss_all.index"
FAISS_ID_MAP_FILE = "/root/data/root/23tnt/Track1/retrieval/faiss_id_map_all.json"
MODEL_NAME = "google/siglip2-giant-opt-patch16-384"
EXAMPLE_SUMMARY = """TuSimple has completed about 2 million miles of road tests with its 70 prototype trucks across the US, China and Europe. TuSimple has deals in place with two of the world's largest truck manufacturers to design and build fully autonomous models, which it hopes to launch by 2024. Oceanbird is a wind-powered transatlantic car carrier that cuts carbon emissions by 90%, compared to a standard car carrier. Cities around the world are adopting electric ferries. Hyperloop could be a sustainable replacement to short-haul flights. Researchers hope to switch out the kerosene with a sustainable fuel source, like liquid hydrogen. In September, Airbus unveiled plans for three hydrogen-powered, zero-emission aircraft which can carry 100 to 200 passengers. In December 2019, Canadian airline Harbour Air flew the world's first all-electric, zero-emission commercial aircraft. E-bikes are now even available on ride-share apps, like Uber. TuSimple's latest road test involved hauling fresh produce 951 miles, from Nogales, Arizona to Oklahoma City. The pickup and the dropoff were handled by a human driver, but for the bulk of the route the truck drove itself. TuSimple's trucks will only be able to drive along pre-mapped trade routes. The company has created a detailed map of each route. Trucks won't be required to operate in bustling city traffic. TuSimple is currently mapping routes between Arizona and Texas. It plans to have mapped routes across the nation by 2024. The technology will add about $50,000 to the cost of a truck. TuSimple aims to take over the routes between terminals and distribution centers. The company believes it can create new freight capacity without creating new demand for drivers. TuSimple is planning its first fully autonomous tests, without a human safety driver in the cabin at all, before the end of the year. The results of such tests will indicate whether the company can meet its goal to launch its own trucks by 2024. Seven-foot "Model-T" robots have been stacking shelves in two of Tokyo's largest convenience store franchises. MIT collaborated with Ava Robotics and the Greater Boston Food Bank to design a robot that can use UV light to sanitize the floor of a 4,000-square foot warehouse. Stretch is the latest robot from Boston Dynamics and can work in warehouses. "Handle" is made for the warehouse and equipped with an on-board vision system. "Spot" can lift objects, pick itself up after a fall, open and walk through doors. TuSimple has prominent competitors, such as Google spinoff Waymo and Uber-backed Aurora. TuSimple is working exclusively on driverless trucks, like US companies Plus and Embark.  tes has other benefits. The company says its trucks react 15 times faster than human drivers. "The truck doesn't get tired, doesn't watch a movie or look at a phone," says Lu."""
EXAMPLE_GENERATED_CAPTION = """The image captures a pivotal moment in the development of TuSimple's autonomous trucking technology. We see one of TuSimple's prototype trucks, predominantly black with a distinctive white cab and silver grille, driving down a black asphalt road. The truck is positioned against a striking backdrop of a vibrant red sky, suggesting either a dramatic sunset or sunrise. This striking contrast emphasizes the innovation and advancement of the technology being showcased. Notably absent from the image are any human drivers, highlighting the core focus of TuSimple's endeavor: fully autonomous driving. This road test is part of TuSimple's ambitious journey to revolutionize the trucking industry. They have already completed approximately 2 million miles of road tests with their prototypes across the US, China, and Europe. The company has partnerships with major truck manufacturers, aiming to launch fully autonomous trucks by 2024. The company selected this stretch of road for its testing due to it being a part of a pre-mapped trade corridor, allowing for greater control and data collection. A team of safety engineers and drivers are present during this test, a necessary precaution as the technology nears full autonomy. However, TuSimple's ultimate goal is to operate these trucks entirely without human intervention, relying on advanced sensors, software, and mapping data to navigate safely and efficiently. This specific road test, according to the accompanying article, was a 951-mile journey from Nogales, Arizona to Oklahoma City. This long haul underscores TuSimple's focus on optimizing long-distance trucking routes. TuSimple believes their technology can offer significant benefits, including increased safety - reducing highway fatalities often caused by human error - as well as increased efficiency and cost savings through longer driving hours and reduced reliance on human drivers. The ambitious vision presented in this image points to a future where autonomous trucks transform the trucking industry and reshape how goods are transported across the globe. While challenges remain, TuSimple appears poised to play a leading role in this technological revolution."""

# ==============================================================================
# LLaVA Caption Generation
# ==============================================================================
def create_llava_prompt(article_summary):
    return f"""You are a photojournalist assistant AI.

Your task is to generate a **detailed and informative caption** for an image, by combining:
1. Visual content from the image,
2. The event summary provided,
3. Automatically extracted semantic tags (you will generate these yourself from the summary).

Be sure to include important **visual elements**, **people**, **actions**, and **semantic keywords** relevant to the article.

---

**Example Summary:**
{EXAMPLE_SUMMARY}

**Example Semantic Tags:**
["autonomous trucks", "TuSimple", "transportation", "technology", "Arizona", "freight", "road test", "prototype", "innovation", "logistics"]

**Example Generated Caption:**
{EXAMPLE_GENERATED_CAPTION}

---

**Current Summary:**
{article_summary}

**Step 1: Semantic Tags (generate key concepts from the article):**
["""

def generate_caption_with_llava(image, summary, llava_model, llava_processor, device, max_new_tokens=8192):
    # Use text-only one-shot prompting with semantic tags generation
    print("✅ Using TEXT-ONLY ONE-SHOT PROMPTING with SEMANTIC TAGS")
    
    prompt_content = f"""You are a photojournalist assistant AI.

Your task is to generate a **detailed and informative caption** for an image, by combining:
1. Visual content from the image,
2. The event summary provided,
3. Automatically extracted semantic tags (you will generate these yourself from the summary).

Be sure to include important **visual elements**, **people**, **actions**, and **semantic keywords** relevant to the article.

---

**Example Summary:**
{EXAMPLE_SUMMARY}

**Example Semantic Tags:**
["autonomous trucks", "TuSimple", "transportation", "technology", "Arizona", "freight", "road test", "prototype", "innovation", "logistics"]

**Example Generated Caption:**
{EXAMPLE_GENERATED_CAPTION}

---

**Current Summary:**
{summary}

**Step 1: Semantic Tags (generate key concepts from the article):**
["""
    
    conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_content}]}]
    images = [image]
    
    prompt = llava_processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = llava_processor(text=prompt, images=images, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[-1]
    
    with torch.no_grad():
        output_ids = llava_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    
    generated_ids = output_ids[:, input_len:]
    full_decoded = llava_processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    return full_decoded.strip()


# ==============================================================================
# Retrieval Functions
# ==============================================================================

def setup_database(db_path=DATABASE_FILE):
    conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    return conn

def get_metadata_for_db_ids(conn, db_ids_list):
    if not db_ids_list:
        return {}
    cursor = conn.cursor()
    placeholders = ', '.join(['?'] * len(db_ids_list))
    query = f"""SELECT id, image_path, article_title, article_url, image_id_json, article_id, content FROM image_embeddings WHERE id IN ({placeholders})"""
    cursor.execute(query, db_ids_list)
    results_map = {}
    for row in cursor.fetchall():
        results_map[row[0]] = {
            "db_id": row[0],
            "image_path": row[1],
            "article_title": row[2],
            "article_url": row[3],
            "image_id_json": row[4],
            "article_id": row[5],
            "content": row[6],
        }
    return results_map

def load_faiss_index_and_map(index_path=FAISS_INDEX_FILE, map_path=FAISS_ID_MAP_FILE):
    if not Path(index_path).exists() or not Path(map_path).exists():
        return None, None
    try:
        index = faiss.read_index(index_path)
        with open(map_path, 'r') as f:
            faiss_to_db_id_map = json.load(f)
        return index, faiss_to_db_id_map
    except Exception as e:
        return None, None

def search_images_by_image_faiss(query_image_path, faiss_index, faiss_to_db_id_map, db_conn, model, processor, device, top_k=10):
    query_image_path = str(query_image_path)
    if not Path(query_image_path).exists():
        return []
    if faiss_index is None or faiss_index.ntotal == 0:
        return []

    model.eval()

    with torch.no_grad():
        image = load_image(query_image_path)
        inputs = processor(images=[image], return_tensors="pt")
        model_dtype = next(model.parameters()).dtype
        inputs = {k: (v.to(device, dtype=model_dtype) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        query_image_embedding = model.get_image_features(**inputs).cpu().numpy()

    query_image_embedding = query_image_embedding.astype(np.float32)
    if query_image_embedding.ndim == 1:
        query_image_embedding = query_image_embedding.reshape(1, -1)
    faiss.normalize_L2(query_image_embedding)
    distances, faiss_indices = faiss_index.search(query_image_embedding, top_k + 1)

    if faiss_indices.size == 0 or faiss_indices[0][0] == -1: return []

    retrieved_db_ids = [faiss_to_db_id_map[i] for i in faiss_indices[0] if i != -1]
    metadata_map = get_metadata_for_db_ids(db_conn, retrieved_db_ids)
    results = []
    query_image_abs_path = Path(query_image_path).resolve()

    for i, faiss_idx in enumerate(faiss_indices[0]):
        if faiss_idx == -1: continue
        db_id = faiss_to_db_id_map[faiss_idx]
        if db_id in metadata_map:
            meta = metadata_map[db_id]
            if Path(meta['image_path']).resolve() == query_image_abs_path: continue
            results.append({**meta, 'similarity': distances[0][i]})
            if len(results) >= top_k: break
    return results

def find_image_file(folder_path, img_name):
    folder_path = Path(folder_path)
    if not folder_path.exists():
        return None
    extensions = ['', '.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    for ext in extensions:
        img_path = folder_path / f"{img_name}{ext}"
        if img_path.exists(): return img_path
    
    try:
        for file in folder_path.iterdir():
            if img_name.lower() in file.name.lower(): return file
    except Exception: pass
    return None



def sanitize_filename(name):
    """Removes characters that are invalid for filenames."""
    return re.sub(r'[\\/*?:"<>|]',"", name)

def process_inference_for_ab_test(input_csv, img_folder, output_csv_base, llava_model_ids, max_new_tokens=2048):
    # --- STAGE 1: Setup and Retrieval ---
    # This stage is run only once for all models.
    print("--- Stage 1: Initializing and running retrieval ---")
    db_conn = setup_database(DATABASE_FILE)
    faiss_index, faiss_id_map = load_faiss_index_and_map()
    
    if faiss_index is None or faiss_id_map is None or faiss_index.ntotal == 0:
        print("FAISS index not available or empty. Exiting.")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model_dtype = torch.float16 if device == "cuda" else torch.float32
    retrieval_model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=model_dtype).to(device)
    retrieval_processor = SiglipImageProcessor.from_pretrained(MODEL_NAME)

    try:
        df = pd.read_csv(input_csv)
        if df.columns[0] != 'query_index':
            df = df.rename(columns={df.columns[0]: 'query_index'})
        df = df[df.query_index.astype(str) != "query_index"].dropna().reset_index(drop=True)
    except Exception as e:
        print(f"Error reading input CSV: {e}")
        db_conn.close()
        return
    
    img_folder_path = Path(img_folder).resolve()
    retrieval_cache = []

    print(f"Retrieving articles for {len(df)} queries...")
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Retrieving articles"):
        query_index = row['query_index']
        query_image_path = find_image_file(img_folder_path, str(query_index))
        
        if not query_image_path:
            print(f"Warning: Image for query_index '{query_index}' not found. Skipping.")
            continue
        
        retrieved_results = search_images_by_image_faiss(
            query_image_path, faiss_index, faiss_id_map, db_conn, retrieval_model, retrieval_processor, device, top_k=30 # Retrieve more to ensure 10 unique
        )
        
        raw_article_ids = [res.get('article_id') for res in retrieved_results if res.get('article_id') and res.get('article_id') != '#']
        
        # Ensure article IDs are unique
        unique_article_ids = list(dict.fromkeys(raw_article_ids))
        pred_article_ids = unique_article_ids[:10] + ['#'] * (10 - len(unique_article_ids[:10]))
        
        summary_for_captioning = "Summary not available for captioning."
        if retrieved_results and retrieved_results[0].get('content'):
            summary_for_captioning = retrieved_results[0]['content']
            
        retrieval_cache.append({
            "query_index": query_index,
            "query_image_path": query_image_path,
            "pred_article_ids": pred_article_ids,
            "content": summary_for_captioning
        })
    
    db_conn.close()
    del retrieval_model, retrieval_processor, faiss_index, faiss_id_map
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("--- Retrieval stage complete ---")

    # --- STAGE 2: Caption Generation (A/B Test Loop) ---
    print("\n--- Stage 2: Generating captions for each model ---")
    if not retrieval_cache:
        print("No queries were successfully retrieved. Exiting.")
        return

    output_path_base = Path(output_csv_base)

    for model_id in llava_model_ids:
        print(f"\nProcessing with model: {model_id}")
        
        try:
            # Load LLaVA Model
            llava_processor = LlavaNextProcessor.from_pretrained(model_id)
            llava_model = LlavaNextForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=model_dtype, low_cpu_mem_usage=True
            ).to(device)
            
            # Prepare output file for this model
            sanitized_model_name = sanitize_filename(model_id.split('/')[-1])
            output_filename = output_path_base.with_name(f"{output_path_base.stem}_{sanitized_model_name}.csv")
            
            print(f"Output will be saved to: {output_filename}")

            with open(output_filename, 'w', newline='', encoding='utf-8') as f_out:
                writer = csv.writer(f_out)
                writer.writerow(['query_id'] + [f"article_id_{i+1}" for i in range(10)] + ["generated_caption"])
                
                for item in tqdm(retrieval_cache, desc=f"Generating captions ({model_id})"):
                    try:
                        query_pil_image = Image.open(item["query_image_path"]).convert("RGB")
                        generated_caption = generate_caption_with_llava(
                            query_pil_image, item["content"], llava_model, llava_processor, device, max_new_tokens
                        )
                    except Exception as e:
                        generated_caption = f"Error during caption generation: {e}"

                    writer.writerow([item["query_index"]] + item["pred_article_ids"] + [generated_caption])
            
        except Exception as e:
            print(f"!!! Critical error processing model {model_id}: {e}. Skipping this model. !!!")
        finally:
            # Unload model to free memory for the next one
            del llava_model, llava_processor
            if torch.cuda.is_available(): torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description="Image Retrieval and Captioning Inference for A/B Testing.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file with query_index.")
    parser.add_argument("--img_folder", type=str, required=True, help="Path to the folder containing query images.")
    parser.add_argument("--output_csv", type=str, default="submission.csv", help="Base path for the output CSV file(s).")
    parser.add_argument("--llava_model_ids", type=str, nargs='+', default=[LLAVA_MODEL_ID_DEFAULT], help="One or more LLaVA model IDs to test.")
    parser.add_argument("--max_new_tokens", type=int, default=8192, help="Max new tokens for LLaVA generation.")
    args = parser.parse_args()
    
    process_inference_for_ab_test(args.input_csv, args.img_folder, args.output_csv, args.llava_model_ids, args.max_new_tokens)

if __name__ == "__main__":
    main() 