import torch
import torch.nn as nn
import os
import numpy as np
import open_clip
from PIL import Image
import pandas as pd
from tqdm import tqdm
import json
from pathlib import Path
import sqlite3
import faiss # Import FAISS

# --- Constants ---
EMBEDDING_DIM = 1024 # For OpenCLIP ViT-H-14 image_embeds
DATABASE_FILE = "image_embedding_store_CLIP.db"
FAISS_INDEX_FILE = "image_faiss_CLIP.index"
FAISS_ID_MAP_FILE = "faiss_id_map_CLIP.json" # To map FAISS indices to DB IDs

# --- Database Utility Functions (SQLite) ---
def adapt_array(arr):
    return arr.tobytes()

def convert_array(text):
    return np.frombuffer(text, dtype=np.float32).reshape(-1, EMBEDDING_DIM)

sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)

def setup_database(db_path=DATABASE_FILE):
    conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS image_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT, /* This 'id' is our db_id */
            article_id TEXT,
            image_id_json TEXT,
            image_path TEXT UNIQUE,
            embedding array, 
            article_url TEXT,
            article_date TEXT,
            article_title TEXT,
            summary TEXT
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_image_path ON image_embeddings (image_path)')
    conn.commit()
    return conn

def insert_image_data_to_db(conn, image_meta_with_embedding):
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO image_embeddings 
            (article_id, image_id_json, image_path, embedding, article_url, article_date, article_title, summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            image_meta_with_embedding['article_id'],
            image_meta_with_embedding['image_id'],
            image_meta_with_embedding['image_path'],
            image_meta_with_embedding['embedding'].astype(np.float32).reshape(1, -1),
            image_meta_with_embedding['url'],
            image_meta_with_embedding['date'],
            image_meta_with_embedding['title'],
            image_meta_with_embedding['summary']
        ))
        conn.commit()
        last_id = cursor.lastrowid # Return the db_id of the inserted row
        # Fetch and check the inserted row
        cursor.execute("SELECT article_id, image_id_json, image_path, embedding, article_url, article_date, article_title, summary FROM image_embeddings WHERE id = ?", (last_id,))
        row = cursor.fetchone()
        field_names = ["article_id", "image_id_json", "image_path", "embedding", "article_url", "article_date", "article_title", "summary"]
        print("[DEBUG] Inserted row:")
        for i, field in enumerate(field_names):
            value = row[i]
            if field == "embedding":
                print(f"  {field}: array shape {value.shape if hasattr(value, 'shape') else type(value)}")
                if value is None or (hasattr(value, 'size') and value.size == 0):
                    print(f"  [WARNING] Field '{field}' is empty!")
            else:
                print(f"  {field}: {repr(value)}")
                if value is None or (isinstance(value, str) and value.strip() == ""):
                    print(f"  [WARNING] Field '{field}' is empty!")
        return last_id
    except sqlite3.IntegrityError:
        # If it already exists, we might want to fetch its ID for FAISS map if we are not re-ingesting
        cursor.execute("SELECT id FROM image_embeddings WHERE image_path = ?", (image_meta_with_embedding['image_path'],))
        row = cursor.fetchone()
        return row[0] if row else None
    except Exception as e:
        print(f"Error inserting data for {image_meta_with_embedding['image_path']}: {e}")
        return None

def get_all_embeddings_and_db_ids(conn):
    """Fetches all embeddings and their corresponding database IDs."""
    cursor = conn.cursor()
    cursor.execute("SELECT id, embedding FROM image_embeddings")
    data = []
    for row in cursor.fetchall():
        db_id = row[0]
        embedding_array = row[1] # Already converted by SQLite type converter
        if embedding_array is not None and embedding_array.shape == (1, EMBEDDING_DIM):
            data.append({'db_id': db_id, 'embedding': embedding_array.squeeze()})
        elif embedding_array is not None : # Handle potential old format if any
             data.append({'db_id': db_id, 'embedding': embedding_array}) # assume it's already 1D
    
    if not data:
        return [], np.array([], dtype=np.float32)

    db_ids = [item['db_id'] for item in data]
    embeddings = np.array([item['embedding'] for item in data]).astype(np.float32)
    return db_ids, embeddings


def get_metadata_for_db_ids(conn, db_ids_list):
    """Retrieves metadata for a list of database IDs."""
    if not db_ids_list:
        return {}
    cursor = conn.cursor()
    # Create a string of placeholders for the query: (?, ?, ?, ...)
    placeholders = ', '.join(['?'] * len(db_ids_list))
    query = f"""SELECT id, image_path, article_title, article_url, image_id_json, summary 
                FROM image_embeddings WHERE id IN ({placeholders})"""
    cursor.execute(query, db_ids_list)
    
    results_map = {}
    for row in cursor.fetchall():
        results_map[row[0]] = { # Keyed by db_id
            "db_id": row[0],
            "image_path": row[1],
            "article_title": row[2],
            "article_url": row[3],
            "image_id_json": row[4],
            "summary": row[5],
        }
    return results_map


def check_if_image_exists_in_db(conn, image_path_to_check):
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(id) FROM image_embeddings WHERE image_path = ?", (image_path_to_check,))
    count = cursor.fetchone()[0]
    return count > 0

# --- FAISS Utility Functions ---
def build_and_save_faiss_index(db_conn, index_path=FAISS_INDEX_FILE, map_path=FAISS_ID_MAP_FILE):
    print("Building FAISS index from database...")
    db_ids, embeddings_np = get_all_embeddings_and_db_ids(db_conn)

    if embeddings_np.size == 0:
        print("No embeddings found in the database to build FAISS index.")
        # Create empty files or handle as an error
        if Path(index_path).exists(): os.remove(index_path)
        if Path(map_path).exists(): os.remove(map_path)
        return None, []

    # Normalize embeddings for IndexFlatIP (cosine similarity)
    faiss.normalize_L2(embeddings_np)

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings_np)
    
    print(f"FAISS index built with {index.ntotal} vectors.")
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to {index_path}")

    # Save the mapping from FAISS's sequential IDs (0 to ntotal-1) to our db_ids
    # In this case, since we add all at once, FAISS ID 'i' corresponds to db_ids[i]
    faiss_to_db_id_map = db_ids # list where index is faiss_id, value is db_id
    with open(map_path, 'w') as f:
        json.dump(faiss_to_db_id_map, f)
    print(f"FAISS ID map saved to {map_path}")
    
    return index, faiss_to_db_id_map

def load_faiss_index_and_map(index_path=FAISS_INDEX_FILE, map_path=FAISS_ID_MAP_FILE):
    if not Path(index_path).exists() or not Path(map_path).exists():
        print("FAISS index or map file not found.")
        return None, None
    try:
        index = faiss.read_index(index_path)
        print(f"FAISS index loaded from {index_path} with {index.ntotal} vectors.")
        with open(map_path, 'r') as f:
            faiss_to_db_id_map = json.load(f)
        print(f"FAISS ID map loaded from {map_path}")
        return index, faiss_to_db_id_map
    except Exception as e:
        print(f"Error loading FAISS index or map: {e}")
        return None, None

# --- Embedding and Search Functions ---
def process_image_batch(conn, batch_images, batch_metas, model, preprocess, device):
    if not batch_images:
        return 0
    try:
        # Convert PIL images to tensor using OpenCLIP preprocess
        image_tensors = []
        for img in batch_images:
            if not isinstance(img, Image.Image):
                img = Image.open(img).convert('RGB')
            image_tensor = preprocess(img).unsqueeze(0)
            image_tensors.append(image_tensor)
        
        # Stack tensors and move to device
        image_batch = torch.cat(image_tensors, dim=0).to(device)
        
        with torch.no_grad():
            try:
                # Use OpenCLIP's encode_image method
                image_features = model.encode_image(image_batch)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # Normalize
                embeddings_np = image_features.cpu().detach().numpy()
            except Exception as e:
                print(f"[Batch] Error extracting embeddings: {e}")
                return 0
                
        count = 0
        for i, meta in enumerate(batch_metas):
            meta['embedding'] = embeddings_np[i]
            insert_image_data_to_db(conn, meta)
            count += 1
        return count
    except Exception as e:
        print(f"[Batch] Error processing image batch: {e}")
        return 0

def ingest_embeddings_from_json_to_db(db_conn, image_folder_path, metadata_json_file, summaries_json_file, model, preprocess, device, force_reingest_all=False, batch_size=8):
    print(f"Loading metadata from: {metadata_json_file}")
    with open(metadata_json_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    print(f"Loading summaries from: {summaries_json_file}")
    with open(summaries_json_file, 'r', encoding='utf-8') as f:
        summaries_data = json.load(f)

    total_images = 0
    for article_id, article_details in metadata.items():
        total_images += len(article_details.get("images", []))
    print(f"Total images in metadata file: {total_images}")

    images_to_process_count = 0
    newly_inserted_count = 0
    skipped_count = 0
    error_count = 0
    
    model.eval()
    batch_images = []
    batch_metas = []
    with torch.no_grad():
        article_pbar = tqdm(metadata.items(), desc="Processing articles", total=len(metadata))
        for article_id, article_details in article_pbar:
            article_pbar.set_description(f"Article {article_id}")
            images = article_details.get("images", [])
            image_pbar = tqdm(images, desc=f"Images in article {article_id}", leave=False) if len(images) > 5 else images
            for image_filename_stem in image_pbar:
                if isinstance(image_pbar, tqdm):
                    image_pbar.set_description(f"Image {image_filename_stem}")
                found_image_path_str = None
                for ext in ['.jpg', '.jpeg', '.png', '.webp', '']:
                    current_path = Path(image_folder_path) / (image_filename_stem + ext)
                    if current_path.is_file():
                        found_image_path_str = str(current_path)
                        break
                if not found_image_path_str:
                    continue
                images_to_process_count += 1
                if not force_reingest_all and check_if_image_exists_in_db(db_conn, found_image_path_str):
                    skipped_count += 1
                    article_pbar.set_postfix(processed=images_to_process_count, inserted=newly_inserted_count, skipped=skipped_count, errors=error_count)
                    continue
                try:
                    image = Image.open(found_image_path_str).convert('RGB')
                    summary = summaries_data.get(article_id, {}).get("summary", "")
                    meta = {
                        'article_id': article_id,
                        'image_id': image_filename_stem,
                        'image_path': found_image_path_str,
                        'url': article_details.get("url"),
                        'date': article_details.get("date"),
                        'title': article_details.get("title"),
                        'summary': summary
                    }
                    batch_images.append(image)
                    batch_metas.append(meta)
                    if len(batch_images) == batch_size:
                        inserted = process_image_batch(db_conn, batch_images, batch_metas, model, preprocess, device)
                        newly_inserted_count += inserted
                        batch_images = []
                        batch_metas = []
                        article_pbar.set_postfix(processed=images_to_process_count, inserted=newly_inserted_count, skipped=skipped_count, errors=error_count)
                except Exception as e:
                    error_count += 1
                    article_pbar.set_postfix(processed=images_to_process_count, inserted=newly_inserted_count, skipped=skipped_count, errors=error_count)
        # Process any remaining images in the last batch
        if batch_images:
            inserted = process_image_batch(db_conn, batch_images, batch_metas, model, preprocess, device)
            newly_inserted_count += inserted
    print(f"\nTổng kết:")
    print(f"- Số lượng ảnh từ JSON: {total_images}")
    print(f"- Số ảnh tìm thấy trong thư mục: {images_to_process_count}")
    print(f"- Số ảnh đã xử lý và thêm mới: {newly_inserted_count}")
    print(f"- Số ảnh bỏ qua (đã tồn tại): {skipped_count}")
    print(f"- Số ảnh lỗi: {error_count}")
    success_rate = (newly_inserted_count / (images_to_process_count - skipped_count)) * 100 if (images_to_process_count - skipped_count) > 0 else 0
    print(f"- Tỷ lệ thành công: {success_rate:.2f}%")

def ingest_single_image(db_conn, image_path, model, preprocess, device, metadata=None):
    """Process and add a single image to the database."""
    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        return None
    
    if check_if_image_exists_in_db(db_conn, image_path) and not metadata.get("force_reingest", False):
        print(f"Image already exists in database: {image_path}")
        cursor = db_conn.cursor()
        cursor.execute("SELECT id FROM image_embeddings WHERE image_path = ?", (image_path,))
        row = cursor.fetchone()
        return row[0] if row else None
    
    print(f"Processing image: {image_path}")
    progress_bar = tqdm(total=5, desc="Processing steps")
    
    model.eval()
    with torch.no_grad():
        try:
            # Step 1: Load image
            progress_bar.set_description("Loading image")
            image = Image.open(image_path).convert('RGB')
            progress_bar.update(1)
            
            # Step 2: Preprocess image
            progress_bar.set_description("Preprocessing")
            image_tensor = preprocess(image).unsqueeze(0)
            progress_bar.update(1)
            
            # Step 3: Move to device
            progress_bar.set_description("Moving to device")
            image_tensor = image_tensor.to(device)
            progress_bar.update(1)
            
            # Step 4: Extract embeddings
            progress_bar.set_description("Extracting embeddings")
            try:
                with torch.cuda.amp.autocast(enabled=device=="cuda"):
                    image_features = model.encode_image(image_tensor)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # Normalize
                embedding_np = image_features.cpu().detach().numpy()
            except Exception as e:
                progress_bar.close()
                raise ValueError(f"Cannot extract image embeddings: {e}")
            progress_bar.update(1)
            
            # Validate embedding dimension
            if embedding_np.shape[1] != EMBEDDING_DIM:
                progress_bar.close()
                print(f"Error: Embedding dimension mismatch. Expected {EMBEDDING_DIM}, got {embedding_np.shape[1]}.")
                return None
            
            # Step 5: Save to database
            progress_bar.set_description("Saving to database")
            if metadata is None:
                metadata = {}
            
            image_filename = Path(image_path).name
            image_id = Path(image_path).stem
            
            image_data_for_db = {
                'article_id': metadata.get('article_id', 'unknown'),
                'image_id': metadata.get('image_id', image_id),
                'image_path': image_path,
                'embedding': embedding_np,
                'url': metadata.get('url', ''),
                'date': metadata.get('date', ''),
                'title': metadata.get('title', ''),
                'summary': metadata.get('summary', '')
            }
            
            db_id = insert_image_data_to_db(db_conn, image_data_for_db)
            progress_bar.update(1)
            progress_bar.close()
            
            print(f"✅ Image processed successfully: {image_path}")
            return db_id
        
        except Exception as e:
            progress_bar.close()
            print(f"❌ Error processing image {image_path}: {e}")
            return None

def search_images_by_image_faiss(query_image_path, faiss_index, faiss_to_db_id_map, db_conn, model, preprocess, device, top_k=5):
    if not Path(query_image_path).exists():
        print(f"Query image not found: {query_image_path}")
        return []
    if faiss_index is None or faiss_index.ntotal == 0:
        print("FAISS index not available or empty.")
        return []

    model.eval()
    with torch.no_grad():
        try:
            image = Image.open(query_image_path).convert('RGB')
            image_tensor = preprocess(image).unsqueeze(0).to(device)
            
            # Extract embeddings using OpenCLIP
            query_image_embedding = model.encode_image(image_tensor)
            query_image_embedding = query_image_embedding / query_image_embedding.norm(dim=-1, keepdim=True)  # Normalize
            query_image_embedding = query_image_embedding.cpu().numpy()
                
        except Exception as e:
            print(f"Error processing query image {query_image_path}: {e}")
            return []
            
    faiss.normalize_L2(query_image_embedding) # Normalize query embedding

    distances, faiss_indices = faiss_index.search(query_image_embedding, top_k + 1) # Fetch one extra to handle self-match
    
    if faiss_indices.size == 0 or faiss_indices[0][0] == -1:
        return []

    retrieved_db_ids = [faiss_to_db_id_map[i] for i in faiss_indices[0] if i != -1]
    metadata_map = get_metadata_for_db_ids(db_conn, retrieved_db_ids)
    
    results = []
    query_image_abs_path = Path(query_image_path).resolve()

    for i, faiss_idx in enumerate(faiss_indices[0]):
        if faiss_idx == -1: continue
        db_id = faiss_to_db_id_map[faiss_idx]
        if db_id in metadata_map:
            meta = metadata_map[db_id]
            # Skip self-match: if the retrieved image is the query image
            if Path(meta['image_path']).resolve() == query_image_abs_path:
                # print(f"Skipping self-match: {meta['image_path']}")
                continue
            
            results.append({
                **meta,
                'similarity': distances[0][i]
            })
            if len(results) >= top_k:
                break 
    return results


def main():
    # --- Configuration ---
    model_name = "ViT-H-14"
    pretrained = "laion2b_s32b_b79k"  # One of the best checkpoints for ViT-H-14
    
    # Check CUDA availability
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using device: {device} (CUDA available)")
    else:
        device = "cpu"
        print(f"CUDA not available, falling back to device: {device}")
    
    # Set these paths to your image folder and metadata JSON
    image_data_folder = "/root/EVENTA/data/database_original_image/database_img"
    metadata_input_json = "/root/EVENTA/data/database/database.json"
    summaries_input_json = "/root/23tnt/Track2/DB_newCaption/summarize.json"
    
    # Debug prints for path existence
    print("Current working directory:", os.getcwd())
    print("Image folder exists:", Path(image_data_folder).exists())
    print("Metadata JSON exists:", Path(metadata_input_json).exists())
    print("Summaries JSON exists:", Path(summaries_input_json).exists())

    # Configuration flags
    # If True, re-processes all images from JSON, updates DB, and rebuilds FAISS index.
    # If False, only adds new images from JSON to DB.
    FORCE_REPROCESS_ALL_JSON_IMAGES = False 
    
    # If True, always rebuilds FAISS index from DB after ingestion, even if index files exist.
    FORCE_REBUILD_FAISS_INDEX = False 

    # --- Database Setup ---
    db_conn = setup_database(DATABASE_FILE)
    print(f"Database setup at {DATABASE_FILE}")

    # --- Load OpenCLIP Model ---
    print(f"Loading OpenCLIP model: {model_name} with {pretrained} weights")
    try:
        # Load OpenCLIP model and transforms
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=pretrained,
            device=device
        )
        
        print(f"✅ OpenCLIP {model_name} loaded successfully")
        print(f"   Embedding dimension: {EMBEDDING_DIM}")
        print(f"   Pretrained weights: {pretrained}")
                
    except Exception as e:
        print(f"Error loading OpenCLIP model: {e}")
        print("Available models:")
        print(open_clip.list_models())
        if db_conn:
            db_conn.close()
        return

    # --- Ingest/Update Embeddings in Database ---
    if not Path(image_data_folder).exists() or not Path(metadata_input_json).exists() or not Path(summaries_input_json).exists():
        print(f"Error: Image data folder, metadata JSON or summaries JSON not found. Paths are crucial.")
    else:
        print("Ingesting/Updating embeddings in database...")
        ingest_embeddings_from_json_to_db(
            db_conn, Path(image_data_folder), Path(metadata_input_json), Path(summaries_input_json),
            model, preprocess, device,
            force_reingest_all=FORCE_REPROCESS_ALL_JSON_IMAGES
        )

    # --- Load or Build FAISS Index ---
    faiss_index, faiss_id_map = None, None
    if FORCE_REBUILD_FAISS_INDEX or not Path(FAISS_INDEX_FILE).exists() or not Path(FAISS_ID_MAP_FILE).exists():
        print("Rebuilding FAISS index and map...")
        faiss_index, faiss_id_map = build_and_save_faiss_index(db_conn)
    else:
        print("Loading existing FAISS index and map...")
        faiss_index, faiss_id_map = load_faiss_index_and_map()

    if faiss_index is None or faiss_id_map is None or faiss_index.ntotal == 0:
        print("Failed to load or build a valid FAISS index. Search will not be available.")
    else:
        print(f"FAISS setup complete. Index has {faiss_index.ntotal} vectors.")
        
        # --- Interactive Image Search Loop ---
        print("\n" + "="*50)
        print("    INTERACTIVE IMAGE SEARCH & MANAGEMENT    ")
        print("="*50 + "\n")
        
        while True:
            print("\nOPTIONS:")
            print("1. Search by image")
            print("2. Add new image to database") 
            print("3. Database statistics")
            print("q. Quit")
            
            choice = input("\nYour choice: ").lower()
            
            if choice == 'q':
                break
                
            elif choice == '1':
                print("\n--- IMAGE SEARCH ---")
                query_img_path = input("Enter path to your query image: ")
                if not Path(query_img_path).is_file():
                    print(f"❌ Error: Image file not found: {query_img_path}")
                    continue
                    
                top_k = 5
                try:
                    top_k = int(input("Number of results to return (default 5): ") or 5)
                except ValueError:
                    print("Invalid number, using default 5")
                    top_k = 5
                
                print("\nSearching for similar images...")
                progress_bar = tqdm(total=3, desc="Search progress")
                
                try:
                    # Step 1: Load and process query image
                    progress_bar.set_description("Processing query image")
                    results = search_images_by_image_faiss(query_img_path, faiss_index, faiss_id_map, db_conn, model, preprocess, device, top_k)
                    progress_bar.update(3)  # Complete all steps
                    progress_bar.close()
                    
                    if results:
                        print("\n✅ Found matching images:")
                        for i, res in enumerate(results):
                            similarity_pct = res['similarity'] * 100
                            print(f"  {i+1}. Image: {res['image_path']}")
                            if res['article_title']:
                                print(f"     Title: {res['article_title']}")
                            if res['article_url']:
                                print(f"     URL: {res['article_url']}")
                            if res.get('summary'):
                                print(f"     Summary: {res['summary'][:200]}...")
                            print(f"     Similarity: {similarity_pct:.2f}%\n")
                    else:
                        print("❌ No matching images found.")
                except Exception as e:
                    progress_bar.close()
                    print(f"❌ Error during search: {e}")
                    
            elif choice == '2':
                print("\n--- ADD NEW IMAGE ---")
                new_img_path = input("Enter path to the new image: ")
                if not Path(new_img_path).is_file():
                    print(f"❌ Error: Image file not found: {new_img_path}")
                    continue
                    
                metadata = {
                    'article_id': input("Article ID (optional): ") or 'user_added',
                    'title': input("Title (optional): ") or '',
                    'url': input("URL (optional): ") or '',
                    'date': input("Date (optional): ") or '',
                    'summary': input("Summary (optional): ") or '',
                    'force_reingest': True
                }
                
                print("\nProcessing and adding image to database...")
                db_id = ingest_single_image(db_conn, new_img_path, model, preprocess, device, metadata)
                
                if db_id is not None:
                    print(f"\n✅ Image added successfully to database (ID: {db_id})")
                    print("\nUpdating FAISS index to include the new image...")
                    
                    # Show progress for FAISS index rebuild
                    rebuild_progress = tqdm(total=3, desc="Rebuilding index")
                    rebuild_progress.update(1)
                    faiss_index, faiss_id_map = build_and_save_faiss_index(db_conn)
                    rebuild_progress.update(2)
                    rebuild_progress.close()
                    
                    print("✅ FAISS index updated successfully.")
                else:
                    print("❌ Failed to add image to database")
                    
            elif choice == '3':
                print("\n--- DATABASE STATISTICS ---")
                try:
                    # Show database stats with progress
                    stats_progress = tqdm(total=3, desc="Collecting statistics")
                    
                    # Count images
                    cursor = db_conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM image_embeddings")
                    image_count = cursor.fetchone()[0]
                    stats_progress.update(1)
                    
                    # Count articles
                    cursor.execute("SELECT COUNT(DISTINCT article_id) FROM image_embeddings")
                    article_count = cursor.fetchone()[0]
                    stats_progress.update(1)
                    
                    # Get FAISS info
                    faiss_count = faiss_index.ntotal if faiss_index else 0
                    stats_progress.update(1)
                    stats_progress.close()
                    
                    print(f"\nTotal images in database: {image_count}")
                    print(f"Total articles: {article_count}")
                    print(f"FAISS index size: {faiss_count} vectors")
                    
                    # Check if database and FAISS are in sync
                    if image_count != faiss_count:
                        print(f"\n⚠️ Warning: Database ({image_count}) and FAISS index ({faiss_count}) are out of sync.")
                        if input("Would you like to rebuild the FAISS index? (y/n): ").lower() == 'y':
                            print("\nRebuilding FAISS index...")
                            rebuild_progress = tqdm(total=1, desc="Rebuilding index")
                            faiss_index, faiss_id_map = build_and_save_faiss_index(db_conn)
                            rebuild_progress.update(1)
                            rebuild_progress.close()
                            print("✅ FAISS index rebuilt successfully.")
                    
                except Exception as e:
                    print(f"❌ Error getting statistics: {e}")
            
            else:
                print("❌ Invalid option. Please try again.")
    
    db_conn.close()
    print("Exited.")

if __name__ == "__main__":
    main()