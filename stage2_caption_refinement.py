import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import sqlite3
import json
from tqdm import tqdm
import re

# Mistral official tokenizer
try:
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    from mistral_common.protocol.instruct.messages import UserMessage
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False
    print("⚠️  mistral_common not installed. Falling back to transformers tokenizer.")

# --- Constants ---
DATABASE_FILE = "/root/data/root/23tnt/Track1/retrieval/image_embedding_store_all.db"
DEFAULT_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"  # 32K context, stable and reliable
# Alternative models:
# "mistralai/Mixtral-8x7B-Instruct-v0.1"  # 32K context, high performance (requires mistral_common)
# "Qwen/Qwen2.5-7B-Instruct"  # 32K context, high accuracy
# "microsoft/Phi-3-medium-4k-instruct"  # 4K context, high accuracy

REFINEMENT_PROMPT = """You are a professional caption editor and photojournalist. Your task is to refine and improve an existing image caption by making it more accurate, detailed, and engaging while maintaining factual correctness.

**Guidelines for refinement:**
1. Preserve all factual information from the original caption
2. Improve clarity, flow, and readability
3. Add relevant visual details if they enhance understanding
4. Ensure the caption is compelling and informative
5. Remove redundancy and improve word choice
6. Maintain professional journalistic tone
7. Keep the caption length appropriate (<8000 words)

**Original Article Context:**
{article_content}

**Original Caption:**
{original_caption}

**Your task:** Refine the above caption to make it better while keeping all factual information intact. Output only the refined caption, no explanations.

**Refined Caption:**"""

def setup_database(db_path=DATABASE_FILE):
    """Setup database connection"""
    conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    return conn

def get_metadata_for_db_ids(conn, db_ids_list):
    """Get metadata for a list of database IDs"""
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

def get_content_for_query_id(conn, query_id):
    """Get article content for a specific query_id using improved database extraction"""
    if not conn:
        return "Article content not available."
    
    try:
        cursor = conn.cursor()
        
        # Method 1: Try to find by exact image_path match (most reliable)
        query = """
        SELECT DISTINCT content, article_title, article_id
        FROM image_embeddings 
        WHERE image_path LIKE ?
        ORDER BY LENGTH(content) DESC
        LIMIT 1
        """
        search_pattern = f"%{query_id}%"
        cursor.execute(query, (search_pattern,))
        result = cursor.fetchone()
        
        if result and result[0]:
            content = result[0]
            title = result[1] if result[1] else ""
            article_id = result[2] if result[2] else ""
            return f"Article ID: {article_id}\nArticle Title: {title}\n\nContent: {content}"
        
        # Method 2: Try by article_id or image_id_json
        query = """
        SELECT DISTINCT content, article_title, article_id
        FROM image_embeddings 
        WHERE article_id LIKE ? OR image_id_json LIKE ?
        ORDER BY LENGTH(content) DESC
        LIMIT 1
        """
        cursor.execute(query, (search_pattern, search_pattern))
        result = cursor.fetchone()
        
        if result and result[0]:
            content = result[0]
            title = result[1] if result[1] else ""
            article_id = result[2] if result[2] else ""
            return f"Article ID: {article_id}\nArticle Title: {title}\n\nContent: {content}"
        
        # Method 3: Get any available content (fallback)
        query = """
        SELECT DISTINCT content, article_title, article_id
        FROM image_embeddings 
        WHERE content IS NOT NULL AND content != ''
        ORDER BY RANDOM()
        LIMIT 1
        """
        cursor.execute(query)
        result = cursor.fetchone()
        
        if result and result[0]:
            content = result[0]
            title = result[1] if result[1] else ""
            article_id = result[2] if result[2] else ""
            return f"[FALLBACK] Article ID: {article_id}\nArticle Title: {title}\n\nContent: {content}"
        
        return "Article content not available for this query."
            
    except Exception as e:
        print(f"Error querying database for {query_id}: {e}")
        return "Article content not available."

def refine_caption_with_model(original_caption, article_content, model, tokenizer, device, max_new_tokens=512, model_id=""):
    """Refine caption using the loaded model"""
    
    # Create prompt
    prompt_text = REFINEMENT_PROMPT.format(
        article_content=article_content[:16000],  # Limit article content to prevent overflow
        original_caption=original_caption
    )
    
    # Handle Mixtral with official tokenizer
    if "mixtral" in model_id.lower() and MISTRAL_AVAILABLE:
        print("🔥 Using Mistral official tokenizer for Mixtral")
        mistral_tokenizer = MistralTokenizer.v1()
        completion_request = ChatCompletionRequest(messages=[UserMessage(content=prompt_text)])
        tokens = mistral_tokenizer.encode_chat_completion(completion_request).tokens
        
        # Convert to tensor and move to device
        input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
        
        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.1,
                pad_token_id=mistral_tokenizer.instruct_tokenizer.tokenizer.eos_id
            )
        
        # Decode with mistral tokenizer
        result = mistral_tokenizer.decode(generated_ids[0].tolist())
        
        # Clean up the output
        refined_caption = result.strip()
        if refined_caption.startswith("Refined Caption:"):
            refined_caption = refined_caption.replace("Refined Caption:", "").strip()
        
        return refined_caption
    
    # Fallback to transformers tokenizer for other models
    else:
        # For Qwen models, use chat template
        if "qwen" in model_id.lower():
            messages = [{"role": "user", "content": prompt_text}]
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Tokenize
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=8192)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Deterministic for consistency
                temperature=0.1,
                pad_token_id=tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else None
            )
        
        # Decode only the generated part
        input_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][input_length:]
        refined_caption = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Clean up the output
        refined_caption = refined_caption.strip()
        
        # Remove common artifacts
        if refined_caption.startswith("Refined Caption:"):
            refined_caption = refined_caption.replace("Refined Caption:", "").strip()
        
        return refined_caption

def process_caption_refinement(input_csv, output_csv, model_id=DEFAULT_MODEL_ID, max_new_tokens=512):
    """Main function to process caption refinement"""
    
    print(f"🔧 Starting caption refinement with model: {model_id}")
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print("✅ Model loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Setup database
    print("Connecting to database...")
    db_conn = setup_database()
    
    # Read input CSV
    try:
        df = pd.read_csv(input_csv)
        print(f"📊 Loaded {len(df)} rows from {input_csv}")
        
        # Ensure required columns exist
        if 'query_id' not in df.columns or 'generated_caption' not in df.columns:
            print("❌ Error: Input CSV must contain 'query_id' and 'generated_caption' columns")
            return
            
    except Exception as e:
        print(f"❌ Error reading input CSV: {e}")
        return
    
    # Process refinement
    refined_captions = []
    
    print("🔄 Processing caption refinement...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Refining captions"):
        query_id = row['query_id']
        original_caption = row['generated_caption']
        
        if pd.isna(original_caption) or original_caption.strip() == "":
            refined_captions.append("Caption not available for refinement.")
            continue
        
        try:
            # Get article content for context
            article_content = get_content_for_query_id(db_conn, query_id)
            
            # Refine caption
            refined_caption = refine_caption_with_model(
                original_caption, article_content, model, tokenizer, device, max_new_tokens, model_id
            )
            
            refined_captions.append(refined_caption)
            
        except Exception as e:
            print(f"⚠️  Error refining caption for {query_id}: {e}")
            refined_captions.append(original_caption)  # Fallback to original
    
    # Add refined captions to dataframe
    df['refined_caption'] = refined_captions
    
    # Save output
    try:
        df.to_csv(output_csv, index=False)
        print(f"✅ Refined captions saved to: {output_csv}")
        
        # Print some statistics
        original_lengths = df['generated_caption'].str.len().mean()
        refined_lengths = df['refined_caption'].str.len().mean()
        print(f"📈 Average caption length: {original_lengths:.1f} → {refined_lengths:.1f} characters")
        
    except Exception as e:
        print(f"❌ Error saving output: {e}")
    
    # Cleanup
    if db_conn:
        db_conn.close()
    
    print("🎉 Caption refinement completed!")

def main():
    parser = argparse.ArgumentParser(description="Refine image captions using a language model")
    parser.add_argument("--input_csv", type=str, required=True, 
                       help="Path to input CSV with query_id and generated_caption columns")
    parser.add_argument("--output_csv", type=str, required=True,
                       help="Path to output CSV (will include refined_caption column)")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID,
                       help=f"Hugging Face model ID (default: {DEFAULT_MODEL_ID})")
    parser.add_argument("--max_new_tokens", type=int, default=8192,
                       help="Maximum new tokens for refinement (default: 512)")
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input_csv).exists():
        print(f"❌ Error: Input file {args.input_csv} does not exist")
        return
    
    process_caption_refinement(args.input_csv, args.output_csv, args.model_id, args.max_new_tokens)

if __name__ == "__main__":
    main() 