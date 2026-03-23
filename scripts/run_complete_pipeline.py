#!/usr/bin/env python3
"""
Complete Pipeline Runner - Simplified
Processes all 22 PDFs, generates chapter summaries, book summaries, and final report.
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from main import PDFEmbeddingPipeline
from summary_cache import SummaryCache
import numpy as np

def main():
    print("="*80)
    print("COMPLETE PDF PROCESSING PIPELINE")
    print("="*80)
    print()
    
    # Configuration
    pdf_dir = Path("/Users/GStahl2/AI/sourcepdfs")
    vector_store_dir = Path("./vector_stores")
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return 1
    
    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = PDFEmbeddingPipeline(
        pdf_dir=str(pdf_dir),
        vector_store_dir=str(vector_store_dir),
        embedder_type="sentence-transformers",
        chunk_size=512,
        chunk_overlap=50,
        openai_api_key=api_key
    )
    
    # Get all PDFs
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDFs to process\n")
    
    # STEP 1: Process all PDFs (create vector stores and chunks)
    print("="*80)
    print("STEP 1: PROCESSING PDFs (Creating vector stores)")
    print("="*80)
    print()
    
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Processing {pdf_path.name}...")
        try:
            pipeline.process_single_pdf(pdf_path)
            print(f"  ✓ Completed {pdf_path.stem}")
        except Exception as e:
            print(f"  ✗ Error processing {pdf_path.stem}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✓ All PDFs processed and vector stores created")
    
    # STEP 2: Generate chapter summaries using hierarchical approach
    print("\n" + "="*80)
    print("STEP 2: GENERATING CHAPTER SUMMARIES (Hierarchical)")
    print("="*80)
    print()
    
    cache = SummaryCache()
    total_chapters = 0
    
    for pdf_name, vector_store in sorted(pipeline.vector_stores.items()):
        print(f"\nProcessing chapters from {pdf_name}.pdf...")
        
        # Get all chapters for this PDF
        collection = vector_store.vector_store._collection
        all_data = collection.get(include=['metadatas'])
        
        if not all_data or 'metadatas' not in all_data:
            print(f"  No data found for {pdf_name}")
            continue
        
        # Get unique chapters
        chapters = set()
        for metadata in all_data['metadatas']:
            chapter = metadata.get('chapter')
            if chapter and not pipeline.is_excluded_chapter(chapter):
                chapters.add(chapter)
        
        chapters = sorted(chapters)
        print(f"  Found {len(chapters)} chapters")
        
        # Generate summary for each chapter
        for j, chapter_name in enumerate(chapters, 1):
            print(f"    [{j}/{len(chapters)}] {chapter_name[:60]}...")
            
            # Check cache first
            if cache.has(pdf_name, chapter_name):
                print(f"      ↻ Using cached summary")
                total_chapters += 1
                continue
            
            try:
                # Generate hierarchical summary
                result = pipeline.summarize_chapter_hierarchical(
                    pdf_name=pdf_name,
                    chapter_name=chapter_name,
                    max_levels=5
                )
                
                if result and 'final_summary' in result:
                    summary = result['final_summary']
                    
                    # Save to cache
                    cache.set(pdf_name, chapter_name, summary)
                    cache.save()
                    
                    word_count = len(summary.split())
                    print(f"      ✓ Summary: {word_count} words")
                    total_chapters += 1
                else:
                    print(f"      ✗ Failed to generate summary")
            except Exception as e:
                print(f"      ✗ Error: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n✓ Generated {total_chapters} chapter summaries")
    
    # STEP 3: Generate book summaries using hierarchical approach
    print("\n" + "="*80)
    print("STEP 3: GENERATING BOOK SUMMARIES (Hierarchical)")
    print("="*80)
    print()
    
    # Load cache and group by books
    cache = SummaryCache()
    books_chapters = {}
    
    for key in cache.cache.keys():
        if '::' in key and '::PDF' not in key:
            pdf_name, chapter_name = key.split('::', 1)
            if pdf_name not in books_chapters:
                books_chapters[pdf_name] = []
            books_chapters[pdf_name].append(chapter_name)
    
    print(f"Found {len(books_chapters)} books with chapters")
    
    book_count = 0
    for pdf_name in sorted(books_chapters.keys()):
        chapters = sorted(books_chapters[pdf_name])
        print(f"\n{book_count+1}/{len(books_chapters)}: {pdf_name} ({len(chapters)} chapters)")
        
        # Check if book summary already exists
        if cache.has_pdf_summary(pdf_name):
            print(f"  ↻ Book summary already exists")
            book_count += 1
            continue
        
        # Build hierarchy from chapter summaries
        print(f"  Building hierarchy from {len(chapters)} chapter summaries...")
        
        # Get chapter summaries
        chapter_texts = []
        for chapter in chapters:
            cached = cache.get(pdf_name, chapter)
            if cached:
                chapter_texts.append(f"{chapter}: {cached['summary']}")
        
        if not chapter_texts:
            print(f"  ✗ No chapter summaries found")
            continue
        
        try:
            # Generate embeddings for chapters
            embeddings = pipeline.embedder.embed_texts(chapter_texts)
            
            # Create base chunks
            chunk_dicts = []
            for i, (chapter, text) in enumerate(zip(chapters, chapter_texts)):
                chunk_dicts.append({
                    'text': text,
                    'page': 0,
                    'chapter': chapter,
                    'source': pdf_name
                })
            
            base_chunks = pipeline.hierarchical_chunker.create_base_chunks(chunk_dicts, embeddings)
            
            # Build hierarchy
            hierarchy = pipeline.hierarchical_chunker.build_hierarchy(base_chunks, max_levels=5)
            
            print(f"    Hierarchy built: {len(hierarchy)} levels")
            
            # Add summaries at each level
            def summarizer_fn(text: str, level: int, target_words: int, is_base_level: bool = True) -> str:
                return pipeline._generate_simple_llm_summary(
                    text=text,
                    target_words=target_words,
                    context=f"Book: {pdf_name}, Level {level}",
                    is_base_level=is_base_level
                )
            
            hierarchy = pipeline.hierarchical_chunker.add_summaries(hierarchy, summarizer_fn)
            
            # Get final merged text
            final_text = pipeline.hierarchical_chunker.get_final_summary(hierarchy)
            print(f"    Final text: {len(final_text.split())} words")
            
            # Generate book summary with keywords using LLM
            from regenerate_book_summaries import generate_book_summary_with_keywords
            
            result = generate_book_summary_with_keywords(
                book_title=pdf_name,
                merged_text=final_text,
                pdf_name=pdf_name
            )
            
            if result:
                cache.set_pdf_summary(
                    pdf_name=pdf_name,
                    summary=result['summary'],
                    book_title=pdf_name,
                    num_chapters=len(chapters),
                    keywords=result.get('keywords', []),
                    domain=result.get('domain', '')
                )
                cache.save()
                
                print(f"    ✓ Summary: {len(result['summary'].split())} words")
                print(f"    ✓ Keywords: {', '.join(result.get('keywords', []))}")
                print(f"    ✓ Domain: {result.get('domain', 'N/A')}")
                book_count += 1
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✓ Generated {book_count} book summaries")
    
    # STEP 4: Generate final report
    print("\n" + "="*80)
    print("STEP 4: GENERATING FINAL REPORT")
    print("="*80)
    print()
    
    from generate_all_books_report import generate_all_books_report
    
    report_path = generate_all_books_report()
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"\n✓ Processed {len(pdf_files)} PDFs")
    print(f"✓ Generated {total_chapters} chapter summaries")
    print(f"✓ Generated {book_count} book summaries")
    print(f"✓ Report saved to: {report_path}")
    print()
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
