"""
PDF Embedding Pipeline with Hierarchical Summarization

A production-ready system for processing academic PDFs with intelligent chapter detection,
semantic chunking, vector storage, and hierarchical summarization.

Key Features:
- Processes 22 PDFs in e-library collection
- Smart chapter detection with font size filtering and exclusion patterns
- Semantic chunking with hierarchical merging (similarity threshold 0.75)
- ChromaDB vector storage with comprehensive metadata (chapter, page, font size)
- OpenAI API-based hierarchical summarization (chapter → book level)
- Persistent summary caching for efficient regeneration
- Comprehensive report generation

Processing Pipeline:
1. PDF parsing with heading extraction (font size, level, page metadata)
2. Chapter identification (configurable heading level, font thresholds, exclusions)
3. Semantic chunking with similarity-based merging
4. Vector embedding and ChromaDB storage
5. Hierarchical summarization (chapters → book summary)
6. Report generation and export

Configuration:
- chapter_level_overrides: Specify H1 or H2 as chapter level per PDF
- chapter_font_size_filters: Minimum font size thresholds per PDF
- excluded_chapter_titles: Front/back matter exclusion patterns
- MIN_CHAPTER_WORDS: Minimum word count (default 100)

Author: Gerry Stahl
Date: February 2026
"""

import os
import shutil
import numpy as np
from typing import List, Dict, Any, Tuple
from pathlib import Path
from collections import defaultdict

from parsers.pdf_parser import PDFParser
from chunkers.semantic_chunker import SemanticChunker
from chunkers.semantic_hierarchical_chunker import SemanticHierarchicalChunker
from embedders.embedder import Embedder
from vectorstores.vector_store import VectorStore
from summary_cache import SummaryCache


class PDFEmbeddingPipeline:
    """
    Main pipeline for processing PDFs into searchable vector stores with hierarchical summarization.
    
    This class orchestrates the complete workflow:
    1. PDF parsing with multi-threshold heading detection
    2. Semantic chunking with 6-level hierarchical merging
    3. Embedding generation using sentence transformers
    4. Vector database storage in ChromaDB
    5. LLM-based chapter summarization using local Ollama
    6. Report generation with filtering and caching
    
    Attributes:
        pdf_dir (Path): Directory containing source PDFs
        vector_store_dir (Path): Directory for ChromaDB storage
        parser (PDFParser): PDF parsing component
        chunker (SemanticChunker): Semantic chunking component
        hierarchical_chunker (SemanticHierarchicalChunker): 6-level hierarchical merger
        embedder (Embedder): Embedding generator
        vector_stores (dict): Cache of loaded vector stores
        summary_cache (SummaryCache): Persistent summary storage
        llm_model (str): Ollama model name (default: gemma3:4b)
        ollama_base_url (str): Ollama API endpoint
        
    Example:
        >>> pipeline = PDFEmbeddingPipeline(
        ...     pdf_dir='/path/to/pdfs',
        ...     vector_store_dir='./vector_stores'
        ... )
        >>> pipeline.process_and_report_single_pdf(
        ...     pdf_name='book',
        ...     output_file='report.txt',
        ...     min_word_count=500
        ... )
    """
    
    def __init__(
        self,
        pdf_dir: str,
        vector_store_dir: str = "./vector_stores",
        embedder_type: str = "sentence-transformers",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        llm_model: str = "gemma3:4b",
        ollama_base_url: str = "http://localhost:11434/v1",
    ):
        self.pdf_dir = Path(pdf_dir)
        self.vector_store_dir = Path(vector_store_dir)
        self.parser = PDFParser()
        self.chunker = SemanticChunker(chunk_size=chunk_size, overlap=chunk_overlap)
        self.embedder = Embedder(embedder_type=embedder_type)
        self.hierarchical_chunker = SemanticHierarchicalChunker(
            embedder=self.embedder,
            similarity_threshold=0.75,
            min_chunks_per_level=3,
            max_text_length=4000
        )
        self.vector_stores = {}  # Store vector stores by PDF name
        self.llm_model = llm_model  # Ollama model e.g., 'gemma3:4b'
        self.ollama_base_url = ollama_base_url
        self.chapter_embeddings = {}  # Store chapter-level embeddings: {(pdf_name, chapter_name): embedding}
        self.chapter_metadata = {}  # Store chapter metadata: {(pdf_name, chapter_name): metadata}
        self.summary_cache = SummaryCache()  # Cache for LLM-generated summaries
        
        # Map PDFs to their chapter heading level (default is 1 for H1)
        self.chapter_level_overrides = {
            '9.cscl': 2,  # Use H2 as chapters
            '15.global': 2,  # Use H2 as chapters (with font size filter)
            '17.proposals': 2,  # Use H2 as chapters
        }
        
        # Map PDFs to minimum font size for chapter headings (filters out smaller subsections)
        self.chapter_font_size_filters = {
            '15.global': 18.0,  # Only H2 headings with 18.0pt font (8 language sections, not 111 subsections)
            '17.proposals': 16.0,  # Only H2 headings with >= 16.1pt font (main proposals, not 13.9pt subsections)
        }
        
        # Chapter titles to exclude (common front/back matter)
        self.excluded_chapter_titles = {
            'introduction', 'contents', 'notes', 'references', 'bibliography',
            'vita', 'tables', 'logs', 'figures', 'acknowledgements', 'acknowledgment',
            'preface', 'authors and collaborators', 'notes ….', 'abstract',
            'notes & comments', 'notes on the investigations', 'note',
            'index of names', 'index of terms', 'author index', "author's biography",
            "authors biography",  # without apostrophe for matching
            'notice'
        }
        
    def _get_llm_client(self):
        """Get OpenAI-compatible client configured for Ollama."""
        from openai import OpenAI
        
        # Use Ollama with OpenAI-compatible API
        return OpenAI(
            base_url=self.ollama_base_url,
            api_key="ollama"  # Ollama doesn't need a real key
        )
        
    def is_excluded_chapter(self, chapter_title: str) -> bool:
        """Check if a chapter title should be excluded."""
        import re
        # Normalize: lowercase, strip all whitespace and trailing punctuation
        normalized = chapter_title.lower().strip()
        # Remove all trailing whitespace and punctuation (., …, spaces, etc.)
        normalized = re.sub(r'[\s\.…]+$', '', normalized)
        # Normalize apostrophes (replace curly quotes with regular apostrophe, then remove all apostrophes)
        # \u2019 is the Unicode right single quotation mark (')
        normalized_no_apos = normalized.replace('\u2019', "'").replace("'", "")
        
        # Check exact match against excluded terms
        if normalized in self.excluded_chapter_titles or normalized_no_apos in self.excluded_chapter_titles:
            return True
        
        # Check if normalized title starts with any excluded term followed by space or 's'
        # This catches variations like "Contents of...", "Acknowledgments", etc.
        for excluded in self.excluded_chapter_titles:
            if (normalized == excluded + 's' or normalized.startswith(excluded + ' ') or
                normalized_no_apos == excluded + 's' or normalized_no_apos.startswith(excluded + ' ')):
                return True
        
        # Check if this looks like a table of contents entry
        # TOC entries typically have multiple dots in sequence (e.g., "Chapter 1 ........ 42")
        if re.search(r'\.{3,}', chapter_title):  # 3 or more consecutive dots
            return True
        
        return False
        
    def process_pdfs(self, pdf_files: List[str] = None):
        """Process all PDFs in the directory."""
        if pdf_files is None:
            pdf_files = list(self.pdf_dir.glob("*.pdf"))
        
        for pdf_path in pdf_files:
            print(f"\nProcessing: {pdf_path.name}")
            result = self.process_single_pdf(pdf_path)
            
            # Query this specific PDF's vector store
            print(f"\nQuerying {pdf_path.stem} vector store...")
            self.query_pdf(pdf_path.stem, "What is the main topic discussed?", top_k=3)
    
    def process_single_pdf(self, pdf_path: Path):
        """Process a single PDF through the entire pipeline."""
        pdf_name = pdf_path.stem  # Get filename without extension
        
        # Determine which heading level to use as chapters for this PDF
        chapter_level = self.chapter_level_overrides.get(pdf_name, 1)
        
        # Get minimum font size filter if specified for this PDF
        min_chapter_font_size = self.chapter_font_size_filters.get(pdf_name, None)
        
        # Create/clear vector store for this PDF
        vector_store_path = self.vector_store_dir / pdf_name
        if vector_store_path.exists():
            print(f"  → Clearing existing vector store: {pdf_name}")
            shutil.rmtree(vector_store_path)
        
        # Delete cached summaries for this PDF using the cache's built-in clear method
        self.summary_cache.clear(pdf_name)
        print(f"  → Cleared cached summaries for {pdf_name}")
        
        # Create new vector store for this PDF
        print(f"  → Creating vector store: {pdf_name}")
        vector_store = VectorStore(
            collection_name=pdf_name,
            persist_directory=str(vector_store_path)
        )
        self.vector_stores[pdf_name] = vector_store
        
        # 1. Parse PDF and extract heading structure
        print("  → Parsing PDF and extracting headings...")
        document_data = self.parser.parse_with_headings(pdf_path)
        print(f"     Extracted {len(document_data['text'])} chars from {document_data['metadata']['num_pages']} pages")
        
        # Display document structure outline
        headings = document_data.get('heading_structure', [])
        print(f"     Detected {len(headings)} headings")
        
        print(f"\n{'='*60}")
        print(f"DOCUMENT STRUCTURE OUTLINE: {pdf_path.name}")
        print(f"Using heading level {chapter_level} as chapters")
        print(f"{'='*60}\n")
        
        current_chapter = None
        subsection_count = 0
        chapter_count = 0
        
        for h in headings:
            if h['level'] == chapter_level:
                if current_chapter is not None and subsection_count > 0:
                    print(f"   ({subsection_count} subsections)\n")
                subsection_count = 0
                current_chapter = h['text']
                chapter_count += 1
                print(f"{chapter_count}. {h['text']}")
                print(f"   (Page {h['page']}, Font: {h['font_size']:.1f}pt)")
            elif h['level'] == chapter_level + 1:
                subsection_count += 1
                print(f"   {chapter_count}.{subsection_count}. {h['text'][:70]}")
                print(f"        (Page {h['page']}, Font: {h['font_size']:.1f}pt)")
        
        if subsection_count > 0:
            print(f"   ({subsection_count} subsections)")
        
        print(f"\n{'='*60}")
        print(f"Total: {len([h for h in headings if h['level'] == chapter_level])} chapters, {len([h for h in headings if h['level'] == chapter_level + 1])} subsections")
        print(f"{'='*60}\n")
        
        # 2. Perform semantic chunking with structural metadata
        print("  → Performing semantic chunking...")
        chunks = self.chunker.chunk_with_metadata(
            text=document_data['text'],
            document_data=document_data,
            chapter_level=chapter_level,
            min_chapter_font_size=min_chapter_font_size
        )
        print(f"     Created {len(chunks)} chunks")
        
        # Filter out chapters with too few words (likely dividers/graphics)
        MIN_CHAPTER_WORDS = 100
        chapter_word_counts = {}
        for chunk in chunks:
            chapter = chunk['metadata'].get('chapter')
            if chapter:
                if chapter not in chapter_word_counts:
                    chapter_word_counts[chapter] = 0
                chapter_word_counts[chapter] += len(chunk['text'].split())
        
        # Identify chapters to exclude
        excluded_chapters = {ch for ch, wc in chapter_word_counts.items() if wc < MIN_CHAPTER_WORDS}
        if excluded_chapters:
            print(f"\n     Filtering out {len(excluded_chapters)} short chapters (< {MIN_CHAPTER_WORDS} words):")
            for ch in sorted(excluded_chapters):
                print(f"       - {ch} ({chapter_word_counts[ch]} words)")
            
            # Remove chunks from excluded chapters
            original_count = len(chunks)
            chunks = [c for c in chunks if c['metadata'].get('chapter') not in excluded_chapters]
            print(f"     Removed {original_count - len(chunks)} chunks, {len(chunks)} remaining")
        
        # Display sample chunk metadata
        if chunks:
            print(f"\n     Sample chunk metadata:")
            for i in [0, 10, 50]:
                if i < len(chunks):
                    metadata = chunks[i]['metadata']
                    print(f"\n     Chunk {i}:")
                    print(f"     - Filename: {metadata.get('filename')}")
                    print(f"     - Page: {metadata.get('page_number')}")
                    print(f"     - Chapter: {metadata.get('chapter')}")
                    print(f"     - Section: {metadata.get('section')}")
                    print(f"     - Text preview: {chunks[i]['text'][:80]}...")
        
        # 3. Generate embeddings
        print("  → Generating embeddings...")
        embeddings = self.embedder.embed_chunks(chunks)
        
        # 4. Store in vector database
        print("  → Storing in vector database...")
        vector_store.add_documents(chunks, embeddings)
        
        return {
            'chunks': chunks,
            'embeddings': embeddings
        }
    
    def query_pdf(self, pdf_name: str, query_text: str, top_k: int = 5):
        """Query a specific PDF's vector store."""
        if pdf_name not in self.vector_stores:
            print(f"Error: No vector store found for {pdf_name}")
            return []
        
        results = self.vector_stores[pdf_name].similarity_search(query_text=query_text, top_k=top_k)
        
        # Display results with metadata
        print(f"\n{'='*60}")
        print(f"QUERY RESULTS: {pdf_name}")
        print(f"{'='*60}\n")
        
        for i, result in enumerate(results, 1):
            metadata = result.get('metadata', {})
            score = result.get('score', 0.0)
            text = result.get('text', '')[:100]
            chapter = metadata.get('chapter') or 'N/A'
            section = metadata.get('section') or 'N/A'
            page = metadata.get('page_number', 'N/A')
            
            print(f"{i}. Score: {score:.3f} | Page: {page} | Chapter: {chapter[:40]} | Section: {section[:40]}")
            print(f"   Text: {text}...")
            print()
        
        return results
    
    def query_all(self, query_text: str, top_k: int = 5):
        """Query all PDF vector stores and combine results."""
        all_results = []
        
        for pdf_name, vector_store in self.vector_stores.items():
            results = vector_store.similarity_search(query_text=query_text, top_k=top_k)
            for result in results:
                result['pdf_name'] = pdf_name
                all_results.append(result)
        
        # Sort by score (descending - higher scores are better)
        all_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Display results
        print(f"\n{'='*60}")
        print(f"COMBINED QUERY RESULTS")
        print(f"{'='*60}\n")
        
        for i, result in enumerate(all_results[:top_k], 1):
            metadata = result.get('metadata', {})
            score = result.get('score', 0.0)
            text = result.get('text', '')[:100]
            chapter = metadata.get('chapter') or 'N/A'
            section = metadata.get('section') or 'N/A'
            page = metadata.get('page_number', 'N/A')
            pdf_name = result.get('pdf_name', 'N/A')
            
            print(f"{i}. Score: {score:.3f} | PDF: {pdf_name} | Page: {page}")
            print(f"   Chapter: {chapter[:50]} | Section: {section[:50]}")
            print(f"   Text: {text}...")
            print()
        
        return all_results[:top_k]
    
    def summarize_pdf(self, pdf_name: str, max_chunks: int = None) -> str:
        """
        Generate a summary of an entire PDF by aggregating its content.
        
        Args:
            pdf_name: Name of the PDF (without .pdf extension)
            max_chunks: Maximum number of chunks to include (None = all chunks)
            
        Returns:
            Summary text including structure and key content
        """
        # Get the document data from processing
        pdf_path = self.pdf_dir / f"{pdf_name}.pdf"
        if not pdf_path.exists():
            return f"Error: PDF '{pdf_name}' not found"
        
        # Parse to get structure
        document_data = self.parser.parse_with_headings(pdf_path, skip_pages=4)
        
        # Build summary
        summary_parts = []
        summary_parts.append(f"SUMMARY OF: {pdf_name}.pdf")
        summary_parts.append("=" * 60)
        summary_parts.append(f"\nTotal Pages: {document_data.get('metadata', {}).get('num_pages', 0)}")
        summary_parts.append(f"Total Characters: {len(document_data['text'])}")
        
        # Add structure overview
        headings = document_data.get('heading_structure', [])
        h1_count = sum(1 for h in headings if h.get('level') == 1)
        h2_count = sum(1 for h in headings if h.get('level') == 2)
        summary_parts.append(f"Chapters (H1): {h1_count}")
        summary_parts.append(f"Sections (H2): {h2_count}")
        
        # Add chapter listing
        summary_parts.append("\n" + "=" * 60)
        summary_parts.append("CHAPTER OUTLINE:")
        summary_parts.append("=" * 60)
        for heading in headings:
            if heading.get('level') == 1:
                summary_parts.append(f"\n• {heading['text']} (Page {heading['page']})")
        
        # Add content sample from beginning
        summary_parts.append("\n" + "=" * 60)
        summary_parts.append("CONTENT OVERVIEW:")
        summary_parts.append("=" * 60)
        
        # Get chunks to sample content
        if pdf_name in self.vector_stores:
            # Use semantic search to find most representative chunks
            # Query with the document's own chapter titles to find key content
            query_text = ""
            if h1_count > 0:
                first_chapters = [h['text'] for h in headings if h.get('level') == 1][:3]
                query_text = " ".join(first_chapters).strip()
            
            # Fallback to document title/topic if no query text
            if not query_text:
                query_text = pdf_name.replace("-", " ").replace("_", " ")
            
            if query_text:  # Make sure we have text to query
                results = self.vector_stores[pdf_name].similarity_search(query_text=query_text, top_k=5)
                
                summary_parts.append("\nKey content excerpts:")
                for i, result in enumerate(results, 1):
                    metadata = result.get('metadata', {})
                    chapter = metadata.get('chapter', 'N/A')
                    text = result.get('text', '')[:200]
                    summary_parts.append(f"\n{i}. From chapter: {chapter}")
                    summary_parts.append(f"   {text}...")
            else:
                summary_parts.append("\nDocument opening:")
                summary_parts.append(document_data['text'][:500] + "...")
        else:
            # Just show beginning of document
            summary_parts.append("\nDocument opening:")
            summary_parts.append(document_data['text'][:500] + "...")
        
        return "\n".join(summary_parts)
    
    def write_chapter_summary(self, pdf_name: str, chapter_name: str, output_file: str = None, use_llm: bool = True) -> str:
        """
        Generate a brief written summary of a specific chapter.
        
        Args:
            pdf_name: Name of the PDF (without .pdf extension)
            chapter_name: Name of the chapter to summarize (H1 heading text)
            output_file: Optional file path to save the summary (if None, just returns it)
            use_llm: If True, use LLM to generate summary; if False, extract sentences
            
        Returns:
            Brief summary text for the chapter
        """
        if pdf_name not in self.vector_stores:
            return f"Error: PDF '{pdf_name}' not found in vector stores. Process it first."
        
        vector_store = self.vector_stores[pdf_name]
        
        # Get all chunks for this chapter
        results = vector_store.similarity_search(query_text=chapter_name, top_k=50)
        
        # Filter to only chunks from this chapter
        chapter_chunks = []
        for result in results:
            metadata = result.get('metadata', {})
            if metadata.get('chapter') and chapter_name.lower() in metadata['chapter'].lower():
                chapter_chunks.append(result)
        
        if not chapter_chunks:
            return f"Error: No content found for chapter '{chapter_name}' in {pdf_name}"
        
        # Sort by page number and position
        chapter_chunks.sort(key=lambda x: (
            x.get('metadata', {}).get('page_number', 999),
            x.get('metadata', {}).get('start_pos', 0)
        ))
        
        # Build brief summary
        summary_lines = []
        summary_lines.append(f"Brief Summary: {chapter_name}")
        summary_lines.append(f"From: {pdf_name}.pdf")
        summary_lines.append("=" * 60)
        summary_lines.append("")
        
        # Get chapter metadata
        first_chunk = chapter_chunks[0]
        metadata = first_chunk.get('metadata', {})
        start_page = metadata.get('page_number', 'N/A')
        
        # Get sections
        sections = []
        seen_sections = set()
        for chunk in chapter_chunks:
            section = chunk.get('metadata', {}).get('section')
            if section and section not in seen_sections:
                sections.append(section)
                seen_sections.add(section)
        
        # Write overview
        summary_lines.append(f"Chapter: {chapter_name}")
        summary_lines.append(f"Starting Page: {start_page}")
        if sections:
            summary_lines.append(f"Main Sections: {len(sections)}")
            for section in sections[:5]:  # Show first 5 sections
                summary_lines.append(f"  • {section}")
        summary_lines.append("")
        
        summary_lines.append("SUMMARY:")
        summary_lines.append("-" * 60)
        summary_lines.append("")
        
        if use_llm:
            # Generate summary using LLM
            generated_summary = self._generate_llm_summary(chapter_name, chapter_chunks)
            summary_lines.append(generated_summary)
        else:
            # Extract key sentences (original approach)
            num_chunks = len(chapter_chunks)
            key_indices = []
            
            if num_chunks <= 3:
                key_indices = list(range(num_chunks))
            else:
                # Beginning (first 2)
                key_indices.extend([0, 1])
                # Middle (1-2 chunks)
                key_indices.append(num_chunks // 2)
                if num_chunks > 6:
                    key_indices.append(num_chunks // 2 + 1)
                # End (last 2)
                key_indices.extend([num_chunks - 2, num_chunks - 1])
            
            # Generate narrative summary from key chunks
            for idx in key_indices:
                chunk = chapter_chunks[idx]
                text = chunk.get('text', '').strip()
                
                # Extract first 2-3 sentences from the chunk
                sentences = text.split('. ')
                key_sentences = '. '.join(sentences[:2])
                if key_sentences and not key_sentences.endswith('.'):
                    key_sentences += '.'
                
                if key_sentences:
                    summary_lines.append(key_sentences)
                    summary_lines.append("")
        
        summary_lines.append("")
        summary_lines.append("-" * 60)
        summary_lines.append(f"Total chapter length: {len(chapter_chunks)} chunks")
        
        summary_text = "\n".join(summary_lines)
        
        # Write to file if specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(summary_text)
            print(f"\nSummary written to: {output_path}")
        
        return summary_text
    
    def _generate_llm_summary(self, chapter_name: str, chapter_chunks: List[Dict], target_words: int = 100, pdf_name: str = None) -> str:
        """
        Generate a coherent summary using an LLM with hierarchical summarization.
        Uses caching to avoid regenerating summaries.
        
        This uses a hierarchical approach:
        1. Summarize individual chunks
        2. Group and summarize those summaries
        3. Final summary at target word count
        
        Args:
            chapter_name: Name of the chapter
            chapter_chunks: List of chunk dictionaries from the chapter
            target_words: Target word count for final summary (default: 100)
            pdf_name: PDF name for caching (optional)
            
        Returns:
            Generated summary text
        """
        # Check cache first if pdf_name provided
        if pdf_name and self.summary_cache.has(pdf_name, chapter_name):
            cached_data = self.summary_cache.get(pdf_name, chapter_name)
            return cached_data['summary'] if cached_data else None
        
        try:
            from openai import OpenAI
            import os
            
            # Use API key from instance or environment
            api_key = self.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                # Fallback to extractive summary
                texts = [chunk.get('text', '')[:300] for chunk in chapter_chunks[:5]]
                summary = " ".join(texts)
                words = summary.split()
                return " ".join(words[:target_words]) + "..."
            
            client = self._get_llm_client()
            
            # Step 1: Summarize individual chunks (if more than 10 chunks)
            chunk_summaries = []
            if len(chapter_chunks) > 10:
                # Process chunks in batches to avoid too many AP
                # I calls
                batch_size = 3
                for i in range(0, min(30, len(chapter_chunks)), batch_size):
                    batch = chapter_chunks[i:i+batch_size]
                    batch_text = "\n\n".join([chunk.get('text', '')[:1000] for chunk in batch])
                    
                    if not batch_text.strip():
                        continue
                    
                    response = client.chat.completions.create(
                        model=self.llm_model,
                        messages=[
                            {"role": "system", "content": "Summarize the following text section."},
                            {"role": "user", "content": batch_text}
                        ],
                        temperature=0.3,
                        max_tokens=150
                    )
                    chunk_summaries.append(response.choices[0].message.content.strip())
                
                # Step 2: Combine chunk summaries
                combined_summary = "\n\n".join(chunk_summaries)
            else:
                # For shorter chapters, use chunks directly
                combined_summary = "\n\n".join([chunk.get('text', '')[:800] for chunk in chapter_chunks[:15]])
            
            # Step 3: Generate final summary at target word count
            prompt = f"""Write a {target_words}-word summary of this chapter titled "{chapter_name}".

The summary should:
- Be approximately {target_words} words
- Capture the main arguments and key insights
- Be written in clear, flowing prose
- Focus on the most important points

Content:

{combined_summary[:8000]}

{target_words}-word summary:"""
            
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a skilled academic summarizer who creates concise, insightful summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=int(target_words * 2)  # Allow some buffer
            )
            
            summary = response.choices[0].message.content.strip()
            
            # Ensure we're close to target word count
            words = summary.split()
            if len(words) > target_words * 1.3:
                summary = " ".join(words[:target_words]) + "..."
            
            # Cache the summary if pdf_name provided
            if pdf_name:
                self.summary_cache.set(pdf_name, chapter_name, summary)
                self.summary_cache.save()
            
            return summary
            
        except ImportError:
            # Fallback to extractive summary
            texts = [chunk.get('text', '')[:300] for chunk in chapter_chunks[:5]]
            summary = " ".join(texts)
            words = summary.split()
            return " ".join(words[:target_words]) + "..."
        except Exception as e:
            # Fallback to extractive summary on error
            print(f"    WARNING: LLM summary failed ({str(e)}), using extractive summary")
            texts = [chunk.get('text', '')[:300] for chunk in chapter_chunks[:5]]
            summary = " ".join(texts)
            words = summary.split()
            return " ".join(words[:target_words]) + "..."
    
    def _generate_simple_llm_summary(self, text: str, target_words: int = 100, context: str = "", is_base_level: bool = True) -> str:
        """
        Generate a simple LLM summary from text.
        
        Args:
            text: Text to summarize
            target_words: Target word count
            context: Optional context description
            is_base_level: Not used anymore, kept for compatibility
            
        Returns:
            Summary string
        """
        try:
            client = self._get_llm_client()
            
            # Truncate input text if too long
            max_input_chars = 8000
            truncated_text = text[:max_input_chars]
            
            # Calculate word range (±10%)
            min_words = int(target_words * 0.9)
            max_words = int(target_words * 1.1)
            
            prompt = f"""Write a summary of approximately {target_words} words (between {min_words}-{max_words} words) of the following text.

Context: {context if context else 'academic content'}

Guidelines:
- Be an academic commentator, not just a summarizer
- Summarize the text into its main ideas, reducing detail while preserving the key points
- Extract the most critical insights and themes from this text
- Try to suggest why this writing might be interesting, innovative, useful, informative, or a contribution to related literature
- Write in clear, flowing prose
- Be concise yet insightful
- If the text is clearly structured around a small number of distinct points or topics, consider reflecting that organization in the summary (e.g., through enumeration or listing of key topics)
- IMPORTANT: Complete all sentences. Do not end mid-sentence.
- CRITICAL: Only use names and technical terms that appear in the original text. Do not introduce names, concepts, or terminology not present in the source material.

Style requirements:
- Do not repeat the title of the chapter or pdf
- Do not use first names or initials of authors in the summary
- Do not name the author in the summary if it is just "Gerry Stahl" or "G. Stahl"
- Use masculine pronouns for a single author ("he", "his", "him", not "their")
- Try to avoid using the exact words from the text in the summary; paraphrase creatively
- Try to use less common, more sophisticated vocabulary where appropriate
- Instead of referring to "this chapter", use varied terms like "paper" (especially in pdfs 16, 3, 4), "essay" (especially in pdfs 7, 8, 9, 10, 11, 12), "investigation" (especially in pdf 19), "analysis", "discussion", "section", "text", "writing", etc.

Text:

{truncated_text}

{target_words}-word summary:"""
            
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "You are a skilled academic summarizer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=int(target_words * 2)
            )
            
            summary = response.choices[0].message.content.strip()
            
            # Ensure close to target - truncate at sentence boundary if needed
            words = summary.split()
            if len(words) > target_words * 1.3:
                # Find last sentence boundary before the limit
                truncated = " ".join(words[:int(target_words * 1.2)])
                # Look for last period, exclamation, or question mark
                last_period = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
                if last_period > len(truncated) * 0.7:  # If we found one in the last 30%
                    summary = truncated[:last_period + 1]
                else:
                    # Fallback: cut at word boundary
                    summary = " ".join(words[:target_words]) + "..."
            
            return summary
            
        except Exception as e:
            # Fallback
            words = text.split()[:target_words]
            return " ".join(words) + "..."
    
    def _generate_hierarchical_pdf_summary(self, pdf_name: str, chapter_summaries: List[str], target_words: int = 300) -> str:
        """
        Generate a PDF-level summary from chapter summaries using hierarchical approach.
        
        Args:
            pdf_name: Name of the PDF
            chapter_summaries: List of chapter summaries to synthesize
            target_words: Target word count for PDF summary (default 300 words)
            
        Returns:
            PDF-level summary
        """
        try:
            from openai import OpenAI
            import os
            
            api_key = self.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                # Fallback: concatenate first few chapter summaries
                combined = " ".join(chapter_summaries[:5])
                words = combined.split()
                return " ".join(words[:target_words]) + "..."
            
            client = OpenAI(api_key=api_key)
            
            # Combine chapter summaries
            combined_summaries = "\n\n".join(chapter_summaries)
            
            # Calculate word range (±10%)
            min_words = int(target_words * 0.9)
            max_words = int(target_words * 1.1)
            
            prompt = f"""Write an overview of approximately {target_words} words (between {min_words}-{max_words} words) of the book/document "{pdf_name}" based on these chapter summaries.

The overview should:
- Be approximately {target_words} words
- Synthesize the main themes across chapters
- Highlight the overall arc and key contributions
- Be written in clear, engaging prose
- IMPORTANT: Complete all sentences. Do not end mid-sentence.
- CRITICAL: Only use names and technical terms that appear in the chapter summaries. Do not introduce names, concepts, or terminology not present in the source material.

Chapter summaries:

{combined_summaries[:10000]}

{target_words}-word overview:"""
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a skilled academic summarizer who creates insightful overviews of scholarly works."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=int(target_words * 2)
            )
            
            summary = response.choices[0].message.content.strip()
            
            # Ensure we're close to target - truncate at sentence boundary if needed
            words = summary.split()
            if len(words) > target_words * 1.3:
                # Find last sentence boundary before the limit
                truncated = " ".join(words[:int(target_words * 1.2)])
                # Look for last period, exclamation, or question mark
                last_period = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
                if last_period > len(truncated) * 0.7:  # If we found one in the last 30%
                    summary = truncated[:last_period + 1]
                else:
                    # Fallback: cut at word boundary
                    summary = " ".join(words[:target_words]) + "..."
            
            return summary
            
        except Exception as e:
            # Fallback
            combined = " ".join(chapter_summaries[:5])
            words = combined.split()
            return " ".join(words[:target_words]) + "..."
    
    def summarize_chapter(self, pdf_name: str, chapter_name: str) -> str:
        """
        Generate a summary of a specific chapter by aggregating its chunks.
        
        Args:
            pdf_name: Name of the PDF (without .pdf extension)
            chapter_name: Name of the chapter to summarize (H1 heading text)
            
        Returns:
            Summary text for the chapter
        """
        if pdf_name not in self.vector_stores:
            return f"Error: PDF '{pdf_name}' not found in vector stores. Process it first."
        
        vector_store = self.vector_stores[pdf_name]
        
        # Get all chunks for this PDF
        # We'll search for chunks matching the chapter name
        results = vector_store.similarity_search(query_text=chapter_name, top_k=50)
        
        # Filter to only chunks from this chapter
        chapter_chunks = []
        for result in results:
            metadata = result.get('metadata', {})
            if metadata.get('chapter') and chapter_name.lower() in metadata['chapter'].lower():
                chapter_chunks.append(result)
        
        if not chapter_chunks:
            return f"Error: No content found for chapter '{chapter_name}' in {pdf_name}"
        
        # Build summary
        summary_parts = []
        summary_parts.append(f"CHAPTER SUMMARY: {chapter_name}")
        summary_parts.append(f"PDF: {pdf_name}.pdf")
        summary_parts.append("=" * 60)
        
        # Get chapter metadata from first chunk
        first_chunk = chapter_chunks[0]
        metadata = first_chunk.get('metadata', {})
        summary_parts.append(f"\nChapter: {metadata.get('chapter', 'N/A')}")
        summary_parts.append(f"Starting Page: {metadata.get('page_number', 'N/A')}")
        summary_parts.append(f"Number of chunks: {len(chapter_chunks)}")
        
        # Get sections within this chapter
        sections = set()
        for chunk in chapter_chunks:
            section = chunk.get('metadata', {}).get('section')
            if section:
                sections.add(section)
        
        if sections:
            summary_parts.append(f"\nSections in this chapter:")
            for section in sorted(sections):
                summary_parts.append(f"  • {section}")
        
        # Show content overview
        summary_parts.append("\n" + "=" * 60)
        summary_parts.append("CONTENT EXCERPTS:")
        summary_parts.append("=" * 60)
        
        # Show first few chunks to give content overview
        num_excerpts = min(5, len(chapter_chunks))
        for i in range(num_excerpts):
            chunk = chapter_chunks[i]
            text = chunk.get('text', '')[:300]
            page = chunk.get('metadata', {}).get('page_number', 'N/A')
            summary_parts.append(f"\n{i+1}. Page {page}:")
            summary_parts.append(f"   {text}...")
        
        return "\n".join(summary_parts)
    
    def list_chapters(self, pdf_name: str) -> List[str]:
        """
        List all chapters (H1 headings) in a PDF.
        
        Args:
            pdf_name: Name of the PDF (without .pdf extension)
            
        Returns:
            List of chapter names
        """
        pdf_path = self.pdf_dir / f"{pdf_name}.pdf"
        if not pdf_path.exists():
            print(f"Error: PDF '{pdf_name}' not found")
            return []
        
        document_data = self.parser.parse_with_headings(pdf_path, skip_pages=4)
        headings = document_data.get('heading_structure', [])
        
        chapters = [h['text'] for h in headings if h.get('level') == 1]
        
        print(f"\nChapters in {pdf_name}.pdf:")
        print("=" * 60)
        for i, chapter in enumerate(chapters, 1):
            print(f"{i}. {chapter}")
        
        return chapters
    
    def summarize_chapter_hierarchical(
        self,
        pdf_name: str,
        chapter_name: str,
        max_levels: int = 5
    ) -> Dict[str, Any]:
        """
        Generate hierarchical summary using semantic similarity-based merging.
        
        This method:
        1. Retrieves base chunks for the chapter
        2. Computes embeddings for each chunk
        3. Iteratively merges most similar adjacent chunks
        4. Generates summaries at each hierarchy level
        5. Returns complete hierarchical structure
        
        Args:
            pdf_name: Name of the PDF (without .pdf extension)
            chapter_name: Name of the chapter
            max_levels: Maximum hierarchy depth
            
        Returns:
            Dictionary with hierarchy info and final summary
        """
        import logging
        logger = logging.getLogger(__name__)
        
        print(f"\n{'='*60}")
        print(f"SEMANTIC HIERARCHICAL SUMMARIZATION")
        print(f"PDF: {pdf_name}")
        print(f"Chapter: {chapter_name}")
        print(f"{'='*60}")
        
        # Get vector store for this PDF
        vector_store = self.vector_stores.get(pdf_name)
        if not vector_store:
            print(f"Error: No vector store found for {pdf_name}")
            return {}
        
        # Retrieve all chunks for this chapter
        collection = vector_store.vector_store._collection
        results = collection.get()
        
        # Filter chunks for this chapter
        chapter_chunks = []
        chapter_embeddings = []
        
        for idx, metadata in enumerate(results['metadatas']):
            if metadata.get('chapter') == chapter_name:
                chunk_dict = {
                    'text': results['documents'][idx],
                    'page': metadata.get('page', 0),
                    'chapter': chapter_name,
                    'source': pdf_name,
                    'chunk_index': idx
                }
                chapter_chunks.append(chunk_dict)
                
                # Get embedding from collection
                if 'embeddings' in results and results['embeddings']:
                    chapter_embeddings.append(np.array(results['embeddings'][idx]))
                else:
                    # Generate if not available
                    emb = self.embedder.embed_texts([chunk_dict['text']])[0]
                    chapter_embeddings.append(emb)
        
        print(f"Found {len(chapter_chunks)} base chunks")
        
        if len(chapter_chunks) == 0:
            return {
                'error': 'No chunks found for this chapter',
                'hierarchy': [],
                'final_summary': ''
            }
        
        # Create base hierarchical chunks
        base_chunks = self.hierarchical_chunker.create_base_chunks(
            chapter_chunks,
            chapter_embeddings
        )
        
        # Build hierarchy through iterative merging
        print(f"Building hierarchy (max {max_levels} levels)...")
        hierarchy = self.hierarchical_chunker.build_hierarchy(
            base_chunks,
            max_levels=max_levels
        )
        
        # Create summarizer function that uses our LLM
        def summarizer_fn(text: str, level: int, target_words: int, is_base_level: bool = True) -> str:
            """Generate summary for a chunk at given hierarchy level."""
            return self._generate_simple_llm_summary(
                text=text,
                target_words=target_words,
                context=f"hierarchy level {level}",
                is_base_level=is_base_level
            )
        
        # Add summaries at each level
        print("Generating summaries at each level...")
        hierarchy = self.hierarchical_chunker.add_summaries(
            hierarchy,
            summarizer_fn
        )
        
        # Get final summary (may be concatenated if multiple top-level chunks)
        final_summary = self.hierarchical_chunker.get_final_summary(hierarchy)
        
        # If there are multiple top-level chunks, generate a final chapter-level summary
        top_level_chunks = hierarchy[-1] if hierarchy else []
        if len(top_level_chunks) > 1:
            print(f"\nGenerating final chapter-level summary from {len(top_level_chunks)} top-level summaries...")
            
            # Combine all top-level summaries
            combined_summaries = "\n\n".join([
                chunk.summary for chunk in top_level_chunks if chunk.summary
            ])
            
            # Generate one final summary for the entire chapter
            final_summary = self._generate_simple_llm_summary(
                text=combined_summaries,
                target_words=180,  # 160-200 words for chapter summary (180 ±10% = 162-198)
                context=f"final chapter summary for {chapter_name}"
            )
            
            print(f"✓ Generated final chapter summary ({len(final_summary.split())} words)")
        
        # Export hierarchy info
        hierarchy_info = self.hierarchical_chunker.export_hierarchy_info(hierarchy)
        
        print(f"\n{'='*60}")
        print(f"HIERARCHY SUMMARY")
        print(f"{'='*60}")
        for level_info in hierarchy_info['levels']:
            print(f"Level {level_info['level']}: "
                  f"{level_info['num_chunks']} chunks, "
                  f"avg {level_info['avg_text_length']:.0f} chars, "
                  f"avg {level_info['avg_summary_length']:.0f} word summary")
        print(f"\n{'='*60}")
        print(f"FINAL SUMMARY ({len(final_summary.split())} words):")
        print(f"{'='*60}")
        print(final_summary)
        print(f"{'='*60}\n")
        
        return {
            'hierarchy': hierarchy,
            'hierarchy_info': hierarchy_info,
            'final_summary': final_summary,
            'num_levels': len(hierarchy),
            'base_chunks': len(chapter_chunks)
        }
    
    def create_chapter_embeddings(self):
        """
        Create embeddings for each chapter across all processed PDFs.
        This generates a single vector representation for each chapter by averaging
        the embeddings of all chunks in that chapter.
        """
        print("\n" + "="*60)
        print("CREATING CHAPTER-LEVEL EMBEDDINGS")
        print("="*60)
        
        self.chapter_embeddings = {}
        self.chapter_metadata = {}
        
        for pdf_name, vector_store in self.vector_stores.items():
            print(f"\nProcessing chapters from {pdf_name}.pdf...")
            
            # Get all chunks from this PDF's vector store
            # We'll need to access the underlying collection
            if hasattr(vector_store.vector_store, '_collection'):
                collection = vector_store.vector_store._collection
                all_data = collection.get(include=['embeddings', 'metadatas', 'documents'])
                
                if all_data is None or 'embeddings' not in all_data or len(all_data['embeddings']) == 0:
                    print(f"  No data found for {pdf_name}")
                    continue
                
                # Group chunks by chapter
                chapter_chunks = defaultdict(list)
                for i, (embedding, metadata, text) in enumerate(zip(
                    all_data['embeddings'],
                    all_data['metadatas'],
                    all_data['documents']
                )):
                    chapter = metadata.get('chapter')
                    if chapter:
                        chapter_chunks[chapter].append({
                            'embedding': np.array(embedding),
                            'metadata': metadata,
                            'text': text
                        })
                
                # Create chapter-level embeddings by averaging chunk embeddings
                for chapter_name, chunks in chapter_chunks.items():
                    if not chunks:
                        continue
                    
                    # Skip excluded chapter titles
                    if self.is_excluded_chapter(chapter_name):
                        print(f"  ⊗ Skipping excluded chapter: {chapter_name}")
                        continue
                    
                    # Average all chunk embeddings for this chapter
                    embeddings = np.array([chunk['embedding'] for chunk in chunks])
                    chapter_embedding = np.mean(embeddings, axis=0)
                    
                    # Normalize the embedding
                    norm = np.linalg.norm(chapter_embedding)
                    if norm > 0:
                        chapter_embedding = chapter_embedding / norm
                    
                    # Store the chapter embedding and metadata
                    key = (pdf_name, chapter_name)
                    self.chapter_embeddings[key] = chapter_embedding
                    
                    # Store metadata
                    first_chunk = chunks[0]['metadata']
                    self.chapter_metadata[key] = {
                        'pdf_name': pdf_name,
                        'chapter_name': chapter_name,
                        'num_chunks': len(chunks),
                        'start_page': first_chunk.get('page_number', 'N/A'),
                        'filename': first_chunk.get('filename', 'N/A')
                    }
                    
                    print(f"  ✓ {chapter_name} ({len(chunks)} chunks)")
        
        print(f"\n✓ Created embeddings for {len(self.chapter_embeddings)} chapters")
        return self.chapter_embeddings
    
    def compute_chapter_similarities(self) -> Dict[Tuple[str, str], float]:
        """
        Compute pairwise cosine similarities between all chapters.
        
        Returns:
            Dictionary mapping (chapter1_key, chapter2_key) -> similarity_score
        """
        if not self.chapter_embeddings:
            print("Error: No chapter embeddings found. Run create_chapter_embeddings() first.")
            return {}
        
        print("\n" + "="*60)
        print("COMPUTING CHAPTER SIMILARITIES")
        print("="*60)
        
        similarities = {}
        chapters = list(self.chapter_embeddings.keys())
        
        for i, ch1 in enumerate(chapters):
            for j, ch2 in enumerate(chapters):
                if i < j:  # Only compute upper triangle
                    emb1 = self.chapter_embeddings[ch1]
                    emb2 = self.chapter_embeddings[ch2]
                    
                    # Cosine similarity (embeddings are already normalized)
                    similarity = np.dot(emb1, emb2)
                    similarities[(ch1, ch2)] = float(similarity)
        
        # Show top similarities
        print(f"\n✓ Computed {len(similarities)} pairwise similarities")
        print("\nTop 10 most similar chapter pairs:")
        print("-" * 60)
        
        sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        for (ch1, ch2), sim in sorted_sims[:10]:
            pdf1, name1 = ch1
            pdf2, name2 = ch2
            print(f"{sim:.3f} | {pdf1}: {name1[:40]}")
            print(f"      | {pdf2}: {name2[:40]}")
            print()
        
        return similarities
    
    def cluster_chapters(self, n_clusters: int = 5, method: str = 'kmeans') -> Dict[str, Any]:
        """
        Cluster chapters based on their embeddings.
        
        Args:
            n_clusters: Number of clusters to create
            method: Clustering method ('kmeans', 'hierarchical', or 'dbscan')
            
        Returns:
            Dictionary with cluster assignments and statistics
        """
        if not self.chapter_embeddings:
            print("Error: No chapter embeddings found. Run create_chapter_embeddings() first.")
            return {}
        
        try:
            from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
            from sklearn.metrics import silhouette_score
        except ImportError:
            print("Error: scikit-learn not installed. Install with: pip install scikit-learn")
            return {}
        
        print("\n" + "="*60)
        print(f"CLUSTERING CHAPTERS ({method.upper()}, k={n_clusters})")
        print("="*60)
        
        # Prepare data
        chapters = list(self.chapter_embeddings.keys())
        embeddings = np.array([self.chapter_embeddings[ch] for ch in chapters])
        
        # Perform clustering
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = clusterer.fit_predict(embeddings)
        elif method == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            labels = clusterer.fit_predict(embeddings)
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.3, min_samples=2)
            labels = clusterer.fit_predict(embeddings)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        else:
            print(f"Unknown clustering method: {method}")
            return {}
        
        # Calculate silhouette score
        if len(set(labels)) > 1:
            sil_score = silhouette_score(embeddings, labels)
        else:
            sil_score = 0.0
        
        # Organize results
        clusters = defaultdict(list)
        for chapter, label in zip(chapters, labels):
            clusters[int(label)].append(chapter)
        
        # Display results
        print(f"\n✓ Created {n_clusters} clusters")
        print(f"Silhouette Score: {sil_score:.3f} (higher is better, range: -1 to 1)")
        print("\n" + "="*60)
        print("CLUSTER ASSIGNMENTS")
        print("="*60)
        
        for cluster_id in sorted(clusters.keys()):
            if cluster_id == -1:
                print(f"\nCluster {cluster_id} (NOISE - {len(clusters[cluster_id])} chapters):")
            else:
                print(f"\nCluster {cluster_id} ({len(clusters[cluster_id])} chapters):")
            print("-" * 60)
            
            for pdf_name, chapter_name in clusters[cluster_id]:
                metadata = self.chapter_metadata.get((pdf_name, chapter_name), {})
                page = metadata.get('start_page', 'N/A')
                print(f"  • [{pdf_name}] {chapter_name[:50]}... (p.{page})")
        
        # Return results
        result = {
            'method': method,
            'n_clusters': n_clusters,
            'silhouette_score': sil_score,
            'clusters': dict(clusters),
            'labels': labels.tolist(),
            'chapters': chapters
        }
        
        return result
    
    def save_chapter_embeddings(self, output_file: str = "./chapter_embeddings.npz"):
        """Save chapter embeddings to file for later use."""
        if not self.chapter_embeddings:
            print("Error: No chapter embeddings to save")
            return
        
        # Prepare data for saving
        chapters = list(self.chapter_embeddings.keys())
        embeddings = np.array([self.chapter_embeddings[ch] for ch in chapters])
        
        # Save
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez(
            output_path,
            embeddings=embeddings,
            chapters=chapters,
            metadata=[self.chapter_metadata[ch] for ch in chapters]
        )
        
        print(f"\n✓ Saved {len(chapters)} chapter embeddings to {output_path}")
    
    def save_chapters_by_pdf_report(self, output_file: str = "./reports/chapters_by_pdf.txt", use_llm: bool = True):
        """
        Save a report listing all chapters for each PDF with hierarchical LLM summaries.
        
        Args:
            output_file: Path to save the report
            use_llm: If True, use hierarchical LLM summarization; if False, use extractive
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Group chapters by PDF
        chapters_by_pdf = defaultdict(list)
        for chapter_key in self.chapter_embeddings.keys():
            pdf_name, chapter_name = chapter_key
            metadata = self.chapter_metadata.get(chapter_key, {})
            chapters_by_pdf[pdf_name].append({
                'key': chapter_key,
                'name': chapter_name,
                'page': metadata.get('start_page', 'N/A'),
                'chunks': metadata.get('num_chunks', 0)
            })
        
        total_chapters = len(self.chapter_embeddings)
        print(f"Generating hierarchical summaries for {total_chapters} chapters across {len(chapters_by_pdf)} PDFs...")
        if use_llm:
            print("Using LLM-based hierarchical summarization (this may take several minutes)...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("CHAPTERS BY PDF (with Hierarchical Summaries)\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total PDFs: {len(chapters_by_pdf)}\n")
            f.write(f"Total Chapters: {total_chapters}\n\n")
            
            # Process each PDF
            pdf_counter = 0
            for pdf_name in sorted(chapters_by_pdf.keys()):
                pdf_counter += 1
                chapters = chapters_by_pdf[pdf_name]
                print(f"\nProcessing {pdf_counter}/{len(chapters_by_pdf)}: {pdf_name} ({len(chapters)} chapters)")
                
                f.write("="*80 + "\n")
                f.write(f"{pdf_name}.pdf ({len(chapters)} chapters)\n")
                f.write("="*80 + "\n\n")
                
                # Collect chapter summaries for PDF-level summary
                chapter_summaries_for_pdf = []
                
                # Process each chapter
                for i, chapter in enumerate(chapters, 1):
                    chapter_name = chapter['name']
                    print(f"  Chapter {i}/{len(chapters)}: {chapter_name[:60]}...")
                    
                    f.write(f"{i}. {chapter_name}\n")
                    f.write(f"   Page: {chapter['page']}, Chunks: {chapter['chunks']}\n")
                    
                    # Get chapter chunks from vector store
                    chapter_summary = ""
                    if pdf_name in self.vector_stores and use_llm:
                        vector_store = self.vector_stores[pdf_name]
                        
                        # Get all chunks for this chapter
                        try:
                            if hasattr(vector_store.vector_store, '_collection'):
                                collection = vector_store.vector_store._collection
                                all_data = collection.get(include=['metadatas', 'documents'])
                                
                                if all_data and 'metadatas' in all_data and 'documents' in all_data:
                                    chapter_chunks = []
                                    for metadata, text in zip(all_data['metadatas'], all_data['documents']):
                                        if metadata.get('chapter') == chapter_name:
                                            chapter_chunks.append({
                                                'text': text,
                                                'metadata': metadata
                                            })
                                    
                                    # Sort by page and position
                                    chapter_chunks.sort(key=lambda x: (
                                        x['metadata'].get('page_number', 999),
                                        x['metadata'].get('start_pos', 0)
                                    ))
                                    
                                    if chapter_chunks:
                                        # Generate hierarchical summary
                                        chapter_summary = self._generate_llm_summary(
                                            chapter_name, 
                                            chapter_chunks, 
                                            target_words=100
                                        )
                                        chapter_summaries_for_pdf.append(chapter_summary)
                        except Exception as e:
                            print(f"    Warning: Could not generate summary: {e}")
                            chapter_summary = "[Summary generation failed]"
                    
                    # Fallback to extractive if LLM not available or failed
                    if not chapter_summary and pdf_name in self.vector_stores:
                        try:
                            if hasattr(vector_store.vector_store, '_collection'):
                                collection = vector_store.vector_store._collection
                                all_data = collection.get(include=['metadatas', 'documents'])
                                
                                if all_data and 'metadatas' in all_data and 'documents' in all_data:
                                    texts = []
                                    for metadata, text in zip(all_data['metadatas'], all_data['documents']):
                                        if metadata.get('chapter') == chapter_name:
                                            texts.append(text)
                                    
                                    if texts:
                                        summary_text = " ".join(texts[:5])
                                        words = summary_text.split()
                                        chapter_summary = " ".join(words[:100]) + "..."
                        except Exception:
                            chapter_summary = "[No summary available]"
                    
                    if chapter_summary:
                        f.write(f"   Summary: {chapter_summary}\n")
                    
                    f.write("\n")
                
                # Generate PDF-level summary from chapter summaries
                if chapter_summaries_for_pdf and use_llm:
                    print(f"  Generating PDF-level summary for {pdf_name}...")
                    try:
                        pdf_summary = self._generate_hierarchical_pdf_summary(
                            pdf_name,
                            chapter_summaries_for_pdf,
                            target_words=200
                        )
                        f.write("PDF OVERVIEW:\n")
                        f.write("-" * 80 + "\n")
                        f.write(f"{pdf_summary}\n")
                        f.write("-" * 80 + "\n\n")
                    except Exception as e:
                        print(f"  Warning: Could not generate PDF summary: {e}")
        
        print(f"\n✓ Chapter list report with hierarchical summaries saved to {output_path}")

    def generate_comprehensive_report(self, pdf_name: str, output_file: str = None, min_word_count: int = 1000):
        """
        Generate a comprehensive report for a single PDF using cached summaries.
        
        This method creates a formatted report containing:
        - Book information (title, author, reference if available)
        - Book-level summary with keywords and domain
        - Table of contents with numbered chapters
        - Individual chapter summaries with keywords and domains
        
        The report filters out:
        - Chapters in the exclusion list (Introduction, Contents, References, etc.)
        - Chapters with fewer words than min_word_count
        
        This is the FAST method - uses existing summaries and vector stores.
        For processing from scratch, use process_and_report_single_pdf().
        
        Args:
            pdf_name (str): Name of the PDF (without .pdf extension), e.g., '1.marx', '11.theory'
            output_file (str, optional): Path to save the report. 
                Defaults to ./reports/{pdf_name}_comprehensive_report.txt
            min_word_count (int, optional): Minimum word count for chapters to include. 
                Default is 1000. Use lower values (e.g., 500) for PDFs with shorter chapters.
        
        Returns:
            None. Saves report to output_file.
            
        Raises:
            FileNotFoundError: If vector store for pdf_name doesn't exist
            ValueError: If no summaries found in cache for this PDF
            
        Example:
            >>> pipeline = PDFEmbeddingPipeline(pdf_dir='./sourcepdfs')
            >>> pipeline.generate_comprehensive_report(
            ...     pdf_name='1.marx',
            ...     output_file='./reports/marx_report.txt',
            ...     min_word_count=1000
            ... )
            
        Note:
            - Requires existing vector store in vector_store_dir
            - Requires cached summaries in summaries_cache.json
            - Book summary must be generated first (use generate_book_summary.py)
        """
        from datetime import datetime
        
        if output_file is None:
            output_file = f"./reports/{pdf_name}_comprehensive_report.txt"
        
        print("="*80)
        print(f"GENERATING COMPREHENSIVE REPORT FOR {pdf_name}.pdf")
        print("="*80)
        
        # Load vector store to get chapter information
        print(f"\nLoading vector store for {pdf_name}...")
        vs = VectorStore(pdf_name, persist_directory=f'{self.vector_store_dir}/{pdf_name}')
        
        # Get all chunks and extract chapter information
        results = vs.vector_store._collection.get()
        
        # Organize by chapter
        chapters_data = {}
        for idx, metadata in enumerate(results['metadatas']):
            chapter_name = metadata.get('chapter')
            if not chapter_name:
                continue
                
            if chapter_name not in chapters_data:
                chapters_data[chapter_name] = {
                    'name': chapter_name,
                    'chunks': [],
                    'pages': set(),
                    'metadata': metadata
                }
            
            chapters_data[chapter_name]['chunks'].append(results['documents'][idx])
            if 'page' in metadata:
                chapters_data[chapter_name]['pages'].add(metadata['page'])
        
        print(f"Found {len(chapters_data)} chapters")
        
        # Get chapter summaries from cache
        cached_summaries = self.summary_cache.get_all_for_pdf(pdf_name)
        print(f"Found {len(cached_summaries)} cached summaries")
        print(f"Exclusion list has {len(self.excluded_chapter_titles)} entries")
        
        # Sort chapters (try to maintain order)
        chapter_list = []
        excluded_count = 0
        too_short_count = 0
        
        for chapter_name, data in chapters_data.items():
            # Check if chapter should be excluded
            if self.is_excluded_chapter(chapter_name):
                excluded_count += 1
                print(f"  Excluding: {chapter_name}")
                continue
            
            # Calculate total words
            num_chunks = len(data['chunks'])
            pages = sorted(data['pages']) if data['pages'] else []
            page_range = f"{min(pages)}-{max(pages)}" if pages else "Unknown"
            total_words = sum(len(chunk.split()) for chunk in data['chunks'])
            
            # Skip chapters with less than minimum word count
            if total_words < min_word_count:
                too_short_count += 1
                print(f"  Too short ({total_words} words): {chapter_name}")
                continue
                
            # Get summary from cache
            summary_data = cached_summaries.get(chapter_name, {})
            summary = summary_data.get('summary', '[No summary available]')
            keywords = summary_data.get('keywords', [])
            domain = summary_data.get('domain', '')
            chapter_number = summary_data.get('chapter_number')
            
            chapter_list.append({
                'name': chapter_name,
                'number': chapter_number,
                'summary': summary,
                'keywords': keywords,
                'domain': domain,
                'num_chunks': num_chunks,
                'page_range': page_range,
                'total_words': total_words,
                'first_page': min(pages) if pages else 9999,
                'summary_data': summary_data  # Keep full data for references
            })
        
        # Sort by chapter number if available, otherwise by first page
        chapter_list.sort(key=lambda x: (x['number'] if x['number'] is not None else 9999, x['first_page']))
        
        print(f"\nAfter filtering: {len(chapter_list)} chapters (excluded {excluded_count}, too short {too_short_count})")
        
        # Get book-level summary if available
        book_summary = self.summary_cache.get_pdf_summary(pdf_name)
        book_data = None
        if book_summary:
            for key, value in self.summary_cache.cache.items():
                if '::PDF' in key and value.get('pdf_name') == pdf_name:
                    book_data = value
                    break
        
        # Generate report
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nWriting report to {output_path}...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("="*80 + "\n")
            f.write(f"COMPREHENSIVE REPORT: {pdf_name}.pdf\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}\n\n")
            
            # Book-level information
            f.write("="*80 + "\n")
            f.write("BOOK INFORMATION\n")
            f.write("="*80 + "\n\n")
            
            if book_data:
                book_title = book_data.get('book_title', pdf_name)
                # Clean up title if it's just the PDF name
                if book_title == pdf_name or book_title.endswith('.pdf'):
                    book_title = pdf_name.replace('.pdf', '').title()
                
                f.write(f"Title: {book_title}\n")
                f.write(f"PDF File: {pdf_name}.pdf\n")
                f.write(f"Number of Chapters: {len(chapter_list)}\n")
                
                if book_data.get('author'):
                    f.write(f"Author: {book_data['author']}\n")
                if book_data.get('reference'):
                    f.write(f"Reference: {book_data['reference']}\n")
                
                if book_data.get('keywords'):
                    f.write(f"Keywords: {', '.join(book_data['keywords'])}\n")
                if book_data.get('domain'):
                    f.write(f"Domain: {book_data['domain']}\n")
                
                f.write(f"\nBOOK SUMMARY:\n\n")
                f.write(book_summary + "\n\n")
            else:
                # Generate a cleaner title from filename
                book_title = pdf_name.replace('.pdf', '').replace('.', ' ').replace('_', ' ').title()
                f.write(f"Title: {book_title}\n")
                f.write(f"PDF File: {pdf_name}.pdf\n")
                f.write(f"Number of Chapters: {len(chapter_list)}\n")
                f.write(f"\n[No book-level summary available]\n\n")
            
            # Table of Contents
            f.write("="*80 + "\n")
            f.write("TABLE OF CONTENTS\n")
            f.write("="*80 + "\n\n")
            
            for i, chapter in enumerate(chapter_list, 1):
                f.write(f"{i}. {chapter['name']}\n")
                if chapter['page_range'] != "Unknown":
                    f.write(f"   Pages: {chapter['page_range']}, Words: {chapter['total_words']:,}\n")
                else:
                    f.write(f"   Words: {chapter['total_words']:,}\n")
            
            f.write("\n")
            f.write("="*80 + "\n")
            f.write("CHAPTER SUMMARIES\n")
            f.write("="*80 + "\n\n")
            
            # Process each chapter
            for i, chapter in enumerate(chapter_list, 1):
                print(f"  Processing chapter {i}/{len(chapter_list)}: {chapter['name'][:50]}...")
                
                # Chapter header
                f.write("-"*80 + "\n")
                f.write(f"CHAPTER {i}: {chapter['name']}\n")
                f.write("-"*80 + "\n\n")
                
                f.write(f"Book: {book_title if 'book_title' in locals() else pdf_name}\n")
                f.write(f"Chapter: {chapter['name']}\n")
                
                # Show reference/page information
                if chapter['page_range'] != "Unknown":
                    f.write(f"Pages: {chapter['page_range']}\n")
                
                f.write(f"Total Words: {chapter['total_words']:,}\n")
                
                if chapter['keywords']:
                    f.write(f"Keywords: {', '.join(chapter['keywords'])}\n")
                if chapter['domain']:
                    f.write(f"Domain: {chapter['domain']}\n")
                
                # Get reference data from cache if available
                if chapter['summary_data'].get('author'):
                    f.write(f"Author: {chapter['summary_data']['author']}\n")
                if chapter['summary_data'].get('reference'):
                    f.write(f"Reference: {chapter['summary_data']['reference']}\n")
                
                # Summary
                f.write(f"\nSUMMARY:\n\n")
                f.write(chapter['summary'])
                f.write("\n\n")
            
            # Footer
            f.write("="*80 + "\n")
            f.write(f"END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"\n✓ Report generated: {output_path}")
        print(f"  Total chapters: {len(chapter_list)}")
        print(f"  File size: {output_path.stat().st_size:,} bytes")

    def process_and_report_single_pdf(
        self, 
        pdf_name: str, 
        output_file: str = None, 
        min_word_count: int = 1000,
        generate_book_summary: bool = True
    ):
        """
        Process a single PDF from scratch and generate a comprehensive report.
        
        This is the COMPLETE PIPELINE that:
        1. Parses the PDF with multi-threshold heading detection
        2. Chunks text with semantic 6-level hierarchical merging
        3. Generates embeddings and stores in ChromaDB vector database
        4. Generates chapter summaries using Ollama LLM (hierarchical summarization)
        5. Synthesizes book-level summary from chapter summaries
        6. Creates comprehensive formatted report
        
        This is the SLOW method (5-10 minutes) - processes everything from scratch.
        For fast report generation with existing data, use generate_comprehensive_report().
        
        Args:
            pdf_name (str): Name of the PDF (without .pdf extension), e.g., '11.theory', '7.philosophy'
            output_file (str, optional): Path to save the report. 
                Defaults to ./reports/{pdf_name}_test_report.txt
            min_word_count (int, optional): Minimum word count for chapters to include. 
                Default is 1000. Use lower values (e.g., 500) for PDFs with shorter chapters.
            generate_book_summary (bool, optional): Whether to generate book-level summary. 
                Default is True. Set to False to skip book summary synthesis.
        
        Returns:
            None. Saves report to output_file.
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist in pdf_dir
            RuntimeError: If Ollama is not running or model not available
            
        Example:
            >>> pipeline = PDFEmbeddingPipeline(
            ...     pdf_dir='/Users/name/AI/sourcepdfs',
            ...     vector_store_dir='./vector_stores'
            ... )
            >>> pipeline.process_and_report_single_pdf(
            ...     pdf_name='11.theory',
            ...     output_file='./reports/11theory_report.txt',
            ...     min_word_count=500,
            ...     generate_book_summary=True
            ... )
            
        Processing Steps:
            1. PDF Parsing: Extracts text with multi-threshold heading detection
            2. Chunking: Semantic chunking + 6-level hierarchical merging (similarity > 0.75)
            3. Embedding: Generates sentence transformer embeddings
            4. Storage: Stores in ChromaDB with metadata (chapter, section, page, etc.)
            5. Summarization: Hierarchical LLM summarization (5-6 merged chunks → summary)
            6. Book Summary: Synthesizes from all chapter summaries
            7. Report: Formatted output with TOC, keywords, domains
            
        Performance:
            - Typical runtime: 5-10 minutes for a full book
            - Memory usage: ~1-2GB
            - Ollama speed: ~2-3 seconds per chapter summary
            
        Note:
            - Requires Ollama running locally (http://localhost:11434/v1)
            - Requires gemma3:4b model: `ollama pull gemma3:4b`
            - Creates new vector store in vector_store_dir
            - Saves summaries to summaries_cache.json
        """
        if output_file is None:
            output_file = f"./reports/{pdf_name}_report.txt"
        
        print("="*80)
        print(f"FULL PIPELINE: PROCESS AND REPORT FOR {pdf_name}.pdf")
        print("="*80)
        
        pdf_path = self.pdf_dir / f"{pdf_name}.pdf"
        if not pdf_path.exists():
            print(f"❌ Error: PDF '{pdf_name}.pdf' not found in {self.pdf_dir}")
            return
        
        # STEP 1: Process PDF (chunking, embedding, vector store)
        print("\n--- STEP 1: Processing PDF ---")
        try:
            self.process_single_pdf(pdf_path)
        except Exception as e:
            print(f"❌ Error processing PDF: {e}")
            return
        
        # STEP 2: Generate chapter summaries
        print("\n--- STEP 2: Generating Chapter Summaries ---")
        
        vector_store = self.vector_stores.get(pdf_name)
        if not vector_store:
            print(f"❌ Error: Vector store not found for {pdf_name}")
            return
        
        # Get all chapters
        collection = vector_store.vector_store._collection
        all_data = collection.get(include=['metadatas'])
        
        if not all_data or 'metadatas' not in all_data:
            print(f"❌ No data found for {pdf_name}")
            return
        
        # Get unique chapters
        chapters = set()
        for metadata in all_data['metadatas']:
            chapter = metadata.get('chapter')
            if chapter and not self.is_excluded_chapter(chapter):
                chapters.add(chapter)
        
        chapters = sorted(chapters)
        print(f"Found {len(chapters)} chapters to summarize")
        
        # Generate summary for each chapter
        for i, chapter_name in enumerate(chapters, 1):
            print(f"  [{i}/{len(chapters)}] {chapter_name[:70]}...")
            
            # Check cache first
            if self.summary_cache.get(chapter_name, pdf_name):
                print(f"    ↻ Using cached summary")
                continue
            
            try:
                # Generate hierarchical summary
                result = self.summarize_chapter_hierarchical(
                    pdf_name=pdf_name,
                    chapter_name=chapter_name,
                    max_levels=5
                )
                
                if result and 'final_summary' in result:
                    summary = result['final_summary']
                    word_count = len(summary.split())
                    
                    # Save to cache
                    self.summary_cache.set(
                        pdf_name=pdf_name,
                        chapter_name=chapter_name,
                        summary=summary
                    )
                    
                    print(f"    ✓ Generated summary ({word_count} words)")
                else:
                    print(f"    ⚠ Failed to generate summary")
            except Exception as e:
                print(f"    ❌ Error: {e}")
        
        # STEP 3: Generate book-level summary (if requested)
        if generate_book_summary:
            print("\n--- STEP 3: Generating Book Summary ---")
            
            from generate_book_summary import generate_book_summary
            try:
                generate_book_summary(pdf_name)
            except Exception as e:
                print(f"⚠ Could not generate book summary: {e}")
                print("  Continuing with chapter summaries only...")
        
        # STEP 4: Generate comprehensive report
        print("\n--- STEP 4: Generating Report ---")
        self.generate_comprehensive_report(
            pdf_name=pdf_name,
            output_file=output_file,
            min_word_count=min_word_count
        )
        
        print(f"\n{'='*80}")
        print(f"✓ COMPLETE: {pdf_name}.pdf")
        print(f"{'='*80}")

    def generate_all_reports(
        self, 
        reports_dir: str = "./reports",
        min_word_count: int = 1000,
        reprocess: bool = False
    ):
        """
        Generate comprehensive reports for all PDFs.
        
        For each PDF, either:
        - If reprocess=True: Process from scratch (chunk, embed, summarize, report)
        - If reprocess=False: Use existing summaries to generate report
        
        Args:
            reports_dir: Directory to save reports (default: ./reports)
            min_word_count: Minimum word count for chapters (default: 1000)
            reprocess: If True, reprocess PDFs from scratch; if False, use existing summaries
        """
        from datetime import datetime
        
        print("="*80)
        print(f"GENERATE REPORTS FOR ALL PDFs")
        print(f"Mode: {'REPROCESS FROM SCRATCH' if reprocess else 'USE EXISTING SUMMARIES'}")
        print("="*80)
        
        # Get all PDFs
        pdf_files = sorted(self.pdf_dir.glob("*.pdf"))
        print(f"\nFound {len(pdf_files)} PDFs")
        
        reports_path = Path(reports_dir)
        reports_path.mkdir(parents=True, exist_ok=True)
        
        start_time = datetime.now()
        success_count = 0
        error_count = 0
        
        for i, pdf_path in enumerate(pdf_files, 1):
            pdf_name = pdf_path.stem
            output_file = reports_path / f"{pdf_name}_report.txt"
            
            print(f"\n{'='*80}")
            print(f"[{i}/{len(pdf_files)}] {pdf_name}.pdf")
            print(f"{'='*80}")
            
            try:
                if reprocess:
                    # Full pipeline: process from scratch
                    self.process_and_report_single_pdf(
                        pdf_name=pdf_name,
                        output_file=str(output_file),
                        min_word_count=min_word_count
                    )
                else:
                    # Quick report from existing summaries
                    self.generate_comprehensive_report(
                        pdf_name=pdf_name,
                        output_file=str(output_file),
                        min_word_count=min_word_count
                    )
                
                success_count += 1
                print(f"✓ Report saved: {output_file}")
                
            except Exception as e:
                error_count += 1
                print(f"❌ Error generating report for {pdf_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n{'='*80}")
        print(f"SUMMARY")
        print(f"{'='*80}")
        print(f"Total PDFs: {len(pdf_files)}")
        print(f"Successful: {success_count}")
        print(f"Errors: {error_count}")
        print(f"Duration: {duration/60:.1f} minutes")
        print(f"Reports saved to: {reports_path}")
        print(f"{'='*80}")

    
    def save_cluster_report(self, cluster_result: Dict[str, Any], similarities: Dict, output_file: str = "./cluster_report.txt"):
        """Save clustering analysis report to a text file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("CHAPTER CLUSTERING ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Summary statistics
            f.write(f"Total Chapters Analyzed: {len(self.chapter_embeddings)}\n")
            f.write(f"Total PDFs: {len(self.vector_stores)}\n")
            f.write(f"Pairwise Similarities Computed: {len(similarities)}\n\n")
            
            # Clustering method and quality
            f.write("="*80 + "\n")
            f.write("CLUSTERING RESULTS\n")
            f.write("="*80 + "\n\n")
            f.write(f"Method: {cluster_result.get('method', 'N/A').upper()}\n")
            f.write(f"Number of Clusters: {cluster_result.get('n_clusters', 0)}\n")
            f.write(f"Silhouette Score: {cluster_result.get('silhouette_score', 0):.3f}")
            f.write(" (higher is better, range: -1 to 1)\n\n")
            
            # Top similarities
            f.write("="*80 + "\n")
            f.write("TOP 20 MOST SIMILAR CHAPTER PAIRS\n")
            f.write("="*80 + "\n\n")
            sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            for i, ((ch1, ch2), sim) in enumerate(sorted_sims[:20], 1):
                pdf1, name1 = ch1
                pdf2, name2 = ch2
                f.write(f"{i}. Similarity: {sim:.3f}\n")
                f.write(f"   [{pdf1}] {name1}\n")
                f.write(f"   [{pdf2}] {name2}\n\n")
            
            # Cluster assignments
            f.write("="*80 + "\n")
            f.write("CLUSTER ASSIGNMENTS\n")
            f.write("="*80 + "\n\n")
            
            clusters = cluster_result.get('clusters', {})
            for cluster_id in sorted(clusters.keys()):
                if cluster_id == -1:
                    f.write(f"Cluster {cluster_id} (NOISE - {len(clusters[cluster_id])} chapters):\n")
                else:
                    f.write(f"Cluster {cluster_id} ({len(clusters[cluster_id])} chapters):\n")
                f.write("-" * 80 + "\n")
                
                for pdf_name, chapter_name in clusters[cluster_id]:
                    metadata = self.chapter_metadata.get((pdf_name, chapter_name), {})
                    page = metadata.get('start_page', 'N/A')
                    chunks = metadata.get('num_chunks', 'N/A')
                    f.write(f"  • [{pdf_name}] {chapter_name}\n")
                    f.write(f"    Page: {page}, Chunks: {chunks}\n")
                f.write("\n")
            
            # Chapter embeddings summary
            f.write("="*80 + "\n")
            f.write("CHAPTER DETAILS\n")
            f.write("="*80 + "\n\n")
            
            # Group by PDF
            pdf_chapters = {}
            for (pdf_name, chapter_name) in self.chapter_embeddings.keys():
                if pdf_name not in pdf_chapters:
                    pdf_chapters[pdf_name] = []
                pdf_chapters[pdf_name].append(chapter_name)
            
            for pdf_name in sorted(pdf_chapters.keys()):
                f.write(f"\n{pdf_name}.pdf ({len(pdf_chapters[pdf_name])} chapters):\n")
                f.write("-" * 80 + "\n")
                for chapter in pdf_chapters[pdf_name]:
                    metadata = self.chapter_metadata.get((pdf_name, chapter), {})
                    f.write(f"  • {chapter} (Page {metadata.get('start_page', 'N/A')})\n")
        
        print(f"\n✓ Cluster report saved to {output_path}")
        return output_path


if __name__ == "__main__":
    # Configuration
    PDF_DIR = "/Users/GStahl2/AI/sourcepdfs"
    VECTOR_STORE_DIR = "./vector_stores"
    EMBEDDER_TYPE = "sentence-transformers"
    
    print("Initializing pipeline...")
    # Initialize pipeline
    pipeline = PDFEmbeddingPipeline(
        pdf_dir=PDF_DIR,
        vector_store_dir=VECTOR_STORE_DIR,
        embedder_type=EMBEDDER_TYPE,
        chunk_size=512,
        chunk_overlap=50
    )
    
    print("Pipeline initialized. Starting to process PDFs...")
    # Process all PDFs (each will be queried individually)
    pipeline.process_pdfs()
    
    # Skip querying and summarization for large batch processing
    print("\nSkipping individual queries and summaries for batch processing...")
    
    # CHAPTER CLUSTERING ANALYSIS
    print("\n" + "="*60)
    print("CHAPTER CLUSTERING ANALYSIS")
    print("="*60)
    
    # Create chapter-level embeddings
    pipeline.create_chapter_embeddings()
    
    # Compute similarities between chapters
    similarities = pipeline.compute_chapter_similarities()
    
    # Cluster chapters using different methods
    print("\n--- K-Means Clustering ---")
    kmeans_result = pipeline.cluster_chapters(n_clusters=8, method='kmeans')
    
    print("\n--- Hierarchical Clustering ---")
    hierarchical_result = pipeline.cluster_chapters(n_clusters=8, method='hierarchical')
    
    # Save chapter embeddings for future use
    pipeline.save_chapter_embeddings("./chapter_embeddings.npz")
    
    # Save cluster reports to files
    print("\n--- Saving Cluster Reports ---")
    pipeline.save_cluster_report(kmeans_result, similarities, "./reports/kmeans_cluster_report.txt")
    pipeline.save_cluster_report(hierarchical_result, similarities, "./reports/hierarchical_cluster_report.txt")
    pipeline.save_chapters_by_pdf_report("./reports/chapters_by_pdf.txt")
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
