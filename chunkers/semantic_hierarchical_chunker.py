"""
Semantic Hierarchical Chunker - uses vector similarity to create hierarchical chunks.

This module implements a bottom-up approach to chunking:
1. Start with base-level chunks (512 chars)
2. Compute embeddings for each chunk
3. Calculate cosine similarity between consecutive chunks
4. Merge most similar adjacent chunks
5. Repeat until reaching chapter-level granularity
6. Generate summaries at each hierarchy level
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


class SemanticHierarchicalChunk:
    """Represents a chunk at any level of the hierarchy."""
    
    def __init__(
        self,
        text: str,
        embedding: np.ndarray,
        level: int,
        metadata: Dict[str, Any],
        children: Optional[List['SemanticHierarchicalChunk']] = None,
        summary: Optional[str] = None
    ):
        """
        Initialize a hierarchical chunk.
        
        Args:
            text: The text content
            embedding: Vector embedding
            level: Hierarchy level (0=base chunks, 1=first merge, etc.)
            metadata: Associated metadata (page, chapter, etc.)
            children: Child chunks that were merged to create this
            summary: LLM-generated summary (optional)
        """
        self.text = text
        self.embedding = embedding
        self.level = level
        self.metadata = metadata
        self.children = children or []
        self.summary = summary
        
    def __repr__(self):
        return f"Chunk(level={self.level}, text_len={len(self.text)}, children={len(self.children)})"


class SemanticHierarchicalChunker:
    """Creates hierarchical chunks using vector similarity."""
    
    def __init__(
        self,
        embedder,
        similarity_threshold: float = 0.70,
        min_chunks_per_level: int = 25,
        max_text_length: int = 3000
    ):
        """
        Initialize the semantic hierarchical chunker.
        
        Args:
            embedder: Embedder instance for generating vectors
            similarity_threshold: Minimum similarity to merge chunks (0-1)
            min_chunks_per_level: Stop merging when this many chunks remain
            max_text_length: Maximum characters when merging chunks
        """
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.min_chunks_per_level = min_chunks_per_level
        self.max_text_length = max_text_length
        
    def create_base_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[np.ndarray]
    ) -> List[SemanticHierarchicalChunk]:
        """
        Convert initial chunks to SemanticHierarchicalChunk objects.
        
        Args:
            chunks: Base-level chunks from semantic_chunker
            embeddings: Pre-computed embeddings
            
        Returns:
            List of level-0 hierarchical chunks
        """
        hierarchical_chunks = []
        
        for chunk, embedding in zip(chunks, embeddings):
            metadata = {
                'page': chunk.get('page', 0),
                'chapter': chunk.get('chapter', 'Unknown'),
                'source': chunk.get('source', ''),
                'chunk_index': chunk.get('chunk_index', 0)
            }
            
            hchunk = SemanticHierarchicalChunk(
                text=chunk['text'],
                embedding=embedding,
                level=0,
                metadata=metadata
            )
            hierarchical_chunks.append(hchunk)
            
        logger.info(f"Created {len(hierarchical_chunks)} base-level chunks")
        return hierarchical_chunks
    
    def compute_similarity_matrix(
        self,
        chunks: List[SemanticHierarchicalChunk]
    ) -> np.ndarray:
        """
        Compute pairwise cosine similarity between consecutive chunks.
        
        Args:
            chunks: List of hierarchical chunks
            
        Returns:
            Array of similarity scores between adjacent chunks
        """
        if len(chunks) <= 1:
            return np.array([])
            
        similarities = []
        for i in range(len(chunks) - 1):
            # Compute cosine similarity between consecutive chunks
            sim = cosine_similarity(
                chunks[i].embedding.reshape(1, -1),
                chunks[i + 1].embedding.reshape(1, -1)
            )[0][0]
            similarities.append(sim)
            
        return np.array(similarities)
    
    def merge_chunks(
        self,
        chunk1: SemanticHierarchicalChunk,
        chunk2: SemanticHierarchicalChunk
    ) -> SemanticHierarchicalChunk:
        """
        Merge two adjacent chunks into a higher-level chunk.
        
        Args:
            chunk1: First chunk
            chunk2: Second chunk
            
        Returns:
            New merged chunk at level+1
        """
        # Combine texts
        merged_text = f"{chunk1.text}\n\n{chunk2.text}"
        
        # Generate new embedding for merged text
        if hasattr(self.embedder, 'embed_texts'):
            merged_embedding = self.embedder.embed_texts([merged_text])[0]
        elif hasattr(self.embedder, 'encode'):
            merged_embedding = self.embedder.encode([merged_text])[0]
        else:
            raise ValueError("Embedder must have either embed_texts() or encode() method")
        
        # Merge metadata (prefer first chunk's chapter/page)
        merged_metadata = {
            'page': chunk1.metadata.get('page', 0),
            'chapter': chunk1.metadata.get('chapter', 'Unknown'),
            'source': chunk1.metadata.get('source', ''),
            'page_range': f"{chunk1.metadata.get('page', 0)}-{chunk2.metadata.get('page', 0)}"
        }
        
        # Create new chunk
        merged_chunk = SemanticHierarchicalChunk(
            text=merged_text,
            embedding=merged_embedding,
            level=max(chunk1.level, chunk2.level) + 1,
            metadata=merged_metadata,
            children=[chunk1, chunk2]
        )
        
        return merged_chunk
    
    def merge_most_similar(
        self,
        chunks: List[SemanticHierarchicalChunk],
        threshold: Optional[float] = None
    ) -> List[SemanticHierarchicalChunk]:
        """
        Perform one iteration of merging consecutive chunks above similarity threshold.
        
        Merges consecutive RUNS of chunks where ALL adjacent pairs have 
        similarity >= similarity_threshold. For example, if A→B→C→D all have
        similarity ≥ threshold, merges A+B+C+D into one chunk, not just pairs.
        
        Args:
            chunks: Current list of chunks
            threshold: Similarity threshold for this level (overrides default)
            
        Returns:
            New list with all qualifying runs merged
        """
        if len(chunks) <= 1:
            return chunks
        
        # Use provided threshold or fall back to instance default
        current_threshold = threshold if threshold is not None else self.similarity_threshold
            
        # Compute similarities
        similarities = self.compute_similarity_matrix(chunks)
        
        # Debug: show similarity distribution
        if len(similarities) > 0:
            logger.info(
                f"Similarity stats: min={np.min(similarities):.3f}, "
                f"max={np.max(similarities):.3f}, "
                f"mean={np.mean(similarities):.3f}, "
                f"threshold={current_threshold}"
            )
        
        if len(similarities) == 0:
            return chunks
        
        # Find consecutive runs of chunks to merge
        # A run is a sequence where ALL consecutive pairs have similarity >= threshold
        runs = []  # List of (start_idx, end_idx) for each run to merge
        i = 0
        
        while i < len(chunks):
            # Check if this chunk starts a run
            if i < len(similarities) and similarities[i] >= current_threshold:
                # Start a new run
                run_start = i
                run_end = i + 1  # Include the next chunk
                
                # Extend the run as far as possible
                while run_end < len(chunks):
                    # Check if we can extend further
                    if run_end < len(similarities) and similarities[run_end] >= current_threshold:
                        # Check total length wouldn't exceed max
                        total_length = sum(len(chunks[j].text) for j in range(run_start, run_end + 2))
                        if total_length <= self.max_text_length:
                            run_end += 1
                        else:
                            # Can't extend further due to length constraint
                            break
                    else:
                        # Next pair not similar enough
                        break
                
                # Record this run (inclusive indices)
                runs.append((run_start, run_end))
                logger.debug(
                    f"Found run: chunks {run_start}-{run_end} "
                    f"({run_end - run_start + 1} chunks)"
                )
                
                # Skip past this run
                i = run_end + 1
            else:
                # No run starting here, move to next
                i += 1
        
        # If no runs found, return original chunks
        if not runs:
            logger.debug("No runs above similarity threshold; stopping merging")
            return chunks
        
        # Build new chunk list with all runs merged
        new_chunks = []
        i = 0
        run_idx = 0
        
        while i < len(chunks):
            # Check if we're at the start of a run
            if run_idx < len(runs) and i == runs[run_idx][0]:
                run_start, run_end = runs[run_idx]
                
                # Merge all chunks in this run
                merged_chunk = chunks[run_start]
                for j in range(run_start + 1, run_end + 1):
                    merged_chunk = self.merge_chunks(merged_chunk, chunks[j])
                
                new_chunks.append(merged_chunk)
                
                # Skip to end of run
                i = run_end + 1
                run_idx += 1
            else:
                # Keep unchanged
                new_chunks.append(chunks[i])
                i += 1
        
        total_merged = sum(end - start + 1 for start, end in runs)
        logger.info(
            f"Merged {len(runs)} runs ({total_merged} chunks): "
            f"{len(chunks)} → {len(new_chunks)} chunks"
        )
        
        return new_chunks
    
    def build_hierarchy(
        self,
        base_chunks: List[SemanticHierarchicalChunk],
        max_levels: int = 6
    ) -> List[List[SemanticHierarchicalChunk]]:
        """
        Build complete hierarchical structure by iteratively merging.
        
        Uses decreasing similarity thresholds at each level:
        - Level 1: similarity_threshold (0.70)
        - Level 2: similarity_threshold - 0.10 (0.60)
        - Level 3: similarity_threshold - 0.20 (0.50)
        - Level 4: similarity_threshold - 0.30 (0.40)
        - Level 5: similarity_threshold - 0.40 (0.30)
        - Level 6: similarity_threshold - 0.50 (0.20)
        - etc.
        
        Continues iterating even if no merges occur at some levels,
        until chunk count drops below min_chunks_per_level or max_levels reached.
        
        Args:
            base_chunks: Starting chunks (level 0)
            max_levels: Maximum hierarchy depth (default: 6, stops at threshold 0.20)
            
        Returns:
            List of chunk lists, one per hierarchy level
        """
        hierarchy = [base_chunks]
        current_chunks = base_chunks
        
        for level in range(1, max_levels + 1):
            if len(current_chunks) <= self.min_chunks_per_level:
                logger.info(
                    f"Stopping at level {level - 1}: "
                    f"only {len(current_chunks)} chunks remain"
                )
                break
            
            # Calculate threshold for this level (decrease by 0.10 per level)
            level_threshold = self.similarity_threshold - (0.10 * (level - 1))
            level_threshold = max(0.20, level_threshold)  # Don't go below 0.20
            
            logger.info(
                f"Level {level}: Using threshold {level_threshold:.2f}"
            )
            
            # Merge using level-specific threshold
            new_chunks = self.merge_most_similar(current_chunks, threshold=level_threshold)
            
            # Note: We continue even if no merges occur this iteration
            # The decreasing threshold may enable merges in future iterations
            
            logger.info(
                f"Level {level}: {len(current_chunks)} → {len(new_chunks)} chunks"
            )
            
            hierarchy.append(new_chunks)
            current_chunks = new_chunks
        
        return hierarchy
    
    def add_summaries(
        self,
        hierarchy: List[List[SemanticHierarchicalChunk]],
        summarizer_fn
    ) -> List[List[SemanticHierarchicalChunk]]:
        """
        Generate LLM summaries for chunks at each level using true hierarchical approach.
        
        - Level 0 (base): Summarize raw text
        - Level 1+: Combine summaries from child chunks
        
        This creates a true hierarchy where higher-level summaries are built
        from lower-level summaries, not from raw text.
        
        Args:
            hierarchy: The hierarchical structure
            summarizer_fn: Function that takes (text, level, target_words, is_base_level) and returns summary
            
        Returns:
            Hierarchy with summaries added
        """
        for level_idx, level_chunks in enumerate(hierarchy):
            logger.info(f"Generating summaries for level {level_idx}")
            
            for chunk in level_chunks:
                if chunk.summary is None:
                    target_words = self._get_target_words(level_idx, len(hierarchy))
                    
                    # True hierarchical summarization: use child summaries if available
                    if chunk.children and len(chunk.children) > 0:
                        # Higher-level chunk: combine child summaries
                        child_summaries = [child.summary for child in chunk.children if child.summary]
                        
                        if child_summaries:
                            # Create combined text from child summaries
                            combined_summaries = "\n\n".join(child_summaries)
                            chunk.summary = summarizer_fn(
                                combined_summaries, 
                                level_idx, 
                                target_words,
                                is_base_level=False  # Combining summaries, not raw text
                            )
                            logger.debug(
                                f"Level {level_idx}: Generated summary from {len(child_summaries)} child summaries"
                            )
                        else:
                            # Fallback: no child summaries available, use raw text
                            chunk.summary = summarizer_fn(chunk.text, level_idx, target_words, is_base_level=True)
                            logger.warning(
                                f"Level {level_idx}: Children exist but no summaries, using raw text"
                            )
                    else:
                        # Base-level chunk: summarize raw text
                        chunk.summary = summarizer_fn(chunk.text, level_idx, target_words, is_base_level=True)
                        logger.debug(f"Level {level_idx}: Generated summary from raw text")
        
        return hierarchy
    
    def _get_target_words(self, level: int, total_levels: int) -> int:
        """
        Determine target summary length based on hierarchy level.
        
        Lower levels: shorter summaries (50-100 words)
        Higher levels: longer summaries (150-200 words)
        """
        if total_levels <= 2:
            # Simple case: base + top
            return 100 if level == 0 else 200
        
        # Scale from 50 words (bottom) to 200 words (top)
        min_words = 50
        max_words = 200
        ratio = level / (total_levels - 1)
        
        return int(min_words + (max_words - min_words) * ratio)
    
    def get_final_summary(
        self,
        hierarchy: List[List[SemanticHierarchicalChunk]]
    ) -> str:
        """
        Get the top-level summary (entire chapter).
        
        Args:
            hierarchy: Complete hierarchy
            
        Returns:
            Summary from the top-level chunk
        """
        if not hierarchy or not hierarchy[-1]:
            return ""
        
        top_chunks = hierarchy[-1]
        
        if len(top_chunks) == 1:
            return top_chunks[0].summary or ""
        
        # If multiple top-level chunks remain, combine their summaries
        summaries = [chunk.summary for chunk in top_chunks if chunk.summary]
        return "\n\n".join(summaries)
    
    def export_hierarchy_info(
        self,
        hierarchy: List[List[SemanticHierarchicalChunk]]
    ) -> Dict[str, Any]:
        """
        Export hierarchy structure for analysis/debugging.
        
        Returns:
            Dictionary with hierarchy statistics
        """
        info = {
            'total_levels': len(hierarchy),
            'levels': []
        }
        
        for level_idx, level_chunks in enumerate(hierarchy):
            level_info = {
                'level': level_idx,
                'num_chunks': len(level_chunks),
                'avg_text_length': np.mean([len(c.text) for c in level_chunks]),
                'avg_summary_length': np.mean([
                    len(c.summary) if c.summary else 0 
                    for c in level_chunks
                ])
            }
            info['levels'].append(level_info)
        
        return info
