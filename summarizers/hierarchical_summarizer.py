"""
Hierarchical Summarizer
Generates summaries by progressively summarizing larger text segments.
"""

from typing import List, Dict, Any
from openai import OpenAI
import os


class HierarchicalSummarizer:
    """
    Generate hierarchical summaries of documents.
    
    This is one of the THREE KEY FUNCTIONS you wanted help with.
    
    Strategy:
    1. Start with small chunks (paragraphs or sections)
    2. Summarize each chunk
    3. Group summaries and summarize them again
    4. Repeat until you have a single document summary
    """
    
    def __init__(
        self,
        model: str = "gpt-4",
        max_chunk_tokens: int = 2000,
        summary_ratio: float = 0.3
    ):
        self.model = model
        self.max_chunk_tokens = max_chunk_tokens
        self.summary_ratio = summary_ratio
        
        # Initialize OpenAI client (v1.0+ API) - optional
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key else None
        
    def summarize(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate hierarchical summary of document chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'text' and 'metadata'
            
        Returns:
            Dictionary containing:
                - final_summary: Top-level summary
                - section_summaries: Summaries by section
                - chunk_summaries: Individual chunk summaries
        """
        # Level 1: Summarize individual chunks
        chunk_summaries = self._summarize_chunks(chunks)
        
        # Level 2: Group by section and summarize
        section_summaries = self._summarize_by_section(chunks, chunk_summaries)
        
        # Level 3: Create final document summary
        final_summary = self._create_final_summary(section_summaries)
        
        return {
            'final_summary': final_summary,
            'section_summaries': section_summaries,
            'chunk_summaries': chunk_summaries
        }
    
    def _summarize_chunks(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Summarize individual chunks.
        
        TODO: Implement the actual summarization logic here.
        """
        summaries = []
        
        for chunk in chunks:
            summary = self._summarize_text(
                chunk['text'],
                context=f"Chapter: {chunk['metadata'].get('chapter', 'N/A')}, "
                        f"Section: {chunk['metadata'].get('section', 'N/A')}"
            )
            summaries.append(summary)
        
        return summaries
    
    def _summarize_by_section(
        self, 
        chunks: List[Dict[str, Any]], 
        chunk_summaries: List[str]
    ) -> Dict[str, str]:
        """
        Group chunk summaries by section and create section-level summaries.
        
        TODO: Implement section grouping and summarization.
        """
        # Group chunks by section
        sections = {}
        for chunk, summary in zip(chunks, chunk_summaries):
            section = chunk['metadata'].get('section', 'Unknown Section')
            if section not in sections:
                sections[section] = []
            sections[section].append(summary)
        
        # Summarize each section
        section_summaries = {}
        for section, summaries in sections.items():
            combined_text = "\n\n".join(summaries)
            section_summary = self._summarize_text(
                combined_text,
                context=f"Section: {section}",
                instruction="Summarize the key points from this section"
            )
            section_summaries[section] = section_summary
        
        return section_summaries
    
    def _create_final_summary(self, section_summaries: Dict[str, str]) -> str:
        """
        Create final document-level summary from section summaries.
        
        TODO: Implement final summarization logic.
        """
        combined_sections = "\n\n".join([
            f"**{section}**\n{summary}" 
            for section, summary in section_summaries.items()
        ])
        
        final_summary = self._summarize_text(
            combined_sections,
            context="Full Document",
            instruction="Create a comprehensive summary of the entire document, "
                       "highlighting the main themes and key takeaways"
        )
        
        return final_summary
    
    def _summarize_text(
        self, 
        text: str, 
        context: str = "", 
        instruction: str = "Summarize the following text concisely"
    ) -> str:
        """
        Call LLM to summarize a piece of text using OpenAI API v1.0+.
        """
        if not self.client:
            return f"[Summarization disabled - no OpenAI API key]"
        
        prompt = f"{instruction}\n\nContext: {context}\n\nText:\n{text}"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates concise, accurate summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return f"[Summary placeholder for: {context}]"
