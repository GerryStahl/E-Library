"""
PDF Parser with Heading Structure Extraction
Extracts text and heading hierarchy (chapters, sections) from PDFs for metadata enrichment.
"""

from typing import Dict, List, Any, Tuple
from pathlib import Path
import fitz  # PyMuPDF
import re


class PDFParser:
    """Parse PDFs and extract heading structure for metadata."""
    
    def __init__(self, h1_min_size: float = 20.0, h2_min_size: float = 14.0):
        """
        Initialize parser with heading detection thresholds.
        
        Args:
            h1_min_size: Minimum font size for H1 headings (chapters)
            h2_min_size: Minimum font size for H2 headings (sections)
        """
        self.h1_min_size = h1_min_size
        self.h2_min_size = h2_min_size
        self.h2_only_pdfs = []  # List of PDF filenames that should only use H2 as chapters (18pt+)
        self.h2_moderate_pdfs = []  # List of PDF filenames that should use moderate H2s (16pt+)
        self.h2_small_pdfs = []  # List of PDF filenames that should use smaller H2s (14pt+)
        
    def parse_with_headings(self, pdf_path: Path, skip_pages: int = 4) -> Dict[str, Any]:
        """
        Parse PDF and extract text with heading structure and page numbers.
        
        Args:
            pdf_path: Path to PDF file
            skip_pages: Number of initial pages to skip (default: 4 for front matter)
            
        Returns:
            Dictionary containing:
                - text: Full text content
                - metadata: Document-level metadata
                - page_chunks: List of (text, page_number, char_start, char_end)
                - heading_structure: List of detected headings with positions
        """
        doc = fitz.open(str(pdf_path))
        pdf_filename = pdf_path.name
        
        full_text = ""
        page_chunks = []  # Track text by page
        heading_structure = []
        current_position = 0
        
        for page_num in range(len(doc)):
            # Skip front matter pages
            if page_num < skip_pages:
                continue
                
            page = doc[page_num]
            page_text = page.get_text("text")
            
            # Track this page's text position
            start_pos = current_position
            full_text += page_text
            end_pos = current_position + len(page_text)
            
            page_chunks.append({
                'page_number': page_num + 1,
                'start_pos': start_pos,
                'end_pos': end_pos,
                'text': page_text
            })
            
            # Extract headings from this page using font size
            page_headings = self._extract_headings_from_page(page, page_num + 1, start_pos, pdf_filename)
            heading_structure.extend(page_headings)
            
            current_position = end_pos
        
        doc.close()
        
        # Combine consecutive headings with the same font size on the same page
        heading_structure = self._combine_split_headings(heading_structure)
        
        return {
            'text': full_text,
            'metadata': {
                'filename': pdf_path.name,
                'path': str(pdf_path),
                'num_pages': len(page_chunks),
                'skipped_pages': skip_pages
            },
            'page_chunks': page_chunks,
            'heading_structure': heading_structure
        }
    
    def _extract_headings_from_page(
        self, 
        page, 
        page_num: int, 
        text_offset: int,
        pdf_filename: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Extract headings from a page based on font size.
        
        Args:
            page: PyMuPDF page object
            page_num: Page number (1-indexed)
            text_offset: Character offset in full document text
            pdf_filename: Name of the PDF file being processed
            
        Returns:
            List of heading dictionaries
        """
        headings = []
        blocks = page.get_text("dict")["blocks"]
        
        current_offset = text_offset
        
        for block in blocks:
            if "lines" not in block:
                continue
                
            for line in block["lines"]:
                line_text = ""
                max_font_size = 0
                
                # Get text and max font size for this line
                for span in line["spans"]:
                    line_text += span["text"]
                    max_font_size = max(max_font_size, span["size"])
                
                line_text = line_text.strip()
                
                # Detect heading level based on font size
                if line_text and len(line_text) > 0:
                    heading_level = self._detect_heading_level(max_font_size, line_text, pdf_filename)
                    
                    if heading_level:
                        # Normalize heading text: strip leading numbers like "10. " or "1. "
                        normalized_text = self._normalize_heading(line_text)
                        headings.append({
                            'text': normalized_text,
                            'level': heading_level,
                            'page': page_num,
                            'position': current_offset,
                            'font_size': max_font_size
                        })
                
                # Update position (approximate)
                current_offset += len(line_text) + 1  # +1 for newline
        
        return headings
    
    def _normalize_heading(self, text: str) -> str:
        """
        Normalize heading text by removing leading chapter numbers.
        
        Examples:
            "10. My Chapter" -> "My Chapter"
            "1. Introduction" -> "Introduction"
            "Chapter Title" -> "Chapter Title"
        """
        import re
        # Strip leading numbers followed by period and optional whitespace
        # Pattern: starts with one or more digits, followed by period and spaces
        normalized = re.sub(r'^\d+\.\s*', '', text)
        return normalized
    
    def _detect_heading_level(self, font_size: float, text: str, pdf_filename: str = "") -> int:
        """
        Detect if text is a heading based on font size and patterns.
        
        Args:
            font_size: Font size of the text
            text: Text content
            pdf_filename: Name of the PDF file (to check if H2-only mode)
            
        Returns:
            Heading level (1 or 2) or None if not a heading
        """
        # Skip specific heading texts that are duplicates (e.g., Chinese version of English heading)
        ignored_headings = [
            '電腦支援的協作學習：一個歷史脈絡的',  # Chinese duplicate of Traditional Chinese Translation
        ]
        if text.strip() in ignored_headings:
            return None
        
        # Skip very long lines (likely not headings)
        if len(text) > 150:
            return None
        
        # Skip very small fonts (definitely not headings)
        if font_size < 13.0:
            return None
        
        # Skip single characters or very short text (likely decorative)
        if len(text.strip()) <= 2:
            return None
            
        # Check for common heading patterns
        heading_patterns = [
            r'^Chapter\s+\d+',
            r'^\d+\.\s+[A-Z]',
            r'^[A-Z\s]{5,}$',  # ALL CAPS
            r'^[IVX]+\.',  # Roman numerals
        ]
        
        is_heading_pattern = any(re.match(pattern, text) for pattern in heading_patterns)
        
        # Check if this PDF should only use H2 headings as chapters
        h2_only_mode = pdf_filename in self.h2_only_pdfs
        h2_moderate_mode = pdf_filename in self.h2_moderate_pdfs
        h2_small_mode = pdf_filename in self.h2_small_pdfs
        
        # Classify by font size and patterns
        if h2_only_mode:
            # For H2-only PDFs like 15.global (language sections), only detect major headings
            # These are typically larger fonts (18pt+) for language section titles
            if font_size >= 17.5:  # Major section headers only
                return 2  # Treat as H2 (will become chapter)
            # Ignore smaller headings in H2-only mode
            return None
        elif h2_moderate_mode:
            # For PDFs like 17.proposals: detect moderate-sized headings (16pt+)
            # Skip "Part" headings - only want individual proposal chapters
            if text.strip().startswith('Part '):
                return None
            # to catch individual proposal chapters that are between Part headers and subsections
            if font_size >= 16.0:  # Moderate headings for individual chapters
                return 2  # Treat as H2 (will become chapter)
            # Ignore smaller headings
            return None
        elif h2_small_mode:
            # For PDFs like 9.cscl: detect smaller H2 headings (13pt+)
            if font_size >= 13.0:  # Smaller headings for chapters
                return 2  # Treat as H2 (will become chapter)
            # Ignore smaller headings
            return None
        else:
            # Normal mode: distinguish between H1 and H2
            if font_size >= self.h1_min_size:
                return 1  # H1 (Chapter)
            elif font_size >= self.h2_min_size:
                return 2  # H2 (Section)
            elif is_heading_pattern and font_size >= 13.0:
                return 2  # Pattern suggests it's at least a section
            
        return None
    
    def _combine_split_headings(self, headings: List[Dict]) -> List[Dict]:
        """
        Combine consecutive headings with the same font size on the same page.
        This handles headings that wrap to multiple lines.
        
        Args:
            headings: List of heading dictionaries
            
        Returns:
            List of combined heading dictionaries
        """
        if not headings:
            return headings
        
        combined = []
        i = 0
        
        while i < len(headings):
            current = headings[i]
            combined_text = current['text']
            
            # Look ahead to see if next headings should be combined
            j = i + 1
            while j < len(headings):
                next_heading = headings[j]
                
                # Combine if same page, same font size, and same level
                if (next_heading['page'] == current['page'] and 
                    abs(next_heading['font_size'] - current['font_size']) < 0.5 and
                    next_heading['level'] == current['level']):
                    # Add space between if current doesn't end with hyphen
                    if combined_text.endswith('-'):
                        combined_text = combined_text[:-1] + next_heading['text']
                    else:
                        combined_text += ' ' + next_heading['text']
                    j += 1
                else:
                    break
            
            # Add the combined heading
            combined.append({
                'text': combined_text,
                'level': current['level'],
                'page': current['page'],
                'position': current['position'],
                'font_size': current['font_size']
            })
            
            # Move to next uncombined heading
            i = j
        
        return combined
    
    def get_page_number(
        self,
        text_position: int,
        page_chunks: List[Dict]
    ) -> int:
        """
        Get page number for a given text position.
        
        Args:
            text_position: Character position in full document
            page_chunks: List of page chunk dictionaries
            
        Returns:
            Page number (1-indexed)
        """
        for chunk in page_chunks:
            if chunk['start_pos'] <= text_position < chunk['end_pos']:
                return chunk['page_number']
        
        # If not found, return last page
        return page_chunks[-1]['page_number'] if page_chunks else 1
    
    def get_heading_context(
        self, 
        text_position: int, 
        heading_structure: List[Dict],
        chapter_level: int = 1,
        min_chapter_font_size: float = None
    ) -> Dict[str, str]:
        """
        Get the current chapter/section context for a given text position.
        
        Args:
            text_position: Character position in full document
            heading_structure: List of heading dictionaries
            chapter_level: Which heading level to treat as chapters (default: 1)
            min_chapter_font_size: Minimum font size for chapter headings (filters out smaller subsections)
            
        Returns:
            Dictionary with current chapter and section
        """
        current_chapter = None
        current_section = None
        section_level = chapter_level + 1
        
        # Find the most recent headings before this position
        for heading in heading_structure:
            if heading['position'] > text_position:
                break
                
            if heading['level'] == chapter_level:
                # Apply font size filter if specified
                if min_chapter_font_size is not None:
                    font_size = heading.get('font_size', 0)
                    if font_size < min_chapter_font_size:
                        continue  # Skip this heading, it's too small
                
                current_chapter = heading['text']
                current_section = None  # Reset section when we hit a new chapter
            elif heading['level'] == section_level:
                current_section = heading['text']
        
        return {
            'chapter': current_chapter,
            'section': current_section
        }
