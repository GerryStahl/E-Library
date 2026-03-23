"""
Regenerate all vector stores from PDFs with correct chapter detection logic.

This script processes all 22 PDFs and creates vector stores with proper:
- Chapter-level overrides (H2 for 9.cscl, 15.global, 17.proposals)
- Font size filters (18.0pt for 15.global, 16.0pt for 17.proposals)
- Excluded chapter titles (Introduction, References, etc.)
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from main import PDFEmbeddingPipeline


def main():
    """Regenerate all vector stores from scratch."""
    
    pdf_dir = "../sourcepdfs"  # PDFs are in parent directory
    vector_store_dir = "./vector_stores"
    
    # Get list of PDFs
    pdf_files = sorted(Path(pdf_dir).glob("*.pdf"))
    pdf_names = [f.stem for f in pdf_files]
    
    logger.info("=" * 80)
    logger.info("REGENERATING ALL VECTOR STORES")
    logger.info("=" * 80)
    logger.info(f"Source: {pdf_dir}")
    logger.info(f"Destination: {vector_store_dir}")
    logger.info(f"Total PDFs: {len(pdf_names)}")
    logger.info("")
    
    # Create pipeline
    pipeline = PDFEmbeddingPipeline(
        pdf_dir=pdf_dir,
        vector_store_dir=vector_store_dir
    )
    
    # Process each PDF
    start_time = datetime.now()
    processed = 0
    failed = []
    
    for idx, pdf_name in enumerate(pdf_names, 1):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Processing {idx}/{len(pdf_names)}: {pdf_name}.pdf")
        logger.info(f"{'=' * 80}")
        
        try:
            # Check if PDF exists
            pdf_path = Path(pdf_dir) / f"{pdf_name}.pdf"
            if not pdf_path.exists():
                logger.warning(f"PDF not found: {pdf_path}")
                failed.append((pdf_name, "PDF not found"))
                continue
            
            # Process PDF
            logger.info(f"Parsing and chunking {pdf_name}.pdf...")
            pipeline.process_single_pdf(pdf_path)
            logger.info(f"✓ {pdf_name}: Completed")
            processed += 1
            
        except Exception as e:
            logger.error(f"✗ Failed to process {pdf_name}: {e}", exc_info=True)
            failed.append((pdf_name, str(e)))
    
    # Summary
    elapsed = datetime.now() - start_time
    
    logger.info(f"\n{'=' * 80}")
    logger.info("REGENERATION COMPLETE")
    logger.info(f"{'=' * 80}")
    logger.info(f"Time elapsed: {elapsed}")
    logger.info(f"Processed: {processed}/{len(pdf_names)}")
    logger.info(f"Failed: {len(failed)}")
    
    if failed:
        logger.info(f"\nFailed PDFs:")
        for pdf_name, error in failed:
            logger.info(f"  • {pdf_name}: {error}")
    
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Run extract_all_chapters_with_metrics.py to build chapter cache")
    logger.info(f"  2. Review reports/all_chapters_with_metrics.txt")
    logger.info(f"  3. Verify 15.global has 8 chapters, 9.cscl has 12, etc.")


if __name__ == "__main__":
    main()
