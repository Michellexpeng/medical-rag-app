#!/usr/bin/env python
"""
Medical Documents Batch Processor

Batch process all medical PDF textbooks in the medical-rag-app/data/raw/ directory.
Supports parallel processing, progress tracking, and error recovery.

Features:
1. Batch process all medical PDFs - Auto-discover and process all PDF files in data/raw/ directory
2. Parallel processing capability - Improve processing efficiency
3. Progress tracking - Real-time progress display
4. Error handling and recovery - Single file failure doesn't affect other files
5. Processing result statistics - Detailed success/failure statistics

Usage:
    python batch_medical_processor.py --all
    python batch_medical_processor.py --files "CT and MRI.pdf" "Liver imaging.pdf"
    python batch_medical_processor.py --all --max-workers 3
"""

import os
import argparse
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

# Add RAG-Anything project path
import sys
project_root = Path(__file__).parent.parent.parent / "RAG-Anything"
sys.path.append(str(project_root))

from raganything import RAGAnything, RAGAnythingConfig
from raganything.batch_parser import BatchParser
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, logger

from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=project_root / ".env", override=False)


class MedicalBatchProcessor:
    """Medical Documents Batch Processor"""
    
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        working_dir: str = "./rag_storage",
        max_workers: int = 2
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.working_dir = Path(working_dir)
        self.max_workers = max_workers
        self.results: Dict[str, Dict] = {}
        
        # Ensure working directory exists
        self.working_dir.mkdir(parents=True, exist_ok=True)

    async def setup_rag_system(self, document_id: str) -> RAGAnything:
        """Set up independent RAG system for each document"""
        
        # Create independent working directory for each document
        doc_working_dir = self.working_dir / f"medical_doc_{document_id}"
        
        config = RAGAnythingConfig(
            working_dir=str(doc_working_dir),
            parser="mineru",  # MinerU has better support for medical PDFs
            parse_method="auto",
            enable_image_processing=True,    # Process medical images
            enable_table_processing=True,    # Process medical data tables
            enable_equation_processing=True, # Process medical equations
        )

        # LLM model function
        def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            return openai_complete_if_cache(
                "gpt-4o-mini",
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=self.api_key,
                base_url=self.base_url,
                **kwargs,
            )

        # Vision model function
        def vision_model_func(
            prompt,
            system_prompt=None,
            history_messages=[],
            image_data=None,
            messages=None,
            **kwargs,
        ):
            if messages:
                return openai_complete_if_cache(
                    "gpt-4o",
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=messages,
                    api_key=self.api_key,
                    base_url=self.base_url,
                    **kwargs,
                )
            else:
                return openai_complete_if_cache(
                    "gpt-4o",
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    api_key=self.api_key,
                    base_url=self.base_url,
                    **kwargs,
                )

        # Embedding function - Use correct EmbeddingFunc class (use large model for best results)
        embedding_func = EmbeddingFunc(
            embedding_dim=3072,  # text-embedding-3-large dimensions
            max_token_size=8192,
            func=lambda texts: openai_embed(
                texts,
                model="text-embedding-3-large",
                api_key=self.api_key,
                base_url=self.base_url,
            ),
        )

        return RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
        )

    async def process_single_document(
        self, 
        file_path: Path, 
        output_dir: Path,
        document_id: str
    ) -> Dict:
        """Process single medical document"""
        start_time = time.time()
        result = {
            "file_name": file_path.name,
            "file_path": str(file_path),
            "status": "processing",
            "start_time": start_time,
            "error": None,
            "processing_time": 0,
            "output_dir": str(output_dir / file_path.stem)
        }
        
        try:
            logger.info(f"🏥 Starting processing: {file_path.name}")
            
            # Create independent output directory for each document
            doc_output_dir = output_dir / file_path.stem
            doc_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Setup RAG system
            rag = await self.setup_rag_system(document_id)
            
            # Process document
            await rag.process_document_complete(
                file_path=str(file_path),
                output_dir=str(doc_output_dir),
                parse_method="auto"
            )
            
            # Execute basic medical query validation
            test_query = "What is the main content of this document?"
            test_result = await rag.aquery(test_query, mode="hybrid")
            
            result.update({
                "status": "completed",
                "processing_time": time.time() - start_time,
                "test_query_result": test_result[:200] + "..." if len(test_result) > 200 else test_result
            })
            
            logger.info(f"✅ Completed processing: {file_path.name} (Time: {result['processing_time']:.2f}s)")
            
        except Exception as e:
            result.update({
                "status": "failed",
                "processing_time": time.time() - start_time,
                "error": str(e)
            })
            logger.error(f"❌ Processing failed: {file_path.name} - {str(e)}")
            
        return result

    async def process_documents_batch(
        self, 
        file_paths: List[Path], 
        output_dir: Path
    ) -> Dict:
        """Batch process medical documents"""
        
        logger.info(f"🚀 Starting batch processing of {len(file_paths)} medical documents")
        logger.info(f"📁 Output directory: {output_dir}")
        logger.info(f"⚡ Parallel workers: {self.max_workers}")
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_with_semaphore(file_path: Path, doc_id: str):
            async with semaphore:
                return await self.process_single_document(file_path, output_dir, doc_id)
        
        # Process all documents in parallel
        tasks = []
        for i, file_path in enumerate(file_paths):
            doc_id = f"med_doc_{i:03d}"
            task = process_with_semaphore(file_path, doc_id)
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        batch_results = {
            "total_documents": len(file_paths),
            "completed": 0,
            "failed": 0,
            "total_time": 0,
            "documents": {}
        }
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                file_name = file_paths[i].name
                batch_results["documents"][file_name] = {
                    "status": "failed", 
                    "error": str(result)
                }
                batch_results["failed"] += 1
            else:
                batch_results["documents"][result["file_name"]] = result
                if result["status"] == "completed":
                    batch_results["completed"] += 1
                else:
                    batch_results["failed"] += 1
                batch_results["total_time"] += result["processing_time"]
        
        return batch_results

    def print_batch_summary(self, results: Dict):
        """Print batch processing results summary"""
        print("\n" + "="*80)
        print("📊 Medical Documents Batch Processing Results Summary")
        print("="*80)
        print(f"📄 Total documents: {results['total_documents']}")
        print(f"✅ Successfully processed: {results['completed']}")
        print(f"❌ Processing failed: {results['failed']}")
        print(f"⏱️  Total time: {results['total_time']:.2f}s")
        print(f"📊 Success rate: {(results['completed']/results['total_documents']*100):.1f}%")
        
        if results['completed'] > 0:
            avg_time = results['total_time'] / results['completed']
            print(f"⚡ Average processing time: {avg_time:.2f}s/document")
        
        print("\n📋 Detailed results:")
        for doc_name, doc_result in results['documents'].items():
            status_emoji = "✅" if doc_result['status'] == 'completed' else "❌"
            print(f"{status_emoji} {doc_name}: {doc_result['status']}")
            if doc_result['status'] == 'completed':
                print(f"   ⏱️ Time: {doc_result['processing_time']:.2f}s")
            elif 'error' in doc_result:
                print(f"   ❌ Error: {doc_result['error']}")
        
        print("="*80)


def get_medical_pdf_files(data_dir: Path) -> List[Path]:
    """Get all PDF files in medical data directory"""
    raw_dir = data_dir / "raw"
    if not raw_dir.exists():
        logger.error(f"❌ Data directory does not exist: {raw_dir}")
        return []
    
    pdf_files = list(raw_dir.glob("*.pdf"))
    logger.info(f"📚 Found {len(pdf_files)} PDF files:")
    for pdf_file in pdf_files:
        logger.info(f"   📄 {pdf_file.name}")
    
    return pdf_files


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Medical Textbooks Batch RAG Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Process all PDF files in data/raw/ directory
  python batch_medical_processor.py --all
  
  # Process specific PDF files
  python batch_medical_processor.py --files "CT and MRI.pdf" "Liver imaging.pdf"
  
  # Set parallel workers and output directory
  python batch_medical_processor.py --all --max-workers 3 --output ./batch_output
        """
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all PDF files in data/raw/ directory"
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="Specify list of PDF file names to process"
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Medical data directory path (default: ./data)"
    )
    parser.add_argument(
        "--output",
        default="./batch_output",
        help="Batch processing output directory (default: ./batch_output)"
    )
    parser.add_argument(
        "--working-dir",
        default="./batch_rag_storage",
        help="RAG storage working directory (default: ./batch_rag_storage)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Maximum parallel workers (default: 2, recommend not exceeding 3)"
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (can be set via OPENAI_API_KEY environment variable)"
    )
    parser.add_argument(
        "--base-url",
        help="API base URL (can be set via OPENAI_BASE_URL environment variable)"
    )

    args = parser.parse_args()

    # Validate parameters
    if not args.all and not args.files:
        logger.error("❌ Please specify --all or --files parameter")
        return

    # Get API configuration
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL")
    
    if not api_key:
        logger.error("❌ OpenAI API key not provided. Please set OPENAI_API_KEY environment variable or use --api-key parameter")
        return

    # Get files to process
    data_dir = Path(args.data_dir)
    if args.all:
        file_paths = get_medical_pdf_files(data_dir)
    else:
        raw_dir = data_dir / "raw"
        file_paths = []
        for file_name in args.files:
            file_path = raw_dir / file_name
            if file_path.exists():
                file_paths.append(file_path)
            else:
                logger.warning(f"⚠️ File does not exist: {file_path}")

    if not file_paths:
        logger.error("❌ No processable PDF files found")
        return

    logger.info(f"🏥 Medical document batch processor starting")
    logger.info(f"📚 Files to process: {len(file_paths)}")
    logger.info(f"📁 Output directory: {args.output}")
    logger.info(f"⚡ Maximum parallel workers: {args.max_workers}")

    try:
        # Create batch processor
        processor = MedicalBatchProcessor(
            api_key=api_key,
            base_url=base_url,
            working_dir=args.working_dir,
            max_workers=args.max_workers
        )

        # Start batch processing
        start_time = time.time()
        results = await processor.process_documents_batch(
            file_paths=file_paths,
            output_dir=Path(args.output)
        )
        
        total_time = time.time() - start_time
        results["total_batch_time"] = total_time

        # Print results summary
        processor.print_batch_summary(results)
        
        logger.info(f"🎉 Batch processing completed! Total time: {total_time:.2f}s")

    except Exception as e:
        logger.error(f"❌ Batch processing error: {str(e)}")
        raise


if __name__ == "__main__":
    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    asyncio.run(main())