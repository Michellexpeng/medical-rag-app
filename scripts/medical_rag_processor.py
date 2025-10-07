#!/usr/bin/env python
"""
Medical RAG Processor

Specialized RAG system for processing medical textbook PDF documents with multimodal content support.
Suitable for processing complex textbooks containing medical images, tables, and formulas.

Features:
1. Medical PDF document processing - Supports imaging, anatomy and other professional textbooks
2. Multimodal content support - Process medical images, data tables, formulas
3. Medical professional queries - Query examples optimized for medical concepts
4. Intelligent parsing methods - Automatically select optimal parsing strategies

Usage:
    python medical_rag_processor.py --file "data/raw/CT and MRI of the Whole Body.pdf" --query "What is CT imaging?"
    python medical_rag_processor.py --file "data/raw/Liver imaging.pdf" --interactive
"""

import os
import argparse
import asyncio
import logging
import logging.config
from pathlib import Path
from typing import Optional, List, Dict

# Add RAG-Anything project path
import sys
project_root = Path(__file__).parent.parent.parent / "RAG-Anything"
sys.path.append(str(project_root))

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from raganything import RAGAnything, RAGAnythingConfig

from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=project_root / ".env", override=False)


def configure_logging():
    """Configure logging system"""
    log_dir = Path(__file__).parent.parent / "logs"
    log_file_path = log_dir / "medical_rag_processor.log"
    
    print(f"\n📋 Medical RAG processor log file: {log_file_path}\n")
    os.makedirs(log_dir, exist_ok=True)

    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "simple": {
                "format": "%(levelname)s - %(message)s",
            },
        },
        "handlers": {
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": log_file_path,
                "maxBytes": log_max_bytes,
                "backupCount": log_backup_count,
                "formatter": "detailed",
                "level": "DEBUG",
            },
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "simple",
                "level": "INFO",
            },
        },
        "root": {
            "level": "DEBUG",
            "handlers": ["file", "console"],
        },
    })


async def process_medical_document(
    file_path: str,
    output_dir: str = "./output",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    working_dir: Optional[str] = None,
    parser: str = "mineru"
):
    """
    Process medical documents and set up RAG system
    
    Args:
        file_path: Medical PDF document path
        output_dir: Output directory
        api_key: OpenAI API key
        base_url: API base URL
        working_dir: RAG storage working directory
        parser: Parser type (mineru or docling)
    """
    try:
        # Create medical RAG configuration - optimized for large medical documents
        config = RAGAnythingConfig(
            working_dir=working_dir or "./rag_storage",
            parser=parser,  # MinerU has better support for medical PDFs
            parse_method="auto",  # Automatically select parsing method
            enable_image_processing=True,    # Process medical images
            enable_table_processing=True,    # Process medical data tables
            enable_equation_processing=True, # Process medical equations
            # Optimize large document processing performance
            max_concurrent_files=1,  # Single file processing, avoid resource competition
        )

        # Define LLM model function
        def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            return openai_complete_if_cache(
                "gpt-4o-mini",
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )

        # Define vision model function - for processing medical images
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
                    api_key=api_key,
                    base_url=base_url,
                    **kwargs,
                )
            else:
                return openai_complete_if_cache(
                    "gpt-4o",
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    api_key=api_key,
                    base_url=base_url,
                    **kwargs,
                )

        # Define embedding function - use correct EmbeddingFunc class (use small model for consistency)
        embedding_func = EmbeddingFunc(
            embedding_dim=1536,  # text-embedding-3-small dimensions
            max_token_size=8192,
            func=lambda texts: openai_embed(
                texts,
                model="text-embedding-3-small",
                api_key=api_key,
                base_url=base_url,
            ),
        )

        # Create RAG instance - optimized configuration for large medical documents
        import lightrag.utils
        
        # Temporarily increase timeout for processing large medical documents
        original_timeout = getattr(lightrag.utils, 'HEALTH_CHECK_TIMEOUT', 75)
        lightrag.utils.HEALTH_CHECK_TIMEOUT = 600  # Increase to 10 minutes
        
        try:
            rag = RAGAnything(
                config=config,
                llm_model_func=llm_model_func,
                vision_model_func=vision_model_func,
                embedding_func=embedding_func,
            )
        finally:
            # Restore original timeout settings
            lightrag.utils.HEALTH_CHECK_TIMEOUT = original_timeout

        # Process medical document
        logger.info(f"🏥 Starting medical document processing: {Path(file_path).name}")
        await rag.process_document_complete(
            file_path=file_path, 
            output_dir=output_dir, 
            parse_method="auto"
        )
        logger.info("✅ Medical document processing completed!")

        return rag

    except Exception as e:
        logger.error(f"❌ Error processing medical document: {str(e)}")
        raise


async def medical_query_examples(rag: RAGAnything):
    """Medical professional query examples"""
    logger.info("\n🔍 Starting medical professional query examples:")

    # 1. Basic medical concept queries
    basic_queries = [
        "What is the main content of this document? What medical topics does it contain?",
        "What imaging examination methods are mentioned in the document?",
        "What important anatomical structures are discussed?",
        "What diseases or pathological conditions are involved in the document?"
    ]

    for query in basic_queries:
        logger.info(f"\n📚 [Basic Query]: {query}")
        try:
            result = await rag.aquery(query, mode="hybrid")
            logger.info(f"💡 Answer: {result}")
        except Exception as e:
            logger.error(f"❌ Query error: {str(e)}")

    # 2. Medical image related queries
    image_queries = [
        "What do the medical images in the document show? Please describe in detail.",
        "What imaging signs are discussed as key points?",
        "What are the differences and characteristics between CT and MRI images?"
    ]

    for query in image_queries:
        logger.info(f"\n🖼️ [Image Query]: {query}")
        try:
            result = await rag.aquery(query, mode="hybrid")
            logger.info(f"💡 Answer: {result}")
        except Exception as e:
            logger.error(f"❌ Query error: {str(e)}")

    # 3. Multimodal medical query examples
    logger.info("\n🔬 [Multimodal Query]: Analyze clinical data table")
    try:
        multimodal_result = await rag.aquery_with_multimodal(
            "Please compare and analyze this clinical data table with relevant information in the document",
            multimodal_content=[
                {
                    "type": "table",
                    "table_data": """Test Item,Normal Value,Abnormal Value,Clinical Significance
                            Liver CT Value,45-65 HU,<45 or >65 HU,Fatty liver or iron deposition
                            Spleen Size,Long axis <12cm,>12cm,Splenomegaly
                            Portal Vein Diameter,<13mm,>13mm,Portal hypertension
                            Ascites,None,Present,Abdominal fluid accumulation""",
                    "table_caption": "Abdominal CT Examination Reference Values",
                }
            ],
            mode="hybrid",
        )
        logger.info(f"💡 Answer: {multimodal_result}")
    except Exception as e:
        logger.error(f"❌ Multimodal query error: {str(e)}")


async def interactive_medical_query(rag: RAGAnything):
    """Interactive medical query mode"""
    logger.info("\n🩺 Entering interactive medical query mode (type 'quit' or 'exit' to exit)")
    
    while True:
        try:
            query = input("\n❓ Please enter your medical question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                logger.info("👋 Exiting interactive mode")
                break
                
            if not query:
                continue
                
            logger.info(f"🔍 Querying: {query}")
            result = await rag.aquery(query, mode="hybrid")
            logger.info(f"💡 Answer: {result}")
            
        except KeyboardInterrupt:
            logger.info("\n👋 User interrupted, exiting interactive mode")
            break
        except Exception as e:
            logger.error(f"❌ Query error: {str(e)}")


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Medical Textbook RAG Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Process single medical PDF and run query examples
  python medical_rag_processor.py --file "data/raw/CT and MRI of the Whole Body.pdf"
  
  # Interactive query mode
  python medical_rag_processor.py --file "data/raw/Liver imaging.pdf" --interactive
  
  # Specify output directory and working directory
  python medical_rag_processor.py --file "data/raw/Diagnostic Imaging_ Abdomen.pdf" --output ./medical_output --working-dir ./medical_rag_storage
        """
    )
    
    parser.add_argument(
        "--file", 
        required=True, 
        help="Medical PDF document path"
    )
    parser.add_argument(
        "--output", 
        default="./output", 
        help="Output directory (default: ./output)"
    )
    parser.add_argument(
        "--working-dir", 
        default="./rag_storage", 
        help="RAG storage working directory (default: ./rag_storage)"
    )
    parser.add_argument(
        "--parser", 
        choices=["mineru", "docling"], 
        default="mineru",
        help="Document parser (default: mineru, recommended for medical PDFs)"
    )
    parser.add_argument(
        "--interactive", 
        action="store_true",
        help="Enable interactive query mode"
    )
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (can be set via OPENAI_API_KEY environment variable)"
    )
    parser.add_argument(
        "--base-url",
        help="API base URL (can be set via OPENAI_BASE_URL environment variable)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose debug logging"
    )

    args = parser.parse_args()

    # Configure logging
    configure_logging()
    
    if args.verbose:
        set_verbose_debug(True)

    # Get API configuration
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL")
    
    if not api_key:
        logger.error("❌ OpenAI API key not provided. Please set OPENAI_API_KEY environment variable or use --api-key parameter")
        return

    # Check if file exists
    if not os.path.exists(args.file):
        logger.error(f"❌ File does not exist: {args.file}")
        return

    logger.info(f"🏥 Medical RAG processor starting")
    logger.info(f"📄 Document: {args.file}")
    logger.info(f"📁 Output directory: {args.output}")
    logger.info(f"🔧 Parser: {args.parser}")

    try:
        # Process medical document
        rag = await process_medical_document(
            file_path=args.file,
            output_dir=args.output,
            api_key=api_key,
            base_url=base_url,
            working_dir=args.working_dir,
            parser=args.parser
        )

        # Execute queries based on mode
        if args.interactive:
            await interactive_medical_query(rag)
        else:
            await medical_query_examples(rag)

        logger.info("🎉 Medical RAG processing completed!")

    except Exception as e:
        logger.error(f"❌ Program execution error: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())