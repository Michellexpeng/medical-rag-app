#!/usr/bin/env python
"""
Medical RAG Query Tool

Directly query processed medical document data without reprocessing documents, suitable for rapid testing and RAG validation.

Usage:
    python medical_rag_query.py --working-dir ./rag_storage
    python medical_rag_query.py --working-dir ./rag_storage --interactive
    python medical_rag_query.py --working-dir ./rag_storage --query "What are the main imaging techniques discussed?"
"""

import os
import argparse
import asyncio
import logging
import time
from pathlib import Path
from typing import Optional, List

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
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )


async def create_rag_instance(
    working_dir: str,
    api_key: str,
    base_url: Optional[str] = None
) -> RAGAnything:
    """
    Create RAG instance using existing processed data
    
    Args:
        working_dir: Working directory with processed data
        api_key: OpenAI API key
        base_url: API base URL
    """
    try:
        # Check if working directory exists
        if not os.path.exists(working_dir):
            raise FileNotFoundError(f"Working directory does not exist: {working_dir}")
            
        logger.info(f"🔍 Using processed data directory: {working_dir}")

        # Create configuration - use existing data, no reprocessing
        config = RAGAnythingConfig(
            working_dir=working_dir,
            parser="mineru",
            parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
            max_concurrent_files=1,
        )

        # LLM model function
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

        # Vision model function - use smaller model to reduce token usage
        def vision_model_func(
            prompt,
            system_prompt=None,
            history_messages=[],
            **kwargs,
        ):
            if isinstance(prompt, list):
                # Multimodal query with images - use gpt-4o-mini to reduce cost and token usage
                return openai_complete_if_cache(
                    "gpt-4o-mini",  # Use smaller model
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    api_key=api_key,
                    base_url=base_url,
                    **kwargs,
                )
            else:
                # Pure text query - fix parameter order
                return openai_complete_if_cache(
                    "gpt-4o-mini",  # Add required first parameter (model name)
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    api_key=api_key,
                    base_url=base_url,
                    **kwargs,
                )

        # Embedding function - match processed data dimensions (1536)
        embedding_func = EmbeddingFunc(
            embedding_dim=1536,  # Match processed data dimensions
            max_token_size=8192,
            func=lambda texts: openai_embed(
                texts,
                model="text-embedding-3-small",  # Match processed data model
                api_key=api_key,
                base_url=base_url,
            ),
        )

        # Create RAG instance
        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
        )

        # Ensure LightRAG instance is initialized
        await rag._ensure_lightrag_initialized()
        
        # Check if processed data exists
        if rag.lightrag is None:
            raise Exception("LightRAG instance not properly initialized")
            
        logger.info("✅ RAG instance created successfully, ready for queries")
        return rag

    except Exception as e:
        logger.error(f"❌ Failed to create RAG instance: {str(e)}")
        raise


async def run_english_medical_queries(rag: RAGAnything):
    """Run English medical query examples"""
    logger.info("\n🔍 Running English Medical Query Examples:")

    # 1. Basic Medical Content Queries - Reduced to avoid rate limits
    basic_queries = [
        "What is the main content of this medical textbook?",
        "What imaging techniques are discussed in this document?",
        "What anatomical structures are covered in detail?"
    ]

    logger.info("\n📚 [Basic Medical Queries]:")
    for i, query in enumerate(basic_queries, 1):
        logger.info(f"\nQuery {i}: {query}")
        try:
            # Use very conservative parameters to avoid rate limits
            result = await rag.aquery(
                query, 
                mode="local",  # Use local mode instead of hybrid to reduce complexity
                top_k=4,  # Further reduce retrieved chunks
                chunk_top_k=4,  # Reduce chunk retrieval
                max_entity_tokens=1500,  # Much smaller entity token limit
                max_relation_tokens=2000,  # Much smaller relation token limit
                max_total_tokens=8000,  # Much smaller total token limit
                enable_rerank=False  # Disable rerank to save tokens
            )
            logger.info(f"Answer: {result}")
            logger.info("-" * 80)
            
            # Add delay between queries to prevent rate limiting
            if i < len(basic_queries):
                logger.info("⏳ Waiting 10 seconds to avoid rate limits...")
                await asyncio.sleep(10)
        except Exception as e:
            logger.error(f"❌ Query failed: {str(e)}")

    # 2. Technical Medical Queries - Reduced to avoid rate limits
    technical_queries = [
        "What are the differences between CT and MRI imaging described in this document?",
        "What are the main radiological findings discussed?"
    ]

    logger.info("\n🔬 [Technical Medical Queries]:")
    for i, query in enumerate(technical_queries, 1):
        logger.info(f"\nQuery {i}: {query}")
        try:
            # Use very conservative parameters to avoid rate limits
            result = await rag.aquery(
                query, 
                mode="local",  # Use local mode instead of hybrid
                top_k=4,  # Further reduce retrieved chunks
                chunk_top_k=4,  # Reduce chunk retrieval
                max_entity_tokens=1500,  # Much smaller entity token limit
                max_relation_tokens=2000,  # Much smaller relation token limit
                max_total_tokens=8000,  # Much smaller total token limit
                enable_rerank=False  # Disable rerank to save tokens
            )
            logger.info(f"Answer: {result}")
            logger.info("-" * 80)
            
            # Add delay between queries to prevent rate limiting
            if i < len(technical_queries):
                logger.info("⏳ Waiting 10 seconds to avoid rate limits...")
                await asyncio.sleep(10)
        except Exception as e:
            logger.error(f"❌ Query failed: {str(e)}")

    # 3. Clinical Application Queries - Reduced to avoid rate limits
    clinical_queries = [
        "What clinical scenarios are presented in this textbook?",
        "What diagnostic criteria are mentioned for various conditions?"
    ]

    logger.info("\n🏥 [Clinical Application Queries]:")
    for i, query in enumerate(clinical_queries, 1):
        logger.info(f"\nQuery {i}: {query}")
        try:
            # Use very conservative parameters to avoid rate limits
            result = await rag.aquery(
                query, 
                mode="local",  # Use local mode instead of hybrid
                top_k=4,  # Further reduce retrieved chunks
                chunk_top_k=4,  # Reduce chunk retrieval
                max_entity_tokens=1500,  # Much smaller entity token limit
                max_relation_tokens=2000,  # Much smaller relation token limit
                max_total_tokens=8000,  # Much smaller total token limit
                enable_rerank=False  # Disable rerank to save tokens
            )
            logger.info(f"Answer: {result}")
            logger.info("-" * 80)
            
            # Add delay between queries to prevent rate limiting
            if i < len(clinical_queries):
                logger.info("⏳ Waiting 10 seconds to avoid rate limits...")
                await asyncio.sleep(10)
        except Exception as e:
            logger.error(f"❌ Query failed: {str(e)}")


async def run_multimodal_query_example(rag: RAGAnything):
    """Run multimodal query example"""
    logger.info("\n🔬 [Multimodal Query Example]: Clinical Data Analysis")
    
    try:
        multimodal_result = await rag.aquery_with_multimodal(
            "Compare this clinical reference table with the imaging criteria mentioned in the document. Are these values consistent with the textbook recommendations?",
            multimodal_content=[
                {
                    "type": "table",
                    "table_data": """Parameter,Normal_Range,Abnormal_Finding,Clinical_Significance
                            Liver_CT_Value,45-65_HU,<45_or_>65_HU,Fatty_liver_or_iron_deposition
                            Spleen_Size,Length<12cm,>12cm,Splenomegaly
                            Portal_Vein_Diameter,<13mm,>13mm,Portal_hypertension
                            Ascites,Absent,Present,Peritoneal_fluid""",
                    "table_caption": "Abdominal CT Reference Values",
                }
            ],
            mode="local",  # Use local mode instead of hybrid
            top_k=3,  # Very small for multimodal queries
            chunk_top_k=3,  # Reduce chunk retrieval
            max_entity_tokens=1200,  # Much smaller entity token limit
            max_relation_tokens=1500,  # Much smaller relation token limit
            max_total_tokens=6000,  # Much smaller total token limit
            enable_rerank=False  # Disable rerank to save tokens
        )
        logger.info(f"Answer: {multimodal_result}")
    except Exception as e:
        logger.error(f"❌ Multimodal query failed: {str(e)}")


async def interactive_query_mode(rag: RAGAnything):
    """Interactive query mode"""
    logger.info("\n🩺 Interactive Medical Query Mode (Type 'quit' or 'exit' to exit)")
    logger.info("💡 You can ask questions in English or Chinese!")
    
    while True:
        try:
            query = input("\n❓ Enter your medical question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                logger.info("👋 Exiting interactive mode")
                break
                
            if not query:
                continue
                
            logger.info(f"🔍 Processing query: {query}")
            result = await rag.aquery(
                query, 
                mode="local",
                top_k=4,
                chunk_top_k=4,
                max_entity_tokens=1500,
                max_relation_tokens=2000,
                max_total_tokens=8000,
                enable_rerank=False
            )
            logger.info(f"💡 Answer: {result}")
            
        except KeyboardInterrupt:
            logger.info("\n👋 User interrupted, exiting interactive mode")
            break
        except Exception as e:
            logger.error(f"❌ Query error: {str(e)}")


async def single_query(rag: RAGAnything, query: str):
    """Execute single query"""
    logger.info(f"🔍 Single Query: {query}")
    try:
        result = await rag.aquery(
            query, 
            mode="local",
            top_k=4,
            chunk_top_k=4,
            max_entity_tokens=1500,
            max_relation_tokens=2000,
            max_total_tokens=8000,
            enable_rerank=False
        )
        logger.info(f"💡 Answer: {result}")
    except Exception as e:
        logger.error(f"❌ Query failed: {str(e)}")


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Medical RAG Query Tool - Query processed data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Run predefined English query examples
  python medical_rag_query.py --working-dir ./rag_storage
  
  # Interactive query mode
  python medical_rag_query.py --working-dir ./rag_storage --interactive
  
  # Execute single query
  python medical_rag_query.py --working-dir ./rag_storage --query "What are the main imaging techniques?"
  
  # Include multimodal query examples
  python medical_rag_query.py --working-dir ./rag_storage --include-multimodal
        """
    )
    
    parser.add_argument(
        "--working-dir", 
        required=True,
        help="Working directory path with processed data (e.g., ./rag_storage)"
    )
    parser.add_argument(
        "--query",
        help="Execute single query (if provided, will only execute this query)"
    )
    parser.add_argument(
        "--interactive", 
        action="store_true",
        help="Enable interactive query mode"
    )
    parser.add_argument(
        "--include-multimodal", 
        action="store_true",
        help="Include multimodal query examples"
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
        logger.error("❌ OpenAI API key is required. Set OPENAI_API_KEY environment variable or use --api-key")
        return

    logger.info("🔍 Medical RAG Query Tool Starting")
    logger.info(f"📁 Working directory: {args.working_dir}")

    try:
        # Create RAG instance
        rag = await create_rag_instance(
            working_dir=args.working_dir,
            api_key=api_key,
            base_url=base_url
        )

        # Execute different modes based on parameters
        if args.query:
            # Single query mode
            await single_query(rag, args.query)
        elif args.interactive:
            # Interactive query mode
            await interactive_query_mode(rag)
        else:
            # Predefined query examples mode
            await run_english_medical_queries(rag)
            
            if args.include_multimodal:
                await run_multimodal_query_example(rag)

        logger.info("🎉 Query session completed!")

    except Exception as e:
        logger.error(f"❌ Program error: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())