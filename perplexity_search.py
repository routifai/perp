#!/usr/bin/env python3
"""
Perplexity Search API - Raw Search Results with Chunks
Uses the official Perplexity SDK to get structured search results with chunks.
Best for: Getting detailed text excerpts (chunks) from web pages before AI processing.

Note: Perplexity API uses "chunks" terminology for text excerpts from web pages.
You can control the number of tokens using max_tokens (total) and max_tokens_per_page.
"""

import os
import sys
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

try:
    from perplexity import Perplexity
    PERPLEXITY_SDK_AVAILABLE = True
except ImportError:
    PERPLEXITY_SDK_AVAILABLE = False

load_dotenv()


def search_perplexity(
    query: str,
    api_key: Optional[str] = None,
    max_results: int = 10,
    max_tokens: Optional[int] = None,
    max_tokens_per_page: int = 2048,
    search_domain_filter: Optional[List[str]] = None,
    search_recency_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Query Perplexity Search API using the official SDK to get sources with chunks.
    
    This uses the Search API which returns raw search results with chunks (text excerpts)
    from web pages. No AI processing - just structured search data.
    
    Note: 
    - max_tokens_per_page controls the maximum number of tokens retrieved from each webpage
    - max_tokens sets the maximum total number of tokens across all search results
    - The API returns chunks (not snippets) - these are text excerpts from web pages
    
    Args:
        query: The search query
        api_key: Perplexity API key (defaults to PERPLEXITY_API_KEY env var)
        max_results: Maximum number of results to return (1-20)
        max_tokens: Maximum total tokens across all results (default: 25000, max: 1000000)
        max_tokens_per_page: Maximum tokens per page (default: 2048)
        search_domain_filter: List of domains to filter results (max 20 domains)
        search_recency_filter: Filter by recency - "day", "week", "month", or "year"
    
    Returns:
        dict: Response with sources, chunks, and results
    """
    if api_key is None:
        api_key = os.getenv("PERPLEXITY_API_KEY")
    
    if not api_key:
        raise ValueError(
            "Perplexity API key is required. "
            "Set it as PERPLEXITY_API_KEY environment variable or in .env file."
        )
    
    if not PERPLEXITY_SDK_AVAILABLE:
        raise Exception(
            "Perplexity SDK not available. Install with: pip install perplexityai"
        )
    
    try:
        client = Perplexity(api_key=api_key)
        
        search_params: Dict[str, Any] = {
            "query": query,
            "max_results": max_results,
            "max_tokens_per_page": max_tokens_per_page,
        }
        
        # Add max_tokens if provided (defaults to 25000 in API if not specified)
        if max_tokens is not None:
            search_params["max_tokens"] = max_tokens
        
        if search_domain_filter:
            search_params["search_domain_filter"] = search_domain_filter
        
        if search_recency_filter:
            search_params["search_recency_filter"] = search_recency_filter
        
        search = client.search.create(**search_params)
        
        # Inspect raw SDK response to understand available fields
        # This helps verify what Perplexity SDK actually returns
        if search.results and len(search.results) > 0:
            first_raw_result = search.results[0]
            # Store inspection info for debugging
            inspection_info = {
                "available_attrs": [attr for attr in dir(first_raw_result) if not attr.startswith('_')],
            }
            # Try to inspect the actual data
            if hasattr(first_raw_result, '__dict__'):
                inspection_info["__dict__"] = first_raw_result.__dict__
        
        results = []
        chunks = []
        sources = []
        
        for r in search.results:
            source = {
                "title": r.title,
                "url": r.url,
            }
            date = getattr(r, "date", None) or getattr(r, "last_updated", None)
            if date:
                source["date"] = date
            
            sources.append(source)
            
            # Perplexity API returns chunks (text excerpts from web pages)
            # Check for chunks field first (Perplexity's terminology)
            # The API documentation shows "snippet" but Perplexity internally refers to these as "chunks"
            chunk_text = None
            
            # Try different possible field names
            if hasattr(r, "chunks") and r.chunks:
                # If chunks is a list, join them
                if isinstance(r.chunks, list):
                    chunk_text = "\n\n".join(str(chunk) for chunk in r.chunks if chunk)
                else:
                    chunk_text = str(r.chunks) if r.chunks else None
            elif hasattr(r, "chunk") and r.chunk:
                chunk_text = str(r.chunk)
            elif hasattr(r, "snippet") and r.snippet:
                # Fallback to snippet (API response field name)
                chunk_text = str(r.snippet)
            else:
                # Last resort: try to get any text content
                chunk_text = None
            
            # Create chunk data (using Perplexity's terminology)
            if chunk_text:
                chunk_data = {
                    "title": r.title,
                    "url": r.url,
                    "source": r.url,
                    "chunk": chunk_text,  # Use "chunk" instead of "snippet"
                    "chunk_length": len(chunk_text),
                    "chunk_tokens": len(chunk_text.split())  # Approximate token count
                }
                chunks.append(chunk_data)
            
            result_item = {
                "title": r.title,
                "url": r.url,
                "chunk": chunk_text,  # Use "chunk" instead of "snippet"
                "date": date
            }
            
            if chunk_text:
                result_item["chunk_length"] = len(chunk_text)
                result_item["chunk_tokens"] = len(chunk_text.split())
            
            results.append(result_item)
        
        return {
            "sources": sources,
            "chunks": chunks,
            "results": results,
            "query": query,
            "_inspection_info": inspection_info if 'inspection_info' in locals() else None
        }
    
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg or "not found" in error_msg.lower():
            raise Exception(
                "Search API is not enabled for your API key or account. "
                "You can use perplexity_chat.py for Chat Completions API instead."
            )
        raise Exception(f"Error calling Perplexity Search API: {str(e)}")


def display_search_results(result: Dict[str, Any], max_sources: int = 20, max_chunks: int = 20) -> None:
    """Display search results: sources and chunks."""
    print(f"\n{'='*70}")
    print(f"🔍 SEARCH RESULTS: {result['query']}")
    print(f"{'='*70}")
    
    # Display sources
    sources = result.get("sources", [])
    if sources:
        print(f"\n📚 SOURCES ({len(sources)} found, showing {min(len(sources), max_sources)})")
        print("-" * 70)
        for idx, source in enumerate(sources[:max_sources], 1):
            title = source.get('title', 'Untitled')
            url = source.get('url', 'N/A')
            date = source.get('date')
            print(f"\n{idx}. {title}")
            print(f"   URL: {url}")
            if date:
                print(f"   Date: {date}")
    else:
        print("\n⚠️  No sources found")
    
    # Display chunks (Perplexity's terminology)
    chunks = result.get("chunks", [])
    if chunks:
        print(f"\n📄 CHUNKS ({len(chunks)} found, showing {min(len(chunks), max_chunks)})")
        print("-" * 70)
        total_chars = 0
        total_tokens = 0
        for idx, chunk_data in enumerate(chunks[:max_chunks], 1):
            print(f"\n--- Chunk {idx} ---")
            title = chunk_data.get('title', 'Untitled')
            url = chunk_data.get('url', 'N/A')
            chunk_text = chunk_data.get('chunk', '')
            chunk_length = chunk_data.get('chunk_length', len(chunk_text) if chunk_text else 0)
            chunk_tokens = chunk_data.get('chunk_tokens', 0)
            
            print(f"Title: {title}")
            print(f"URL: {url}")
            if chunk_text:
                print(f"Length: {chunk_length:,} characters (~{chunk_tokens:,} tokens)")
                print(f"\nChunk Content:")
                print(chunk_text)
                total_chars += chunk_length
                total_tokens += chunk_tokens
            else:
                print("⚠️  No chunk content available")
        
        if chunks:
            print(f"\n📊 Summary:")
            print(f"   Total characters: {total_chars:,}")
            print(f"   Total tokens (approx): {total_tokens:,}")
            print(f"   Average per chunk: {total_chars // len(chunks):,} characters (~{total_tokens // len(chunks):,} tokens)")
    else:
        print("\n⚠️  No chunks available")


def inspect_result_fields(result_obj) -> None:
    """Inspect all available fields in a search result object."""
    print(f"\n{'='*70}")
    print("🔍 INSPECTING RESULT OBJECT FIELDS")
    print(f"{'='*70}")
    
    if hasattr(result_obj, '__dict__'):
        print("\nAvailable attributes:")
        for attr in dir(result_obj):
            if not attr.startswith('_'):
                try:
                    value = getattr(result_obj, attr)
                    if not callable(value):
                        value_str = str(value)
                        if len(value_str) > 100:
                            value_str = value_str[:100] + "..."
                        print(f"  - {attr}: {value_str}")
                except:
                    pass


def test_max_tokens(query: str, api_key: str, test_values: List[int] = [1024, 2048, 4096, 8192]) -> None:
    """Test different max_tokens_per_page values to find optimal extraction."""
    print(f"\n{'='*70}")
    print("🧪 TESTING DIFFERENT max_tokens_per_page VALUES")
    print(f"{'='*70}")
    
    for max_tokens_per_page in test_values:
        try:
            result = search_perplexity(
                query,
                api_key=api_key,
                max_results=3,
                max_tokens_per_page=max_tokens_per_page
            )
            
            chunks = result.get("chunks", [])
            if chunks:
                total_chars = sum(chunk.get('chunk_length', 0) for chunk in chunks)
                total_tokens = sum(chunk.get('chunk_tokens', 0) for chunk in chunks)
                avg_chars = total_chars // len(chunks) if chunks else 0
                avg_tokens = total_tokens // len(chunks) if chunks else 0
                print(f"\nmax_tokens_per_page={max_tokens_per_page}:")
                print(f"  - Chunks: {len(chunks)}")
                print(f"  - Total chars: {total_chars:,}")
                print(f"  - Total tokens (approx): {total_tokens:,}")
                print(f"  - Avg chars/chunk: {avg_chars:,}")
                print(f"  - Avg tokens/chunk: {avg_tokens:,}")
            else:
                print(f"\nmax_tokens_per_page={max_tokens_per_page}: No chunks returned")
        except Exception as e:
            print(f"\nmax_tokens_per_page={max_tokens_per_page}: Error - {e}")


def main():
    """Main function - Search API only."""
    import sys
    
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "Latest developments in artificial intelligence 2025"

    api_key = os.getenv("PERPLEXITY_API_KEY")
    
    if not api_key:
        print("\n❌ Error: PERPLEXITY_API_KEY not set")
        print("\n💡 To set the API key, run:")
        print("   export PERPLEXITY_API_KEY='your-api-key-here'")
        return

    try:
        result = search_perplexity(
            query, 
            api_key=api_key,
            max_results=10,
            max_tokens=25000,  # Total tokens across all results
            max_tokens_per_page=2048  # Tokens per page (default)
        )
        display_search_results(result)
        
        # Inspect the raw SDK response to see what fields Perplexity actually returns
        if result.get("_inspection_info"):
            print(f"\n{'='*70}")
            print("🔍 INSPECTING RAW SDK RESPONSE FIELDS")
            print(f"{'='*70}")
            info = result["_inspection_info"]
            print("\nAvailable attributes in SDK result object:")
            for attr in info.get("available_attrs", []):
                print(f"  - {attr}")
            if "__dict__" in info:
                print("\nSDK result __dict__ contents:")
                for key, value in info["__dict__"].items():
                    if isinstance(value, str) and len(value) > 100:
                        print(f"  - {key}: {value[:100]}... ({len(value)} chars)")
                    else:
                        print(f"  - {key}: {value}")
        
        # Inspect the processed result fields
        if result.get("results") and len(result["results"]) > 0:
            print(f"\n{'='*70}")
            print("🔍 PROCESSED RESULT FIELDS")
            print(f"{'='*70}")
            first_result = result["results"][0]
            print("\nAvailable fields in processed result:")
            for key in first_result.keys():
                value = first_result[key]
                if isinstance(value, str) and len(value) > 100:
                    print(f"  - {key}: {value[:100]}... ({len(value)} chars)")
                else:
                    print(f"  - {key}: {value}")
        
        if result.get("chunks") and PERPLEXITY_SDK_AVAILABLE:
            print(f"\n{'='*70}")
            print("💡 TIP: To test different extraction sizes, uncomment the test below")
            print(f"{'='*70}")
            # Uncomment to test different max_tokens_per_page values:
            # test_max_tokens(query, api_key, [1024, 2048, 4096, 8192, 16384])
        
        print(f"\n{'='*70}")
        print("💡 This is raw search data from Perplexity Search API")
        print("   - max_tokens: Total tokens across all results (default: 25000, max: 1000000)")
        print("   - max_tokens_per_page: Tokens per page (default: 2048)")
        print("   - The API returns 'chunks' (text excerpts from web pages)")
        print("   - Higher values extract more content but may increase processing time")
        print("   - For AI-generated answers, use perplexity_chat.py")
        print(f"{'='*70}\n")

    except ValueError as e:
        print(f"\n❌ Error: {e}")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()

