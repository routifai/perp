#!/usr/bin/env python3
"""
Simple Perplexity Search API script
Uses the Search API directly via HTTP requests (no SDK).
Just send a query and get JSON results.
"""

import os
import json
import sys
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
import httpx

load_dotenv()


def search_perplexity(
    query: str,
    max_results: int = 10,
    max_tokens: Optional[int] = None,
    max_tokens_per_page: int = 2048,
    search_domain_filter: Optional[List[str]] = None,
    search_recency_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Search the web using Perplexity's Search API.
    
    Args:
        query: The search query
        max_results: Maximum number of results (1-20, default: 10)
        max_tokens: Maximum total tokens across all results (default: 25000)
        max_tokens_per_page: Maximum tokens per page (default: 2048)
        search_domain_filter: List of domains to filter (max 20)
        search_recency_filter: "day", "week", "month", or "year"
    
    Returns:
        Dictionary with search results
    """
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        raise ValueError(
            "PERPLEXITY_API_KEY environment variable is not set. "
            "Set it in your .env file or environment."
        )
    
    # Prepare request payload
    payload: Dict[str, Any] = {
        "query": query,
        "max_results": max_results,
        "max_tokens_per_page": max_tokens_per_page,
    }
    
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    
    if search_domain_filter:
        payload["search_domain_filter"] = search_domain_filter
    
    if search_recency_filter:
        payload["search_recency_filter"] = search_recency_filter
    
    # Make API request
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                "https://api.perplexity.ai/search",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise Exception(
                "Search API is not enabled for your API key. "
                "Please check your Perplexity API subscription."
            )
        raise Exception(f"Perplexity API error: {e.response.status_code} - {e.response.text}")
    except httpx.RequestError as e:
        raise Exception(f"Network error: {str(e)}")


def main():
    """Main function - simple search and JSON output."""
    if len(sys.argv) < 2:
        print("Usage: python perplexity_search_simple.py <query> [max_results]")
        print("Example: python perplexity_search_simple.py 'latest AI news' 5")
        sys.exit(1)
    
    # Check if last argument is a number (max_results)
    max_results = 10
    query_parts = sys.argv[1:]
    
    if len(query_parts) > 1 and query_parts[-1].isdigit():
        max_results = int(query_parts[-1])
        query = " ".join(query_parts[:-1])
    else:
        query = " ".join(query_parts)
    
    try:
        result = search_perplexity(query, max_results=max_results)
        
        # Output as JSON
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

