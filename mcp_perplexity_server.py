#!/usr/bin/env python3
"""
MCP Server for Perplexity Search API
Uses fastmcp to create an MCP server with a search_web tool that calls
Perplexity Search API directly (not using the SDK).
"""

import os
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from fastmcp import FastMCP
import httpx

load_dotenv()

# Initialize the MCP server
mcp = FastMCP(
    name="Perplexity Search MCP Server",
    instructions="Provides web search for current events, news, information verification, and up-to-date content. Use this for any query about recent developments, facts that need verification, or information that may be after the knowledge cutoff."
)


@mcp.tool()
def search_web(
    query: str,
    max_results: int = 10,
    max_tokens: Optional[int] = None,
    max_tokens_per_page: int = 2048,
    search_domain_filter: Optional[List[str]] = None,
    search_recency_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Search the web for current information, news, and verified facts.
    
    Use this tool whenever you need:
    - Current events, news, or recent developments
    - Information that may be after your knowledge cutoff date
    - Verification of facts, claims, or information
    - Up-to-date data, statistics, or research findings
    - Latest updates on any topic, person, or organization
    
    This tool searches the web and returns results with text excerpts from relevant sources,
    allowing you to access the most current and accurate information available online.
    
    Args:
        query: What to search for (e.g., "latest AI developments", "weather in Toronto", "2025 election results")
        max_results: How many results to return (1-20, default: 10)
        max_tokens: Total content length across all results - higher values return more text (default: 25000)
        max_tokens_per_page: Content length per result - higher values return longer excerpts (default: 2048)
        search_domain_filter: Limit results to specific websites (e.g., ["wikipedia.org", "github.com"])
        search_recency_filter: Get only recent content - "day", "week", "month", or "year"
    
    Returns:
        Search results with sources, text excerpts, and metadata
    """
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        raise ValueError(
            "PERPLEXITY_API_KEY environment variable is not set. "
            "Please set it in your .env file or environment."
        )
    
    # Prepare the request payload
    payload: Dict[str, Any] = {
        "query": query,
        "max_results": max_results,
        "max_tokens_per_page": max_tokens_per_page,
    }
    
    # Add optional parameters
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    
    if search_domain_filter:
        payload["search_domain_filter"] = search_domain_filter
    
    if search_recency_filter:
        payload["search_recency_filter"] = search_recency_filter
    
    # Make the API request
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
            data = response.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise Exception(
                "Search API is not enabled for your API key or account. "
                "Please check your Perplexity API subscription."
            )
        raise Exception(f"Perplexity API error: {e.response.status_code} - {e.response.text}")
    except httpx.RequestError as e:
        raise Exception(f"Network error calling Perplexity API: {str(e)}")
    
    # Process the results
    results = data.get("results", [])
    
    sources = []
    chunks = []
    processed_results = []
    
    for result in results:
        # Extract source information
        source = {
            "title": result.get("title", "Untitled"),
            "url": result.get("url", ""),
        }
        
        date = result.get("date") or result.get("last_updated")
        if date:
            source["date"] = date
        
        sources.append(source)
        
        # Extract chunk (snippet) - Perplexity calls these chunks conceptually
        chunk_text = result.get("snippet")
        
        if chunk_text:
            # Create chunk data
            chunk_data = {
                "title": result.get("title", "Untitled"),
                "url": result.get("url", ""),
                "source": result.get("url", ""),
                "chunk": chunk_text,  # Using "chunk" terminology as Perplexity does
                "chunk_length": len(chunk_text),
                "chunk_tokens": len(chunk_text.split()),  # Approximate token count
            }
            
            if date:
                chunk_data["date"] = date
            
            chunks.append(chunk_data)
        
        # Create processed result
        result_item = {
            "title": result.get("title", "Untitled"),
            "url": result.get("url", ""),
            "chunk": chunk_text,
            "date": date
        }
        
        if chunk_text:
            result_item["chunk_length"] = len(chunk_text)
            result_item["chunk_tokens"] = len(chunk_text.split())
        
        processed_results.append(result_item)
    
    return {
        "query": query,
        "sources": sources,
        "chunks": chunks,
        "results": processed_results,
        "total_sources": len(sources),
        "total_chunks": len(chunks),
    }


if __name__ == "__main__":
    # Run the MCP server
    # This will start the server and handle MCP protocol communication
    mcp.run()

