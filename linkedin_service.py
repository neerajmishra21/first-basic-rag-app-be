from serpapi import GoogleSearch
import os

def search_linkedin_profiles(name: str):
    query = name.strip()
    params = {
        "engine": "google",
        "q": f'site:linkedin.com/in/ "{query}"',
        "api_key": os.getenv("SERP_API_KEY"),
        "num":10
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    profiles = []

    organic_results = results.get("organic_results", [])

    for results in organic_results:
        link = results.get("link")

        if not link or "linkedin.com/in/" not in link:
            continue
              
        profiles.append({
            "title":results.get("title"),
            "linkedin_url":results.get("link"),
            "snippet": results.get("snippet")
        })

    return profiles