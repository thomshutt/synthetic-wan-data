"""
CivitAI LoRA Scraper - Stage 1 of the synthetic data pipeline.

Scrapes Flux LoRAs and prompt metadata from CivitAI for use in Stage 2 dataset generation.

Usage:
    python -m src.stage1_scrape.scraper https://civitai.com/models/317208/zombie-style-fluxsdxl
"""

import argparse
import json
import os
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests
from dotenv import load_dotenv


def sanitize_prompt(prompt: str) -> str:
    """Remove LoRA invocation tags and clean up the prompt.

    Removes patterns like <lora:name:weight> which are specific to
    certain image generators and not needed for our pipeline.
    """
    # Remove <lora:...> tags (handles nested colons and decimal weights)
    cleaned = re.sub(r"<lora:[^>]+>", "", prompt)
    # Clean up extra whitespace and commas left behind
    cleaned = re.sub(r"\s*,\s*,\s*", ", ", cleaned)  # multiple commas
    cleaned = re.sub(r"\s{2,}", " ", cleaned)  # multiple spaces
    cleaned = cleaned.strip(" ,")  # leading/trailing spaces and commas
    return cleaned


def extract_model_id(url: str) -> int:
    """Extract model ID from CivitAI URL."""
    match = re.search(r"/models/(\d+)", url)
    if not match:
        raise ValueError(f"Could not extract model ID from URL: {url}")
    return int(match.group(1))


def extract_version_id_from_url(url: str) -> int | None:
    """Extract modelVersionId from URL query params if present."""
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    version_ids = params.get("modelVersionId", [])
    return int(version_ids[0]) if version_ids else None


def get_model_data(model_id: int, api_key: str) -> dict:
    """Fetch model data from CivitAI API."""
    response = requests.get(
        f"https://civitai.com/api/v1/models/{model_id}",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def get_version_images(
    version_id: int, api_key: str, username: str | None = None, expected_hashes: set | None = None
) -> list[dict]:
    """Fetch images with full metadata for a model version.

    Args:
        version_id: The model version ID
        api_key: CivitAI API key
        username: Filter to only this creator's images (recommended)
        expected_hashes: If provided, only return images matching these hashes
    """
    images = []
    cursor = None

    while True:
        params = {"modelVersionId": version_id, "limit": 100}
        if username:
            params["username"] = username
        if cursor:
            params["cursor"] = cursor

        response = requests.get(
            "https://civitai.com/api/v1/images",
            params=params,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        for item in data.get("items", []):
            # If we have expected hashes, only include matching images
            if expected_hashes is None or item.get("hash") in expected_hashes:
                images.append(item)

        # Check for more pages
        cursor = data.get("metadata", {}).get("nextCursor")
        if not cursor:
            break

    return images


def find_flux_version(model_data: dict, preferred_version_id: int | None = None) -> dict | None:
    """Find the Flux version from model versions.

    If preferred_version_id is provided and exists, use that.
    Otherwise, search for a version with 'flux' in baseModel or name.
    """
    versions = model_data.get("modelVersions", [])

    # If specific version requested, find and verify it
    if preferred_version_id:
        for v in versions:
            if v["id"] == preferred_version_id:
                return v

    # Otherwise search for Flux version
    for v in versions:
        base_model = v.get("baseModel", "").lower()
        name = v.get("name", "").lower()
        if "flux" in base_model or "flux" in name:
            return v

    return None


def download_lora(version_id: int, api_key: str, output_dir: Path) -> str:
    """Download the LoRA file and return the filename."""
    output_dir.mkdir(parents=True, exist_ok=True)

    response = requests.get(
        f"https://civitai.com/api/download/models/{version_id}?token={api_key}",
        stream=True,
        timeout=60,
    )
    response.raise_for_status()

    # Extract filename from Content-Disposition header
    content_disp = response.headers.get("content-disposition", "")
    if 'filename="' in content_disp:
        filename = content_disp.split('filename="')[1].rstrip('"')
    else:
        filename = f"lora_{version_id}.safetensors"

    filepath = output_dir / filename

    print(f"       Downloading to {filepath}...")
    with open(filepath, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return filename


def extract_prompts(model_data: dict, api_key: str) -> list[dict]:
    """Extract prompt data from model's example images (creator uploads only)."""
    all_versions = []
    creator_username = model_data.get("creator", {}).get("username")

    for version in model_data.get("modelVersions", []):
        version_id = version["id"]
        version_data = {
            "version_id": version_id,
            "version_name": version.get("name", ""),
            "base_model": version.get("baseModel", ""),
            "trained_words": version.get("trainedWords", []),
            "prompts": [],
        }

        # Get hashes of example images from model data (creator's uploads)
        example_hashes = {img.get("hash") for img in version.get("images", []) if img.get("hash")}

        # Fetch full metadata for creator's example images only
        images = get_version_images(
            version_id, api_key, username=creator_username, expected_hashes=example_hashes
        )

        for image in images:
            meta = image.get("meta")
            if meta and "prompt" in meta:
                prompt_data = {
                    "prompt": sanitize_prompt(meta["prompt"]),
                    "negative_prompt": meta.get("negativePrompt", ""),
                    "seed": meta.get("seed"),
                    "steps": meta.get("steps"),
                    "cfg_scale": meta.get("cfgScale"),
                    "sampler": meta.get("sampler"),
                    "image_url": image.get("url", ""),
                }
                version_data["prompts"].append(prompt_data)

        all_versions.append(version_data)

    return all_versions


def scrape_lora(url: str, api_key: str, data_dir: Path, download: bool = True) -> dict:
    """
    Main scraping function.

    Args:
        url: CivitAI model URL
        api_key: CivitAI API key
        data_dir: Base data directory
        download: Whether to download the LoRA file (set False for testing)

    Returns:
        Output metadata dict
    """
    loras_dir = data_dir / "loras"
    prompts_dir = data_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Parse URL
    print(f"\n[1/6] Parsing URL...")
    model_id = extract_model_id(url)
    query_version_id = extract_version_id_from_url(url)
    print(f"       Model ID: {model_id}")
    if query_version_id:
        print(f"       Version ID from URL: {query_version_id}")

    # Step 2: Get model data
    print(f"\n[2/6] Fetching model data from CivitAI API...")
    model_data = get_model_data(model_id, api_key)
    print(f"       Model name: {model_data.get('name', 'Unknown')}")
    print(f"       Found {len(model_data.get('modelVersions', []))} version(s)")

    # Step 3: Find Flux version
    print(f"\n[3/6] Identifying Flux version...")
    flux_version = find_flux_version(model_data, query_version_id)

    if not flux_version:
        raise ValueError("No Flux version found in this model")

    flux_version_id = flux_version["id"]
    print(f"       Flux version: {flux_version.get('name', 'Unknown')} (ID: {flux_version_id})")
    print(f"       Base model: {flux_version.get('baseModel', 'Unknown')}")
    print(f"       Trained words: {flux_version.get('trainedWords', [])}")

    # Step 4: Download LoRA
    filename = None
    if download:
        print(f"\n[4/6] Downloading Flux LoRA...")
        filename = download_lora(flux_version_id, api_key, loras_dir)
        print(f"       Saved: {loras_dir / filename}")
    else:
        print(f"\n[4/6] Skipping download (dry run)")

    # Step 5: Extract prompts
    print(f"\n[5/6] Extracting prompts from all versions...")
    all_versions = extract_prompts(model_data, api_key)
    total_prompts = sum(len(v["prompts"]) for v in all_versions)
    print(f"       Extracted {total_prompts} prompts across {len(all_versions)} version(s)")

    # Step 6: Save JSON
    print(f"\n[6/6] Saving prompt data...")
    output = {
        "model_id": model_id,
        "model_name": model_data.get("name", ""),
        "model_url": url,
        "flux_version_id": flux_version_id,
        "flux_version_name": flux_version.get("name", ""),
        "flux_base_model": flux_version.get("baseModel", ""),
        "trained_words": flux_version.get("trainedWords", []),
        "flux_file": filename,
        "all_versions": all_versions,
    }

    json_path = prompts_dir / f"{model_id}_prompts.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"       Saved: {json_path}")

    print(f"\n{'='*50}")
    print(f"Scraping complete!")
    print(f"{'='*50}")
    if filename:
        print(f"  LoRA: {loras_dir / filename}")
    print(f"  Prompts: {json_path}")
    print(f"  Total prompts: {total_prompts}")
    print()

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Scrape Flux LoRAs and prompts from CivitAI"
    )
    parser.add_argument("url", type=str, help="CivitAI model URL")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Base data directory (default: ./data)",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip downloading the LoRA file (for testing)",
    )
    parser.add_argument(
        "--expand-to",
        type=int,
        default=50,
        help="Expand prompts to this target count using Ollama (default: 50, set to 0 to skip)",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default="llama3.2",
        help="Ollama model for prompt expansion (default: llama3.2)",
    )

    args = parser.parse_args()

    # Load .env file
    load_dotenv()

    # Get API key from environment
    api_key = os.environ.get("CIVITAI_API")
    if not api_key:
        raise ValueError(
            "CIVITAI_API environment variable not set. "
            "Add it to .env or get your API key from https://civitai.com/user/account"
        )

    print("=" * 50)
    print("CivitAI LoRA Scraper")
    print("=" * 50)

    result = scrape_lora(
        url=args.url,
        api_key=api_key,
        data_dir=Path(args.data_dir),
        download=not args.no_download,
    )

    # Expand prompts if requested
    if args.expand_to > 0:
        from src.stage1_scrape.expand_prompts import expand_prompts

        prompts_path = Path(args.data_dir) / "prompts" / f"{result['model_id']}_prompts.json"
        expanded_data = expand_prompts(
            input_path=prompts_path,
            target_count=args.expand_to,
            model=args.ollama_model,
        )

        # Save expanded data back
        with open(prompts_path, "w", encoding="utf-8") as f:
            json.dump(expanded_data, f, indent=2)

        total_prompts = len(expanded_data.get("generated_prompts", []))
        for v in expanded_data.get("all_versions", []):
            total_prompts += len(v.get("prompts", []))

        print(f"\n{'=' * 50}")
        print("Pipeline complete!")
        print(f"{'=' * 50}")
        print(f"  Total prompts: {total_prompts}")


if __name__ == "__main__":
    main()
