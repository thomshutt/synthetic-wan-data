"""
Prompt Expander - Uses Ollama to generate more prompts from examples.

Takes scraped prompts and uses few-shot conditioning to expand the dataset
to a target number of prompts.

Usage:
    python -m src.stage1_scrape.expand_prompts data/prompts/317208_prompts.json --target 50
"""

import argparse
import json
import random
from pathlib import Path

import ollama


def build_system_prompt(trained_words: list[str]) -> str:
    """Build the system prompt for the LLM."""
    trigger_words = ", ".join(trained_words) if trained_words else "the style trigger word"
    return f"""You are a creative prompt writer for image generation models.
Your task is to generate new prompts in the same style as the examples provided.

Important rules:
- Include the trigger word(s): {trigger_words}
- Match the style, structure, and complexity of the examples
- Be creative with subjects, settings, and modifiers
- Do NOT include any <lora:...> tags
- Output ONLY the prompt text, nothing else
- Each prompt should be unique and different from the examples"""


def build_few_shot_prompt(examples: list[str], num_examples: int = 5) -> str:
    """Build a few-shot prompt with random examples."""
    selected = random.sample(examples, min(num_examples, len(examples)))

    prompt = "Here are example prompts in the target style:\n\n"
    for i, ex in enumerate(selected, 1):
        prompt += f"Example {i}:\n{ex}\n\n"

    prompt += "Now generate a new, unique prompt in the same style. Output only the prompt text:"
    return prompt


def generate_prompt(
    client: ollama.Client,
    model: str,
    system_prompt: str,
    examples: list[str],
    num_examples: int = 5,
) -> str:
    """Generate a single new prompt using Ollama."""
    user_prompt = build_few_shot_prompt(examples, num_examples)

    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response["message"]["content"].strip()


def expand_prompts(
    input_path: Path,
    target_count: int = 50,
    model: str = "llama3.2",
    ollama_host: str | None = None,
    num_examples: int = 5,
) -> dict:
    """
    Expand prompts using Ollama few-shot generation.

    Args:
        input_path: Path to the scraped prompts JSON
        target_count: Target number of total prompts
        model: Ollama model to use
        ollama_host: Ollama server URL (default: localhost)
        num_examples: Number of examples to show in each few-shot prompt

    Returns:
        Updated prompt data dict
    """
    # Load existing prompts
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    # Collect all existing prompts (prefer Flux version)
    existing_prompts = []
    trained_words = data.get("trained_words", [])

    for version in data.get("all_versions", []):
        for p in version.get("prompts", []):
            if p.get("prompt"):
                existing_prompts.append(p["prompt"])

    if not existing_prompts:
        raise ValueError("No existing prompts found to use as examples")

    current_count = len(existing_prompts)
    prompts_needed = max(0, target_count - current_count)

    print(f"\n[Prompt Expansion]")
    print(f"  Existing prompts: {current_count}")
    print(f"  Target count: {target_count}")
    print(f"  Prompts to generate: {prompts_needed}")

    if prompts_needed == 0:
        print("  Already at or above target count, skipping expansion")
        return data

    # Initialize Ollama client
    client = ollama.Client(host=ollama_host) if ollama_host else ollama.Client()
    system_prompt = build_system_prompt(trained_words)

    print(f"  Using model: {model}")
    print(f"  Trigger words: {trained_words}")
    print()

    # Generate new prompts
    generated_prompts = []
    for i in range(prompts_needed):
        print(f"  Generating prompt {i + 1}/{prompts_needed}...", end=" ", flush=True)
        try:
            new_prompt = generate_prompt(
                client, model, system_prompt, existing_prompts, num_examples
            )
            generated_prompts.append(new_prompt)
            print("done")
        except Exception as e:
            print(f"failed: {e}")

    # Add generated prompts to the data structure
    # Store them in a new "generated_prompts" field
    data["generated_prompts"] = [
        {"prompt": p, "source": "ollama", "model": model}
        for p in generated_prompts
    ]

    print(f"\n  Successfully generated {len(generated_prompts)} new prompts")

    return data


def main():
    parser = argparse.ArgumentParser(
        description="Expand prompts using Ollama few-shot generation"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to scraped prompts JSON file",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=50,
        help="Target number of total prompts (default: 50)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.2",
        help="Ollama model to use (default: llama3.2)",
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default=None,
        help="Ollama server URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=5,
        help="Number of examples to show in few-shot prompt (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: overwrites input file)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print("=" * 50)
    print("Prompt Expander (Ollama)")
    print("=" * 50)

    # Expand prompts
    data = expand_prompts(
        input_path=input_path,
        target_count=args.target,
        model=args.model,
        ollama_host=args.ollama_host,
        num_examples=args.num_examples,
    )

    # Save output
    output_path = Path(args.output) if args.output else input_path
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"\n  Saved to: {output_path}")

    # Summary
    total_prompts = len(data.get("generated_prompts", []))
    for v in data.get("all_versions", []):
        total_prompts += len(v.get("prompts", []))

    print(f"\n{'=' * 50}")
    print(f"Expansion complete!")
    print(f"{'=' * 50}")
    print(f"  Total prompts: {total_prompts}")
    print()


if __name__ == "__main__":
    main()
