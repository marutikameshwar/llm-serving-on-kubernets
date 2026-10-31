#!/usr/bin/env python3
"""
scripts/test-client.py
Interactive and streaming test client for vLLM OpenAI-compatible endpoint.
Uses standard Python library (zero third-party dependencies required).
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error


def stream_chat_completion(base_url: str, model: str, prompt: str):
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer None",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful, concise AI assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 256,
        "stream": True,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    print(f"\n[Connecting to {url} | Model: {model}]")
    print(f"[Prompt]: {prompt}\n")
    print("--- [Streamed Response] ---")

    start_time = time.time()
    first_token_time = None
    token_count = 0

    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            if first_token_time is None:
                                first_token_time = time.time()
                            sys.stdout.write(content)
                            sys.stdout.flush()
                            token_count += 1
                except json.JSONDecodeError:
                    continue

        total_time = time.time() - start_time
        ttft = (first_token_time - start_time) * 1000 if first_token_time else 0
        gen_time = (time.time() - first_token_time) if first_token_time else total_time
        tokens_per_sec = token_count / gen_time if gen_time > 0 else 0

        print("\n\n--- [Telemetry Metrics] ---")
        print(f"Time to First Token (TTFT): {ttft:.1f} ms")
        print(f"Total Generation Time:       {total_time:.2f} s")
        print(f"Tokens Generated:            {token_count}")
        print(f"Throughput:                  {tokens_per_sec:.2f} tok/s")
        print("---------------------------")

    except urllib.error.URLError as e:
        print(f"\n[ERROR]: Failed to connect to vLLM server at {url}: {e}")
        print("Ensure the vLLM pod is Running and port-forwarded (`kubectl port-forward svc/vllm-service 8000:8000 -n llm-serving`)")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="vLLM Streaming Test Client")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="vLLM OpenAI endpoint URL (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5-7b",
        help="Model identifier (default: qwen2.5-7b)",
    )
    parser.add_argument(
        "--prompt",
        default="Explain PagedAttention in three concise sentences.",
        help="Prompt to send to the model",
    )
    args = parser.parse_args()

    stream_chat_completion(args.base_url, args.model, args.prompt)


if __name__ == "__main__":
    main()
