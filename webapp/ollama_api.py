import requests
import json
import base64
from io import BytesIO


def chat_via_backend(
    prompt: str,
    backend_base: str = "http://localhost:7860",
    model: str | None = None,
    image_path: str | None = None,
    timeout: int = 60,
) -> str:
    """
    Minimal call to the FastAPI backend chatbot endpoint.
    - Sends a POST to `<backend_base>/api/chat` with JSON.
    - Returns the full text response (collected if backend streams).

    Example:
        chat_via_backend("Hello", backend_base="http://127.0.0.1:7860")
    """
    url = f"{backend_base.rstrip('/')}/api/chat"
    payload: dict = {"prompt": prompt}
    if model:
        payload["model"] = model
    if image_path:
        payload["image_path"] = image_path

    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    # Backend streams text/plain; requests collects it into .text here.
    return resp.text

def ollama_stream_inference(
    prompt: str,
    model: str = "tonai_chat",
    url: str = "http://127.0.0.1:7860/api/generate",
    image_path: str = ""
):
    """
    Send a streaming request to Ollama using the given prompt and model,
    and print out the response text in real time.
    """
    # Configure the payload according to Ollama's API.
    # You can add parameters like 'temperature' or 'top_p' if your server supports them.
    payload = {
        "model": model,
        "prompt": prompt
    }
    if image_path is not None and image_path != "":
        payload["images"] = [encode_image_to_base64(image_path)]

    # We’ll store the entire response in this list as we stream chunks
    all_chunks = []

    # Use 'stream=True' for streaming responses
    with requests.post(url, json=payload, stream=True) as resp:
        # Raise an error if the request is not 200 OK
        resp.raise_for_status()

        # Iterate over each line that Ollama sends back
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                # If there's a blank line (keep-alive), just skip
                continue

            try:
                # Each line is a separate JSON object
                data = json.loads(line)
            except json.JSONDecodeError:
                # If you get partial or malformed data, handle/log it
                continue

            # Extract the chunk of text
            text_chunk = data.get("response", "")
            # Print directly to terminal (no extra newline, flush so it appears in real time)
            print(text_chunk, end="", flush=True)

            # Append chunk to our list so we can reconstruct later if we want
            all_chunks.append(text_chunk)

            # If "done" is True, the server indicates it's done streaming
            if data.get("done", False):
                break

    # Combine all chunks if you want the comprehensive string
    full_response = "".join(all_chunks)
    # print("\n\n---\nComplete response:\n", full_response)
    return full_response

def encode_image_to_base64(image_path: str) -> str:
    """
    Convert an image file to a base64 encoded string.

    Args:
        image_path (str): The file path of the image to encode.

    Returns:
        str: The base64 encoded string representation of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# Example usage
if __name__ == "__main__":
    # Direct call to local Ollama at port 7860 (text-only)
    print(ollama_stream_inference("Why is the sky blue?"))
