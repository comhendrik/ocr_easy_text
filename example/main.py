from ollama_client import run_ollama_request
from detect import detect_text_in_image
from generate_text import text_to_speech


def main() -> None:
    text = detect_text_in_image("image.png")
    prompt = "Simplify the follwing text" + text
    result = run_ollama_request(prompt)
    text_to_speech(result)


if __name__ == "__main__":
    main()
