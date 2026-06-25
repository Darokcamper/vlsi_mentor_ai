import os
from pathlib import Path
from openai import OpenAI
import base64

api_key = os.getenv("DEEPSEEK_API_KEY", "sk-placeholder")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

IMG_DIR = Path("../../knowledge/images/1. Level 1 session 1")

markdown = ""

for image_file in sorted(IMG_DIR.glob("*.png")):

    print("Processing", image_file.name)

    with open(image_file, "rb") as f:
        image_base64 = base64.b64encode(
            f.read()
        ).decode()

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """
Convert this handwritten DFT note into clean markdown.

Preserve:
- headings
- commands
- examples
- DFT terms

Return markdown only.
"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url":
                            f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]
    )

    markdown += (
        f"\n\n# {image_file.name}\n\n"
    )

    markdown += (
        response.choices[0]
        .message
        .content
    )

with open(
    "../../knowledge/markdown/1. Level 1 session 1.md",
    "w",
    encoding="utf-8"
) as f:

    f.write(markdown)

print("Done")