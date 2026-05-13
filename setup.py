from huggingface_hub import snapshot_download
import os, requests

print("Downloading LLM weights...")
snapshot_download("Qwen/Qwen2.5-0.5B-Instruct")

print("Downloading embedding model...")
snapshot_download("sentence-transformers/all-MiniLM-L6-v2")

print("Downloading PDF...")
pdf_path = "world-history-text.pdf"
if not os.path.exists(pdf_path):
    url = "https://assets.openstax.org/oscms-prodcms/media/documents/World_History_Volume_2-WEB.pdf"
    with open(pdf_path, "wb") as f:
        f.write(requests.get(url).content)

print("All set! You can now run the notebook offline.")