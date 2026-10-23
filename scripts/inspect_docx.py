import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).parent.parent
docx_path = ROOT / "docs" / "Decide_Hackathon_Implementation_Playbook.docx"

if not docx_path.exists():
    print("docx not found")
    exit()

with zipfile.ZipFile(docx_path) as z:
    xml_content = z.read("word/document.xml")
    tree = ET.fromstring(xml_content)
    texts = []
    for node in tree.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"):
        if node.text:
            texts.append(node.text)
    full_text = " ".join(texts)
    print("Total words in docx:", len(full_text.split()))

    terms = ["pro", "limit", "100", "policy", "threshold", "refund", "credit", "approval", "reject", "$100", "tier"]
    for term in terms:
        count = full_text.lower().count(term)
        print(f"Count of '{term}': {count}")
        idx = 0
        shown = 0
        while shown < 3:
            idx = full_text.lower().find(term, idx)
            if idx == -1:
                break
            snippet = full_text[max(0, idx-60):min(len(full_text), idx+100)].replace("\n", " ")
            print(f"    Snippet: ...{snippet}...")
            idx += len(term) + 150
            shown += 1
