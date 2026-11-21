from engine import Passage, chunk_passages, extract, build_prompt

def test_text_extraction():
    p = extract('notes.txt', b'Hello RAG')
    assert p[0].text == 'Hello RAG'

def test_chunking_preserves_source():
    parts = chunk_passages([Passage('hello world ' * 120, 'notes.txt', 3)], chunk_size=100, overlap=20)
    assert len(parts) > 1
    assert all(p.source == 'notes.txt' and p.page == 3 for p in parts)

def test_prompt_contains_citations():
    prompt = build_prompt('What is RAG?', [Passage('RAG retrieves context', 'a.pdf', 2)])
    assert '[SOURCE 1]' in prompt and 'a.pdf' in prompt
