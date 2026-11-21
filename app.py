import os
import time
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from engine import KnowledgeBase, answer, generate, build_prompt

load_dotenv()
st.set_page_config(page_title='DocuMind AI | Hybrid RAG Studio', page_icon='🧠', layout='wide')
st.markdown('''<style>
.stApp {background: #0b1220; color: #e2e8f0;} .block-container {padding-top: 2rem; max-width: 1350px;}
[data-testid="stMetric"] {background:#162238;border:1px solid #263750;padding:15px;border-radius:14px;}
.hero {background:linear-gradient(115deg,#102e43,#18203b);padding:30px;border-radius:18px;border:1px solid #28516a;margin-bottom:20px;}
.small {color:#9bb0c8;font-size:0.95rem;}
</style>''', unsafe_allow_html=True)

@st.cache_resource
def get_kb():
    return KnowledgeBase()

st.markdown('<div class="hero"><h1>🧠 DocuMind AI</h1><div class="small">Intelligent document understanding • Hybrid retrieval • Grounded LLM answers</div></div>', unsafe_allow_html=True)
with st.sidebar:
    st.header('⚙️ Model settings')
    provider = st.selectbox('LLM provider', ['Ollama', 'Gemini'])
    default_model = 'llama3.2' if provider == 'Ollama' else 'gemini-2.5-flash'
    model = st.text_input('Model name', value=default_model)
    key = st.text_input('Gemini API key', value=os.getenv('GEMINI_API_KEY', ''), type='password') if provider == 'Gemini' else ''
    hybrid = st.toggle('Hybrid retrieval (vector + BM25)', value=True)
    top_k = st.slider('Number of retrieved chunks', 2, 10, 5)
    st.caption('Documents are indexed locally. Gemini requests send the selected excerpts to Google.')

try:
    kb = get_kb()
except Exception as exc:
    st.error(f'Could not initialize embedding model / database: {exc}')
    st.stop()

if 'chat' not in st.session_state:
    st.session_state.chat = []
if 'logs' not in st.session_state:
    st.session_state.logs = []

tab_chat, tab_docs, tab_compare, tab_analytics = st.tabs(['💬 AI Assistant', '📚 Knowledge Base', '⚖️ RAG vs LLM', '📊 Analytics'])
with tab_docs:
    st.subheader('Upload and index documents')
    uploads = st.file_uploader('PDF, DOCX, TXT, or Markdown', type=['pdf','docx','txt','md'], accept_multiple_files=True)
    if st.button('⚡ Index selected documents', type='primary', disabled=not uploads):
        for upload in uploads:
            try:
                with st.spinner(f'Indexing {upload.name}...'):
                    count = kb.add(upload.name, upload.getvalue())
                st.success(f'{upload.name}: {count} chunks indexed') if count else st.warning(f'{upload.name}: no extractable text found')
            except Exception as exc:
                st.error(f'{upload.name}: {exc}')
    docs = kb.list_documents()
    if docs:
        st.dataframe(pd.DataFrame(docs), use_container_width=True, hide_index=True)
    else:
        st.info('No documents indexed yet. Upload a file to get started.')
    with st.expander('Danger zone'):
        if st.button('Delete all indexed documents'):
            kb.clear()
            st.session_state.chat = []
            st.success('Knowledge base cleared')
            st.rerun()

with tab_chat:
    st.caption('Answers cite the retrieved excerpts. Verify important claims against the original documents.')
    for msg in st.session_state.chat:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])
            if msg.get('sources'):
                with st.expander('📎 Retrieved evidence'):
                    for p in msg['sources']:
                        st.markdown(f'**{p.source} · Page {p.page}** (retrieval score {p.score:.4f})')
                        st.caption(p.text[:1200])
    question = st.chat_input('Ask anything about your uploaded documents…')
    if question:
        previous = st.session_state.chat.copy()
        st.session_state.chat.append({'role':'user','content':question})
        with st.chat_message('user'):
            st.markdown(question)
        with st.chat_message('assistant'):
            try:
                with st.spinner('Retrieving evidence and generating an answer...'):
                    response, sources, latency = answer(kb, question, provider, model, key, hybrid, top_k, previous)
                st.markdown(response)
                with st.expander('📎 Retrieved evidence'):
                    for p in sources:
                        st.markdown(f'**{p.source} · Page {p.page}** — {p.score:.4f}')
                        st.caption(p.text[:1200])
                st.session_state.chat.append({'role':'assistant','content':response,'sources':sources})
                st.session_state.logs.append({'question':question,'latency_s':round(latency,2),'sources':len(sources),'mode':'RAG'})
            except Exception as exc:
                st.error(f'Generation failed: {exc}')
    if st.button('Clear chat history'):
        st.session_state.chat = []
        st.rerun()

with tab_compare:
    st.subheader('Compare standalone LLM with grounded RAG')
    q = st.text_area('Comparison question', placeholder='What does the document say about...?')
    if st.button('Run comparison', disabled=not q.strip()):
        left, right = st.columns(2)
        with left:
            st.markdown('#### 🧠 Standalone LLM')
            try:
                with st.spinner('Generating...'):
                    t0 = time.perf_counter()
                    raw = generate('Answer the following question clearly.\nQuestion: ' + q, provider, model, key)
                    raw_time = time.perf_counter() - t0
                st.write(raw)
                st.caption(f'{raw_time:.2f}s')
            except Exception as exc:
                st.error(str(exc))
        with right:
            st.markdown('#### 📚 RAG-enhanced LLM')
            try:
                with st.spinner('Retrieving and generating...'):
                    grounded, sources, elapsed = answer(kb, q, provider, model, key, hybrid, top_k)
                st.write(grounded)
                st.caption(f'{elapsed:.2f}s · {len(sources)} retrieved chunks')
                for p in sources:
                    st.caption(f'{p.source}, page {p.page}')
                st.session_state.logs.append({'question':q,'latency_s':round(elapsed,2),'sources':len(sources),'mode':'Comparison RAG'})
            except Exception as exc:
                st.error(str(exc))
    st.info('Side-by-side outputs are illustrative, not proof that either answer is correct. Use reference answers for rigorous evaluation.')

with tab_analytics:
    docs = kb.list_documents()
    a,b,c = st.columns(3)
    a.metric('Indexed documents', len(docs))
    b.metric('Indexed chunks', sum(d['chunks'] for d in docs))
    c.metric('Queries in this session', len(st.session_state.logs))
    if st.session_state.logs:
        df = pd.DataFrame(st.session_state.logs)
        st.subheader('Query performance')
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.line_chart(df['latency_s'])
        st.download_button('Export session metrics (CSV)', df.to_csv(index=False), 'documind_metrics.csv', 'text/csv')
    else:
        st.info('Ask questions to populate latency and retrieval analytics.')
    st.caption('Retrieval scores are reciprocal-rank fusion scores, not probabilities. This dashboard does not claim RAGAS metrics without labeled evaluation data.')
