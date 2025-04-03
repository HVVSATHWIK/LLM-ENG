import faiss
import json
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import gradio as gr

# Load FAISS index
faiss_index = faiss.read_index("faiss_index.index")

# Load documents
with open("documents.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

# Load sentence-transformers model (for encoding queries)
device = "cpu"  # Assuming no GPU locally; change to "cuda" if you have a GPU
embedder = SentenceTransformer('all-mpnet-base-v2', device=device)

# Load FLAN-T5 for answering
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
llama_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

def retrieve_documents(query, top_k=10):
    query_embedding = embedder.encode([query], show_progress_bar=False)  # Disable progress bar for cleaner chat
    distances, indices = faiss_index.search(query_embedding, top_k * 2)
    results = [documents[i] for i in indices[0]]
    query_words = set(query.lower().split())
    critical_terms = {"tall", "height"} if "tall" in query.lower() or "height" in query.lower() else query_words
    ranked_results = sorted(
        results,
        key=lambda x: (
            sum(10 if word in critical_terms and word in x.lower().split() else 1 
                for word in query_words if word in x.lower()) + 
            0.1 / (distances[0][results.index(x)] + 1e-6)
        ),
        reverse=True
    )
    return ranked_results[:top_k]

def answer_query(query, top_k=10):
    context_chunks = retrieve_documents(query, top_k)
    if not context_chunks:
        return "No relevant information found in the documents."
    context = "\n".join(context_chunks)
    tokenized_context = tokenizer.encode(context, truncation=True, max_length=512)
    truncated_context = tokenizer.decode(tokenized_context, skip_special_tokens=True)
    prompt = f"Using only the context below, provide a detailed and accurate answer to the question, including all relevant points from the provided text. Do not add information not present in the context.\nContext: {truncated_context}\nQuestion: {query}\nAnswer:"
    response = llama_pipeline(
        prompt,
        max_length=200,
        min_length=40,
        do_sample=False,
        num_beams=5,
        early_stopping=True
    )
    return response[0]['generated_text'].strip()

# Chatbot function for Gradio
def chat_with_tree_qa(message, history):
    # Get the answer from the QA system
    response = answer_query(message)
    # Append the user message and response to history
    history.append((message, response))
    return history, history  # Return updated history for display and state

# Create the Gradio chat interface
with gr.Blocks(title="Tree Species FAQ Chatbot") as interface:
    gr.Markdown("# Tree Species FAQ Chatbot")
    gr.Markdown("Ask questions about tree species like Neem, Poplar, Teak, and more based on the FAQ documents.")
    
    # Chatbot component
    chatbot = gr.Chatbot(label="Conversation", type="messages")
    
    # Textbox for user input
    msg = gr.Textbox(placeholder="Type your question here (e.g., 'How tall can Poplar grow?')", label="Your Question")
    
    # State to maintain chat history
    state = gr.State(value=[])
    
    # Buttons
    submit_btn = gr.Button("Submit")
    clear_btn = gr.Button("Clear Chat")
    
    # Example queries
    examples = gr.Examples(
        examples=[
            "What is Neem known for according to the FAQs?",
            "How tall can Poplar grow?",
            "What pests affect Sandalwood?",
            "What are the main plantation practices for Teak?"
        ],
        inputs=msg
    )
    
    # Event handlers
    submit_btn.click(
        fn=chat_with_tree_qa,
        inputs=[msg, state],
        outputs=[chatbot, state],
        _js="() => {let input = document.querySelector('input'); input.value = ''; return [input.value, input.value];}"
    )
    clear_btn.click(
        fn=lambda: ([], []),
        inputs=None,
        outputs=[chatbot, state]
    )

# Launch the interface
interface.launch()