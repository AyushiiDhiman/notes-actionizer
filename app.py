import os
import spacy
import gradio as gr
import pandas as pd

nlp = spacy.load("en_core_web_sm")

ACTION_VERBS = {
    "submit", "send", "revise", "check", "review", "finish", "complete",
    "call", "email", "prepare", "update", "fix", "schedule", "book",
    "confirm", "share", "discuss", "follow", "remind", "pay", "buy",
    "meet", "write", "read", "test", "deploy", "upload", "download",
    "clean", "organize", "plan", "attend", "cancel", "reschedule"
}

def tokenize_text(doc):
    return [t.text for t in doc if not t.is_space]

def pos_tag_text(doc):
    return [(t.text, t.pos_, t.tag_) for t in doc if not t.is_space]

def lemmatize_text(doc):
    return [(t.text, t.lemma_) for t in doc if not t.is_space and t.text.lower() != t.lemma_.lower()]

def remove_stopwords(doc):
    kept = [t.text for t in doc if not t.is_stop and not t.is_punct and not t.is_space]
    removed = [t.text for t in doc if t.is_stop]
    return kept, removed

def extract_entities(doc):
    return [(ent.text, ent.label_) for ent in doc.ents]

def find_action_items(doc):
    items = []
    for sent in doc.sents:
        verbs = [t for t in sent if t.pos_ == "VERB"]
        if not verbs:
            continue
        chosen = None
        for v in verbs:
            if v.lemma_.lower() in ACTION_VERBS:
                chosen = v
                break
        if chosen is None:
            chosen = verbs[0]

        tail_tokens = [c.text for c in chosen.subtree if c.i > chosen.i]
        phrase = chosen.lemma_.capitalize()
        if tail_tokens:
            phrase += " " + " ".join(tail_tokens[:8])

        tags = [f"{e.text} ({e.label_})" for e in sent.ents]
        line = phrase.strip()
        if tags:
            line += "  —  " + ", ".join(tags)
        items.append(line)
    return items

def run_pipeline(raw_text):
    if not raw_text or not raw_text.strip():
        empty_df = pd.DataFrame()
        return "Paste some messy notes first.", empty_df, empty_df, "", empty_df, "No action items yet."

    doc = nlp(raw_text)

    tokens = tokenize_text(doc)
    token_display = ", ".join(tokens)

    pos_rows = pos_tag_text(doc)
    pos_df = pd.DataFrame(pos_rows, columns=["Word", "POS", "Detailed Tag"])

    lemma_rows = lemmatize_text(doc)
    lemma_df = pd.DataFrame(lemma_rows, columns=["Original", "Lemma"])
    if lemma_df.empty:
        lemma_df = pd.DataFrame(columns=["Original", "Lemma"])

    kept, removed = remove_stopwords(doc)
    cleaned_display = ", ".join(kept) if kept else "Nothing left after cleaning."

    ent_rows = extract_entities(doc)
    ent_df = pd.DataFrame(ent_rows, columns=["Entity", "Type"])
    if ent_df.empty:
        ent_df = pd.DataFrame(columns=["Entity", "Type"])

    actions = find_action_items(doc)
    if actions:
        action_display = "\n".join(f"☐ {a}" for a in actions)
    else:
        action_display = "No clear action items found in this text."

    return token_display, pos_df, lemma_df, cleaned_display, ent_df, action_display


sample_notes = (
    "Rahul needs to submit the project report by Friday. "
    "Priya said she will call the vendor tomorrow morning to confirm the delivery. "
    "We also have to revise the budget sheet before the meeting with Mr. Sharma on Monday. "
    "Someone should book the conference room in Delhi office for next week's review."
)

with gr.Blocks(title="Notes Actionizer") as demo:
    gr.Markdown("# 📝 Notes Actionizer")
    gr.Markdown(
        "Paste messy class or meeting notes below. The app runs them through a full "
        "NLP pipeline — tokenization, POS tagging, lemmatization, stopword removal and "
        "named entity recognition — and turns the mess into a clean to-do list."
    )

    with gr.Row():
        input_box = gr.Textbox(
            label="Raw notes",
            placeholder="Paste your messy notes here...",
            lines=8,
            value=sample_notes,
        )

    run_btn = gr.Button("Run NLP Pipeline", variant="primary")

    with gr.Tab("1. Tokenization"):
        token_out = gr.Textbox(label="Tokens", lines=6)

    with gr.Tab("2. POS Tagging"):
        pos_out = gr.Dataframe(label="Part-of-Speech Tags", wrap=True)

    with gr.Tab("3. Lemmatization"):
        lemma_out = gr.Dataframe(label="Words Changed by Lemmatization", wrap=True)

    with gr.Tab("4. Stopword Removal"):
        clean_out = gr.Textbox(label="Text After Removing Stopwords & Punctuation", lines=6)

    with gr.Tab("5. Named Entity Recognition"):
        ent_out = gr.Dataframe(label="Detected Entities", wrap=True)

    with gr.Tab("✅ Final To-Do List"):
        action_out = gr.Textbox(label="Action Items Extracted From Your Notes", lines=8)

    run_btn.click(
        fn=run_pipeline,
        inputs=input_box,
        outputs=[token_out, pos_out, lemma_out, clean_out, ent_out, action_out],
    )

    input_box.change(
        fn=run_pipeline,
        inputs=input_box,
        outputs=[token_out, pos_out, lemma_out, clean_out, ent_out, action_out],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
