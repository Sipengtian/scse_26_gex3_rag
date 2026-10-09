from pathlib import Path
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rag_tool import format_context, has_phone_number, retrieve_documents


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "SCSE_26_GEX3"
    / "models"
    / "qwen3-0.6b"
)


def asks_for_phone_number(question):
    lowered = question.lower()
    return (
        "phone" in lowered
        and "number" in lowered
        and "service desk" in lowered
    )


def load_model(model_path=DEFAULT_MODEL_PATH):
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype="auto",
        device_map="auto",
        local_files_only=True
    )

    return tokenizer, model


def build_prompt(question, context):
    return (
        "You are a university IT support assistant.\n"
        "Answer the user's question using only the retrieved policy context.\n"
        "If the exact requested fact is not present in the context, say that "
        "the policy context does not provide it. Do not invent details.\n\n"
        "Retrieved policy context:\n"
        f"{context}\n\n"
        f"User: {question}\n\n"
        "Assistant:"
    )


def clean_answer(text):
    if "Assistant:" in text:
        text = text.rsplit("Assistant:", 1)[-1]

    text = text.strip()
    text = re.split(r"\n\s*User:", text, maxsplit=1)[0].strip()
    return text


def answer_question(question, model_path=DEFAULT_MODEL_PATH):
    documents = retrieve_documents(question, k=6)
    context = format_context(documents)

    if asks_for_phone_number(question) and not has_phone_number(context):
        return (
            "I do not know the exact phone number for the University IT "
            "Service Desk because the retrieved policy context does not "
            "provide a phone number."
        )

    tokenizer, model = load_model(model_path)
    prompt = build_prompt(question, context)

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    return clean_answer(
        tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )
    )
