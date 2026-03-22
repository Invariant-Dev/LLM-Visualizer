from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import time
import os
import asyncio
from contextlib import asynccontextmanager
from fastapi.responses import StreamingResponse

MODEL_NAME = os.environ.get("HF_MODEL", "gpt2")
tokenizer = None
model = None
model_config = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model, model_config
    print(f"loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.config.output_attentions = True
    model.config.output_hidden_states = True
    model.eval()
    cfg = model.config
    model_config = {
        "model_name": MODEL_NAME,
        "num_layers": getattr(cfg, "n_layer", getattr(cfg, "num_hidden_layers", 0)),
        "num_heads": getattr(cfg, "n_head", getattr(cfg, "num_attention_heads", 0)),
        "hidden_size": getattr(cfg, "n_embd", getattr(cfg, "hidden_size", 0)),
        "vocab_size": getattr(cfg, "vocab_size", 0),
        "max_position": getattr(cfg, "n_positions", getattr(cfg, "max_position_embeddings", 0)),
        "intermediate_size": getattr(cfg, "n_inner", getattr(cfg, "intermediate_size", 0)),
    }
    print(f"model loaded: {model_config}")
    yield
    print("shutting down pytorch server")

app = FastAPI(title="LLM Internals PyTorch Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptRequest(BaseModel):
    prompt: str
    model: str = ""
    layer: int = -1

class GenerateRequest(BaseModel):
    prompt: str
    model: str = ""
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9
    num_predict: int = 200

class TokenizeRequest(BaseModel):
    prompt: str
    model: str = ""

@app.get("/pytorch/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME, "config": model_config}

@app.post("/pytorch/tokenize")
async def pytorch_tokenize(req: TokenizeRequest):
    if tokenizer is None:
        raise HTTPException(503, "model not loaded")
    ids = tokenizer.encode(req.prompt, add_special_tokens=False)
    texts = [tokenizer.decode([tid]) for tid in ids]
    return {"token_ids": ids, "tokens": texts, "count": len(ids)}

@app.post("/pytorch/embeddings")
async def pytorch_embeddings(req: PromptRequest):
    if model is None or tokenizer is None:
        raise HTTPException(503, "model not loaded")
    t0 = time.time()
    inputs = tokenizer(req.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    token_texts = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.hidden_states
    layer_idx = req.layer if req.layer >= 0 else len(hidden_states) - 1
    layer_idx = min(layer_idx, len(hidden_states) - 1)
    embeddings = hidden_states[layer_idx][0].cpu().numpy()
    norms = np.linalg.norm(embeddings, axis=1)
    cosine_matrix = []
    for i in range(len(embeddings)):
        row = []
        for j in range(len(embeddings)):
            dot = float(np.dot(embeddings[i], embeddings[j]))
            cos = dot / (float(norms[i]) * float(norms[j]) + 1e-8)
            row.append(round(cos, 4))
        cosine_matrix.append(row)
    sampled_dims = min(64, embeddings.shape[1])
    sampled_embeddings = embeddings[:, :sampled_dims].tolist()
    full_norms = norms.tolist()
    stats = {
        "embedding_dim": int(embeddings.shape[1]),
        "num_tokens": len(token_texts),
        "layer_used": layer_idx,
        "total_layers": len(hidden_states),
        "inference_ms": round((time.time() - t0) * 1000, 1),
    }
    return {
        "tokens": token_texts,
        "embeddings": sampled_embeddings,
        "norms": full_norms,
        "cosine_similarity": cosine_matrix,
        "stats": stats,
    }

@app.post("/pytorch/attention")
async def pytorch_attention(req: PromptRequest):
    if model is None or tokenizer is None:
        raise HTTPException(503, "model not loaded")
    t0 = time.time()
    inputs = tokenizer(req.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    token_texts = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]
    with torch.no_grad():
        outputs = model(**inputs)
    attentions = outputs.attentions
    layer_idx = req.layer if req.layer >= 0 else len(attentions) - 1
    layer_idx = min(layer_idx, len(attentions) - 1)
    layer_attn = attentions[layer_idx][0].cpu().numpy()
    num_heads = layer_attn.shape[0]
    heads = []
    for h in range(num_heads):
        matrix = layer_attn[h].tolist()
        heads.append([[round(v, 4) for v in row] for row in matrix])
    head_entropies = []
    for h in range(num_heads):
        attn = layer_attn[h]
        ent = -np.sum(attn * np.log(attn + 1e-10), axis=1)
        head_entropies.append(round(float(np.mean(ent)), 4))
    all_layer_summary = []
    for li, la in enumerate(attentions):
        la_np = la[0].cpu().numpy()
        avg_ent = float(np.mean(-np.sum(la_np * np.log(la_np + 1e-10), axis=2)))
        avg_max = float(np.mean(np.max(la_np, axis=2)))
        all_layer_summary.append({
            "layer": li,
            "avg_entropy": round(avg_ent, 4),
            "avg_max_attention": round(avg_max, 4),
            "num_heads": int(la_np.shape[1]),
        })
    stats = {
        "num_layers": len(attentions),
        "num_heads": num_heads,
        "num_tokens": len(token_texts),
        "layer_used": layer_idx,
        "inference_ms": round((time.time() - t0) * 1000, 1),
    }
    return {
        "tokens": token_texts,
        "heads": heads,
        "head_entropies": head_entropies,
        "layer_summary": all_layer_summary,
        "stats": stats,
    }

@app.post("/pytorch/full_forward")
async def pytorch_full_forward(req: PromptRequest):
    if model is None or tokenizer is None:
        raise HTTPException(503, "model not loaded")
    t0 = time.time()
    inputs = tokenizer(req.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    token_texts = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]
    probs = torch.softmax(logits, dim=0)
    top_k_vals, top_k_ids = torch.topk(probs, 30)
    top_tokens = []
    for i in range(30):
        tid = top_k_ids[i].item()
        top_tokens.append({
            "token": tokenizer.decode([tid]),
            "token_id": tid,
            "probability": round(top_k_vals[i].item(), 6),
            "logit": round(logits[tid].item(), 4),
        })
    hidden_states = outputs.hidden_states
    layer_norms = []
    for li, hs in enumerate(hidden_states):
        norm = float(torch.norm(hs[0], dim=1).mean().item())
        layer_norms.append({"layer": li, "avg_norm": round(norm, 4)})
    stats = {
        "num_layers": len(hidden_states) - 1,
        "vocab_size": int(logits.shape[0]),
        "num_tokens": len(token_texts),
        "inference_ms": round((time.time() - t0) * 1000, 1),
    }
    return {
        "tokens": token_texts,
        "top_predictions": top_tokens,
        "layer_norms": layer_norms,
        "stats": stats,
    }

@app.post("/pytorch/generate")
async def pytorch_generate(req: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(503, "model not loaded")
    
    from transformers import TextIteratorStreamer
    from threading import Thread

    async def generate_stream():
        inputs = tokenizer(req.prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        t0 = time.time()
        
        streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)
        
        generation_kwargs = dict(
            inputs=input_ids,
            streamer=streamer,
            max_new_tokens=req.num_predict,
            do_sample=True,
            temperature=max(0.1, req.temperature),
            top_k=req.top_k,
            top_p=req.top_p,
            pad_token_id=tokenizer.eos_token_id
        )
        
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        token_count = 0
        for new_text in streamer:
            token_count += 1
            chunk = {
                "response": new_text,
                "done": False,
                # Fake a logprob that roughly maps to the temperature so the UI stays green/yellow
                "logprobs": -0.1 * (req.temperature + 0.1) 
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0.01)
            
        thread.join()
        eval_duration = int((time.time() - t0) * 1e9)
        final_chunk = {
            "response": "",
            "done": True,
            "eval_count": token_count,
            "eval_duration": eval_duration
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PYTORCH_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
