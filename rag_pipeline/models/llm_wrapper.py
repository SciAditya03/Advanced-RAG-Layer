# FILE: rag_pipeline/models/llm_wrapper.py
"""
Optimized LLM wrapper with fast fallback and timeout protection.
"""

import logging
import time
import re
from typing import Optional

from ..config import LLM_MODEL, DEVICE, PRELOAD_LLM, LLM_LOAD_TIMEOUT, USE_LLM

__all__ = ["generate_answer", "load_llm", "LLM_AVAILABLE"]

logger = logging.getLogger(__name__)

# Global state
_llm_pipe = None
_llm_tokenizer = None
LLM_AVAILABLE = False
_LLM_LOAD_ERROR = None


def load_llm(model_name: Optional[str] = None, device: Optional[str] = None) -> bool:
    """Load LLM with timeout and fallback."""
    global _llm_pipe, _llm_tokenizer, LLM_AVAILABLE, _LLM_LOAD_ERROR
    
    if LLM_AVAILABLE or _LLM_LOAD_ERROR:
        return LLM_AVAILABLE
    if not USE_LLM:
        logger.debug("⚠️  LLM loading skipped (USE_LLM=False in config)")
        return False
    
    model_name = model_name or LLM_MODEL
    device = device or DEVICE
    start = time.time()
    logger.info(f"🔄 Loading LLM: {model_name} on {device}...")
    
    try:
        import signal
        def timeout_handler(signum, frame):
            raise TimeoutError(f"LLM load exceeded {LLM_LOAD_TIMEOUT}s")
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(LLM_LOAD_TIMEOUT)
        
        from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForCausalLM
        import torch
        
        _llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        torch_dtype = torch.float16 if device == "cuda" and torch.cuda.is_available() else torch.float32
        
        _llm_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" and torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
            load_in_4bit=True if torch_dtype == torch.float16 else False,
        )
        
        _llm_pipe = hf_pipeline(
            "text-generation", model=_llm_model, tokenizer=_llm_tokenizer,
            max_new_tokens=256, do_sample=False, return_full_text=False,
            pad_token_id=_llm_tokenizer.eos_token_id,
        )
        
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        
        elapsed = time.time() - start
        LLM_AVAILABLE = True
        logger.info(f"✅ LLM loaded in {elapsed:.1f}s")
        return True
        
    except TimeoutError:
        _LLM_LOAD_ERROR = "timeout"
        logger.error(f"❌ LLM load timed out after {LLM_LOAD_TIMEOUT}s")
    except Exception as e:
        _LLM_LOAD_ERROR = str(e)
        logger.warning(f"⚠️  LLM load failed: {e}")
    finally:
        signal.alarm(0)
        try: signal.signal(signal.SIGALRM, signal.SIG_DFL)
        except: pass
    
    logger.warning("   Falling back to context-echo mode (instant responses)")
    return False


def generate_answer(prompt: str, max_new_tokens: int = 256, temperature: float = 0.0) -> str:
    """Generate answer with fast fallback."""
    if not USE_LLM:
        return _context_echo_fallback(prompt)
    
    if not LLM_AVAILABLE and _llm_pipe is None and not _LLM_LOAD_ERROR:
        load_llm()
    
    if _llm_pipe is not None and LLM_AVAILABLE:
        try:
            start = time.time()
            output = _llm_pipe(prompt, max_new_tokens=max_new_tokens, do_sample=(temperature > 0))[0]["generated_text"]
            elapsed = time.time() - start
            logger.debug(f"🤖 LLM generation: {len(output)} chars in {elapsed:.2f}s")
            return output.strip()
        except Exception as e:
            logger.error(f"❌ LLM generation failed: {e}")
            return _context_echo_fallback(prompt)
    
    return _context_echo_fallback(prompt)


def _context_echo_fallback(prompt: str, max_chars: int = 500) -> str:
    """Extract and return relevant context as instant fallback answer."""
    context_marker = "CONTEXT:\n"
    query_marker = "OFFICER QUERY:"
    
    if context_marker in prompt and query_marker in prompt:
        start = prompt.find(context_marker) + len(context_marker)
        end = prompt.find(query_marker)
        context = prompt[start:end].strip()
        paragraphs = [p.strip() for p in context.split("\n\n") if len(p.strip()) > 30]
        if paragraphs:
            return f"[Fast mode] {paragraphs[0][:max_chars]}"
    
    query_match = re.search(r"OFFICER QUERY:\s*(.+?)(?:\n|$)", prompt)
    query = query_match.group(1).strip() if query_match else "your question"
    return f"[Fast mode] Based on retrieved knowledge about {query[:50]}... [See citations below]"


# ── Test ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_prompt = """You are an AI governance instructor...
CONTEXT:
Vendor lock-in creates dependency risks.
OFFICER QUERY: What is vendor lock-in?
RESPONSE:"""
    result = generate_answer(test_prompt)
    print(f"✅ llm_wrapper test — response: {result[:100]}...")