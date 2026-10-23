#!/usr/bin/env python3
"""
Phase 2: Shared encoder + three task heads.
Backbone: sentence-transformers/all-MiniLM-L6-v2 (English, Apache 2.0)
Architecture:
  - One shared encoder (MiniLM-L6-v2, 22M params, 384-dim)
  - Three classification heads (Linear layers on top of pooled embedding + structured features)
  - Route: 384+struct_dim -> 256 -> 5
  - Approve: 384+struct_dim -> 128 -> 1
  - Grounded: 384+struct_dim -> 128 -> 1

Training: multi-task, batch balanced across three tasks.
"""

import json
import math
import os
import pickle
import sys
import time
import unicodedata
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
sys.path.insert(0, str(ROOT / "src"))

from baseline import (
    clean_text, safe_str, tier_onehot, amount_features,
    text_length_features, overlap_features, ROUTE_CLASSES, TIERS,
    load_jsonl_safe
)

# -----------------------------------------------------------------------
# Check for torch availability
# -----------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARN] PyTorch not available. Phase 2 requires torch.")

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False
    print("[WARN] sentence-transformers not available.")


# -----------------------------------------------------------------------
# Structured feature dims
# -----------------------------------------------------------------------
STRUCT_DIM_ROUTE = len(TIERS) + 7 + 4   # tier(3) + amount(7) + lengths(4) = 14
STRUCT_DIM_APPROVE = len(TIERS) + 7 + 4  # same = 14
STRUCT_DIM_GROUNDED = len(TIERS) + 7 + 4  # tier(3) + amount(7) + overlap(4) = 14
EMB_DIM = 384  # MiniLM-L6-v2 embedding dim


# -----------------------------------------------------------------------
# Task-tagged serializers (same as baseline)
# -----------------------------------------------------------------------

def serialize_route(state: dict) -> str:
    subject = clean_text(safe_str(state.get("subject", "")))
    body = clean_text(safe_str(state.get("body", "")))
    tier = clean_text(safe_str(state.get("customer_tier", "")))
    amount = state.get("amount", 0) or 0
    return f"[ROUTE] {subject} [SEP] {body} [SEP] {tier} | {amount:.2f}"


def serialize_approve(state: dict) -> str:
    subject = clean_text(safe_str(state.get("subject", "")))
    body = clean_text(safe_str(state.get("body", "")))
    tier = clean_text(safe_str(state.get("customer_tier", "")))
    amount = state.get("amount", 0) or 0
    return f"[APPROVE] {subject} [SEP] {body} [SEP] {tier} | {amount:.2f}"


def serialize_grounded(state: dict) -> str:
    reply = clean_text(safe_str(state.get("draft_reply", "")))
    article = clean_text(safe_str(state.get("kb_article", "")))
    return f"[GROUNDED] {reply} [SEP] {article}"


def structured_route(state: dict) -> np.ndarray:
    tier = safe_str(state.get("customer_tier", ""))
    amount = state.get("amount", 0)
    subject = clean_text(safe_str(state.get("subject", "")))
    body = clean_text(safe_str(state.get("body", "")))
    return np.concatenate([tier_onehot(tier), amount_features(amount), text_length_features(subject, body)])


def structured_approve(state: dict) -> np.ndarray:
    return structured_route(state)


def structured_grounded(state: dict) -> np.ndarray:
    tier = safe_str(state.get("customer_tier", ""))
    amount = state.get("amount", 0)
    reply = clean_text(safe_str(state.get("draft_reply", "")))
    article = clean_text(safe_str(state.get("kb_article", "")))
    return np.concatenate([tier_onehot(tier), amount_features(amount), overlap_features(reply, article)])


# -----------------------------------------------------------------------
# PyTorch heads
# -----------------------------------------------------------------------

class RouteHead(nn.Module):
    """Route classification head: emb+struct -> 5 classes."""
    def __init__(self, input_dim: int, num_classes: int = 5, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class BinaryHead(nn.Module):
    """Binary classification head: emb+struct -> 1 logit."""
    def __init__(self, input_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)


class SharedEncoderModel(nn.Module):
    """
    Shared encoder + three heads.
    The backbone is sentence-transformers/all-MiniLM-L6-v2.
    Heads are task-specific linear layers.
    """
    def __init__(self, encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        from sentence_transformers import SentenceTransformer
        # Load as a plain SentenceTransformer for embedding extraction
        # We'll fine-tune by encoding with gradient
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name)

        self.route_head = RouteHead(EMB_DIM + STRUCT_DIM_ROUTE)
        self.approve_head = BinaryHead(EMB_DIM + STRUCT_DIM_APPROVE)
        self.grounded_head = BinaryHead(EMB_DIM + STRUCT_DIM_GROUNDED)

    def encode(self, texts: List[str], max_length: int = 128) -> torch.Tensor:
        """Encode texts using the shared encoder. Returns pooled embeddings."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(next(self.parameters()).device) for k, v in encoded.items()}
        output = self.encoder(**encoded)
        # Mean pooling
        attention_mask = encoded["attention_mask"]
        token_embeddings = output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
        return embeddings

    def forward(self, texts: List[str], struct_feats: torch.Tensor, task: str, max_length: int = 128):
        embeddings = self.encode(texts, max_length)
        combined = torch.cat([embeddings, struct_feats], dim=1)
        if task == "route":
            return self.route_head(combined)
        elif task == "approve":
            return self.approve_head(combined)
        elif task == "grounded":
            return self.grounded_head(combined)
        else:
            raise ValueError(f"Unknown task: {task}")


# -----------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------

class DecideDataset(Dataset):
    def __init__(self, records: List[dict], task: str):
        self.records = records
        self.task = task
        self.texts = []
        self.struct_feats = []
        self.labels = []

        for r in records:
            state = r.get("state", {}) or {}
            labels = r.get("labels", {}) or {}

            if task == "route":
                self.texts.append(serialize_route(state))
                self.struct_feats.append(structured_route(state))
                route_cls = labels.get("route", "general")
                self.labels.append(ROUTE_CLASSES.index(route_cls) if route_cls in ROUTE_CLASSES else 4)
            elif task == "approve":
                self.texts.append(serialize_approve(state))
                self.struct_feats.append(structured_approve(state))
                self.labels.append(int(bool(labels.get("approve", False))))
            elif task == "grounded":
                self.texts.append(serialize_grounded(state))
                self.struct_feats.append(structured_grounded(state))
                self.labels.append(int(bool(labels.get("grounded", False))))

        self.struct_feats = np.array(self.struct_feats, dtype=np.float32)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "struct": self.struct_feats[idx],
            "label": self.labels[idx],
        }


# -----------------------------------------------------------------------
# Inference wrapper (sklearn-compatible interface)
# -----------------------------------------------------------------------

class EncoderModel:
    """
    Wraps SharedEncoderModel for inference.
    Compatible with the same predict() interface as BaselineModel.
    Uses ONNX for fast inference (loaded separately in Phase 4).
    """

    def __init__(self, model: "SharedEncoderModel" = None, device: str = "cpu"):
        self.model = model
        self.device = device
        self.thresholds = {
            "route": 0.5,
            "approve": 0.75,
            "grounded": 0.5,
        }
        # Temperature scaling params
        self.temperature = {"route": 1.0, "approve": 1.0, "grounded": 1.0}
        # ONNX session (set during Phase 4)
        self.ort_session = None
        self.ort_mode = False

    def _softmax(self, logits: np.ndarray, temp: float = 1.0) -> np.ndarray:
        logits = logits / temp
        logits -= logits.max()
        exp = np.exp(logits)
        return exp / exp.sum()

    def _sigmoid(self, logit: float, temp: float = 1.0) -> float:
        return 1.0 / (1.0 + math.exp(-logit / temp))

    def predict(self, state: dict, decision: str, request_id: str = "") -> dict:
        """Predict using ONNX (preferred) or PyTorch model."""
        t0 = time.perf_counter()

        if self.ort_mode and self.ort_session:
            result = self._predict_onnx(state, decision, request_id)
        else:
            result = self._predict_torch(state, decision, request_id)

        latency_ms = (time.perf_counter() - t0) * 1000
        result["latency_ms"] = round(latency_ms, 3)
        return result

    def _predict_torch(self, state: dict, decision: str, request_id: str) -> dict:
        import torch
        self.model.eval()
        with torch.no_grad():
            if decision == "route":
                texts = [serialize_route(state)]
                struct = torch.tensor([structured_route(state)], dtype=torch.float32).to(self.device)
                logits = self.model(texts, struct, "route", max_length=128).cpu().numpy()[0]
                probs = self._softmax(logits, self.temperature["route"])
                prob_dict = {cls: float(p) for cls, p in zip(ROUTE_CLASSES, probs)}
                value = ROUTE_CLASSES[np.argmax(probs)]
                confidence = float(prob_dict[value])
                abstain = confidence < self.thresholds["route"]
                return {
                    "request_id": request_id,
                    "decision": decision,
                    "type": "choice",
                    "value": value,
                    "probs": {k: round(v, 6) for k, v in prob_dict.items()},
                    "confidence": round(confidence, 6),
                    "abstain": bool(abstain),
                    "latency_ms": 0.0,
                }
            elif decision in ("approve", "grounded"):
                if decision == "approve":
                    texts = [serialize_approve(state)]
                    struct = torch.tensor([structured_approve(state)], dtype=torch.float32).to(self.device)
                else:
                    texts = [serialize_grounded(state)]
                    struct = torch.tensor([structured_grounded(state)], dtype=torch.float32).to(self.device)
                logit = self.model(texts, struct, decision, max_length=128).cpu().numpy()[0][0]
                p_true = self._sigmoid(logit, self.temperature[decision])
                value = p_true >= 0.5
                confidence = max(p_true, 1 - p_true)
                abstain = confidence < self.thresholds[decision]
                return {
                    "request_id": request_id,
                    "decision": decision,
                    "type": "binary",
                    "value": bool(value),
                    "p_true": round(p_true, 6),
                    "abstain": bool(abstain),
                    "latency_ms": 0.0,
                }

    def _predict_onnx(self, state: dict, decision: str, request_id: str) -> dict:
        """ONNX inference path (set up in Phase 4)."""
        # Will be implemented after ONNX export
        raise NotImplementedError("ONNX path not yet set up")

    def save(self, path: Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        # Save torch model weights
        if self.model:
            import torch
            torch.save(self.model.state_dict(), path / "encoder_weights.pt")
        # Save wrapper (without model weights for pickling)
        model_backup = self.model
        self.model = None
        with open(path / "encoder_model.pkl", "wb") as f:
            pickle.dump(self, f, protocol=5)
        self.model = model_backup
        print(f"  Encoder model wrapper saved to {path}/encoder_model.pkl")

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "EncoderModel":
        path = Path(path)
        with open(path / "encoder_model.pkl", "rb") as f:
            wrapper = pickle.load(f)
        # Reload torch model
        if (path / "encoder_weights.pt").exists():
            import torch
            model = SharedEncoderModel()
            model.load_state_dict(torch.load(path / "encoder_weights.pt", map_location=device))
            model.eval()
            wrapper.model = model
            wrapper.device = device
        return wrapper


# -----------------------------------------------------------------------
# Training script
# -----------------------------------------------------------------------

def train_encoder(
    train_records: List[dict],
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 2e-5,
    device: str = "cpu",
):
    """
    Multi-task training of shared encoder + three heads.
    Task batches are balanced (round-robin across tasks).
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required for Phase 2 training")

    import torch
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    model = SharedEncoderModel()
    model.to(device)
    model.train()

    # Build datasets
    route_ds = DecideDataset(train_records, "route")
    approve_ds = DecideDataset(train_records, "approve")
    grounded_ds = DecideDataset(train_records, "grounded")

    def collate(batch):
        texts = [b["text"] for b in batch]
        structs = torch.tensor(np.array([b["struct"] for b in batch]), dtype=torch.float32)
        labels = torch.tensor([b["label"] for b in batch])
        return texts, structs, labels

    route_dl = DataLoader(route_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    approve_dl = DataLoader(approve_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    grounded_dl = DataLoader(grounded_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = epochs * max(len(route_dl), len(approve_dl), len(grounded_dl)) * 3
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # Losses
    route_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    binary_criterion = nn.BCEWithLogitsLoss()

    print(f"\nTraining encoder model for {epochs} epochs on {device}...")
    print(f"  Route batches: {len(route_dl)}, Approve: {len(approve_dl)}, Grounded: {len(grounded_dl)}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_steps = 0

        route_iter = iter(route_dl)
        approve_iter = iter(approve_dl)
        grounded_iter = iter(grounded_dl)

        while True:
            # Balance tasks: one batch per task per step
            done = 0
            for task_name, loader_iter, criterion, is_binary in [
                ("route", route_iter, route_criterion, False),
                ("approve", approve_iter, binary_criterion, True),
                ("grounded", grounded_iter, binary_criterion, True),
            ]:
                try:
                    texts, structs, labels = next(loader_iter)
                except StopIteration:
                    done += 1
                    continue

                structs = structs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                logits = model(texts, structs, task_name)

                if is_binary:
                    loss = criterion(logits.squeeze(1), labels.float())
                else:
                    loss = criterion(logits, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                n_steps += 1

            if done == 3:
                break

        avg_loss = total_loss / max(n_steps, 1)
        print(f"  Epoch {epoch+1}/{epochs}: avg_loss={avg_loss:.4f}")

    return model


if __name__ == "__main__":
    print("Phase 2: Training encoder model...")
    print("NOTE: Requires: pip install sentence-transformers transformers torch")

    try:
        import torch
        import transformers
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Run: pip install sentence-transformers transformers torch")
        sys.exit(1)

    train_path = ROOT / "data" / "train.jsonl"
    dev_path = ROOT / "data" / "dev.jsonl"

    print(f"Loading data...")
    train_records = load_jsonl_safe(str(train_path))
    dev_records = load_jsonl_safe(str(dev_path))

    # Reserve 10% for calibration
    np.random.seed(42)
    n_total = len(train_records)
    indices = np.random.permutation(n_total)
    n_calib = max(int(n_total * 0.1), 100)
    calib_idx = set(indices[:n_calib])
    fit_records = [train_records[i] for i in range(n_total) if i not in calib_idx]
    print(f"  Fit: {len(fit_records)}, Calib: {n_calib}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    torch_model = train_encoder(fit_records, epochs=3, batch_size=32, lr=2e-5, device=device)

    wrapper = EncoderModel(torch_model, device=device)
    wrapper.save(ARTIFACTS_DIR)
    print(f"\n✓ Phase 2 model saved to {ARTIFACTS_DIR}")
