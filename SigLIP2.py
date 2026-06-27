import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def _load_visual(device):
    try:
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16-SigLIP2", pretrained="webli"
        )
        visual = model.visual.to(device).eval()
        for p in visual.parameters():
            p.requires_grad = False
        return visual, 768
    except Exception:
        from transformers import AutoModel
        model = AutoModel.from_pretrained("google/siglip2-base-patch16-224")
        visual = model.vision_model.to(device).eval()
        for p in visual.parameters():
            p.requires_grad = False
        return visual, 768


class _Head(nn.Module):
    def __init__(self, dim, n_classes, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Dropout(drop),
            nn.Linear(dim, 256), nn.GELU(),
            nn.Dropout(drop), nn.Linear(256, n_classes),
        )
    def forward(self, x): return self.net(x)


class SigLIP2Classifier:

    def __init__(self, weights="generic", alpha=0.5, pseudo_percentages=None):
        self.weights            = weights
        self.alpha              = alpha
        self.pseudo_percentages = pseudo_percentages or [25, 50, 75, 100]
        self.device             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.visual             = None
        self.head               = None
        self.embed_dim          = None
        self.n_classes          = None
        self._from_transformers = False
        self.lr       = 1e-3
        self.lr2      = 3e-4
        self.epochs_h = 10
        self.epochs_ft= 15
        self.batch    = 32
        self.patience = 5

    @torch.no_grad()
    def _feats(self, X):
        self.visual.eval()
        out = []
        for (Xb,) in DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                                 batch_size=self.batch):
            Xb = Xb.to(self.device)
            if self._from_transformers:
                emb = self.visual(pixel_values=Xb).pooler_output
            else:
                emb = self.visual(Xb)
                if isinstance(emb, (list, tuple)): emb = emb[0]
            out.append(emb.cpu())
        return torch.cat(out).numpy()

    def _loader(self, F, y=None, shuffle=False):
        Ft = torch.tensor(F, dtype=torch.float32)
        ds = TensorDataset(Ft, torch.tensor(y, dtype=torch.long)) if y is not None else TensorDataset(Ft)
        return DataLoader(ds, batch_size=self.batch, shuffle=shuffle)

    def _run(self, tr_F, y_tr, val_F, y_val, opt, crit, epochs, tag):
        tr  = self._loader(tr_F,  y_tr,  shuffle=True)
        val = self._loader(val_F, y_val)
        best_acc, best_state, wait = 0.0, None, 0
        for ep in range(1, epochs + 1):
            self.head.train()
            for Fb, yb in tr:
                Fb, yb = Fb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                crit(self.head(Fb), yb).backward()
                opt.step()
            self.head.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for Fb, yb in val:
                    Fb, yb = Fb.to(self.device), yb.to(self.device)
                    out = self.head(Fb)
                    correct += (out.argmax(1) == yb).sum().item()
                    total   += Fb.size(0)
            va = correct / total
            print(f"[{tag}] ep{ep} va={va:.4f}")
            if va > best_acc:
                best_acc   = va
                best_state = {k: v.cpu().clone() for k, v in self.head.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    print(f"[{tag}] early stop ep{ep}"); break
        if best_state: self.head.load_state_dict(best_state)

    def fit(self, X, y, X_val, y_val):
        self.n_classes = int(y.max()) + 1
        if self.visual is None:
            try:
                self.visual, self.embed_dim = _load_visual(self.device)
                self._from_transformers = False
            except Exception:
                from transformers import AutoModel
                model  = AutoModel.from_pretrained("google/siglip2-base-patch16-224")
                self.visual = model.vision_model.to(self.device).eval()
                for p in self.visual.parameters():
                    p.requires_grad = False
                self.embed_dim          = 768
                self._from_transformers = True

        self.head = _Head(self.embed_dim, self.n_classes).to(self.device)
        crit      = nn.CrossEntropyLoss()
        print("extracting features...")
        tr_F  = self._feats(X)
        val_F = self._feats(X_val)
        opt = optim.Adam(self.head.parameters(), lr=self.lr)
        self._run(tr_F, y, val_F, y_val, opt, crit, self.epochs_h, "head")
        opt = optim.Adam(self.head.parameters(), lr=self.lr2)
        self._run(tr_F, y, val_F, y_val, opt, crit, self.epochs_ft, "finetune")

    def predict_proba(self, X):
        F = self._feats(X)
        self.head.eval()
        out = []
        with torch.no_grad():
            for (Fb,) in self._loader(F):
                out.append(torch.softmax(self.head(Fb.to(self.device)), dim=1).cpu().numpy())
        return np.vstack(out)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
