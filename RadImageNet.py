import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet50


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        base = resnet50(weights=None)
        self.backbone = nn.Sequential(*list(base.children())[:9])

    def forward(self, x):
        return self.backbone(x)


class _Classifier(nn.Module):
    def __init__(self, n_classes, drop=0.5):
        super().__init__()
        self.drop = nn.Dropout(drop)
        self.fc   = nn.Linear(2048, n_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        return self.fc(x)


class RadImageNetClassifier:

    def __init__(self, weights="medical", alpha=0.5, pseudo_percentages=None):
        self.weights            = weights
        self.alpha              = alpha
        self.pseudo_percentages = pseudo_percentages or [25, 50, 75, 100]
        self.device             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone           = None
        self.classifier         = None
        self.n_classes          = None
        self.lr_head   = 1e-3
        self.lr_ft     = 1e-4
        self.epochs_h  = 10
        self.epochs_ft = 20
        self.batch     = 32
        self.patience  = 5

    def _build(self, n_classes):
        backbone   = _Backbone()
        classifier = _Classifier(n_classes)

        if self.weights == "medical":
            ckpt = os.path.join("pretrained", "RadImageNet-ResNet50.pt")
            if os.path.exists(ckpt):
                state = torch.load(ckpt, map_location="cpu")
                if "model" in state:
                    state = state["model"]
                backbone.load_state_dict(state, strict=True)
                print("[INFO] RadImageNet backbone loaded correctly")
            else:
                warnings.warn(f"RadImageNet checkpoint not found at {ckpt}.")
        else:
            from torchvision.models import ResNet50_Weights
            base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            backbone.backbone.load_state_dict(
                nn.Sequential(*list(base.children())[:9]).state_dict()
            )

        self.backbone   = backbone.to(self.device)
        self.classifier = classifier.to(self.device)

    def _freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def _unfreeze_backbone(self, n_groups=2):
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.backbone.backbone[-n_groups:].parameters():
            p.requires_grad = True

    def _forward(self, x):
        return self.classifier(self.backbone(x))

    def _loader(self, X, y=None, shuffle=False):
        Xt = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            ds = TensorDataset(Xt, torch.tensor(y, dtype=torch.long))
        else:
            ds = TensorDataset(Xt)
        return DataLoader(ds, batch_size=self.batch, shuffle=shuffle)

    def _train_epoch(self, loader, opt, crit):
        self.backbone.train()
        self.classifier.train()
        loss_sum, correct, total = 0.0, 0, 0
        for Xb, yb in loader:
            Xb, yb = Xb.to(self.device), yb.to(self.device)
            opt.zero_grad()
            out  = self._forward(Xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * Xb.size(0)
            correct  += (out.argmax(1) == yb).sum().item()
            total    += Xb.size(0)
        return loss_sum / total, correct / total

    @torch.no_grad()
    def _val_epoch(self, loader, crit):
        self.backbone.eval()
        self.classifier.eval()
        loss_sum, correct, total = 0.0, 0, 0
        for Xb, yb in loader:
            Xb, yb = Xb.to(self.device), yb.to(self.device)
            out  = self._forward(Xb)
            loss = crit(out, yb)
            loss_sum += loss.item() * Xb.size(0)
            correct  += (out.argmax(1) == yb).sum().item()
            total    += Xb.size(0)
        return loss_sum / total, correct / total

    def _run(self, tr_loader, val_loader, opt, crit, epochs, tag):
        best_acc, best_bb, best_clf, wait = 0.0, None, None, 0
        for ep in range(1, epochs + 1):
            tl, ta = self._train_epoch(tr_loader, opt, crit)
            vl, va = self._val_epoch(val_loader, crit)
            print(f"[{tag}] ep{ep} tl={tl:.4f} ta={ta:.4f} vl={vl:.4f} va={va:.4f}")
            if va > best_acc:
                best_acc = va
                best_bb  = {k: v.cpu().clone() for k, v in self.backbone.state_dict().items()}
                best_clf = {k: v.cpu().clone() for k, v in self.classifier.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    print(f"[{tag}] early stop ep{ep}")
                    break
        if best_bb:
            self.backbone.load_state_dict(best_bb)
            self.classifier.load_state_dict(best_clf)

    def fit(self, X, y, X_val, y_val):
        self.n_classes = int(y.max()) + 1
        self._build(self.n_classes)
        crit = nn.CrossEntropyLoss()
        tr   = self._loader(X, y, shuffle=True)
        val  = self._loader(X_val, y_val)

        self._freeze_backbone()
        opt = optim.Adam(self.classifier.parameters(), lr=self.lr_head)
        self._run(tr, val, opt, crit, self.epochs_h, "head")

        self._unfreeze_backbone()
        opt = optim.Adam([
            {"params": self.classifier.parameters(), "lr": self.lr_head},
            {"params": [p for p in self.backbone.parameters() if p.requires_grad], "lr": self.lr_ft},
        ])
        self._run(tr, val, opt, crit, self.epochs_ft, "finetune")

    @torch.no_grad()
    def predict_proba(self, X):
        self.backbone.eval()
        self.classifier.eval()
        out = []
        for (Xb,) in self._loader(X):
            logits = self._forward(Xb.to(self.device))
            out.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.vstack(out)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
