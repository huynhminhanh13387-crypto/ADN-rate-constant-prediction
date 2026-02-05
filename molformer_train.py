import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import os

# =====================
# 1. global config
# =====================
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 42
set_seed(SEED)
print(f"Global random seed set to {SEED}")

MODEL_NAME = "/home/lcb/MoLFormer-XL-both-10pct"#model can be downloaded at https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct
FILE_PATH = "/home/lcb/data/adn.xlsx"

BATCH_SIZE = 128
EPOCHS = 100
LR = 3e-5
MAX_LEN = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SPLITS = 59
PATIENCE = 42

print(f"Using device: {DEVICE}")

# TensorBoard logs
log_dir_root = "testruns"
os.makedirs(log_dir_root, exist_ok=True)

# =====================
# 2. data preparation
# =====================
df_full = pd.read_excel(FILE_PATH, header=None)
df_full.columns = ["smiles", "T", "lnk"]
df_full["T"] = pd.to_numeric(df_full["T"], errors="coerce")
df_full["lnk"] = pd.to_numeric(df_full["lnk"], errors="coerce")
df_full = df_full.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

groups = df_full['smiles']
print(f"Total cleaned rows: {len(df_full)}, unique SMILES: {len(groups.unique())}")

# =====================
# 3. Dataset
# =====================
class ReactionDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128, text_col="smiles", temp_col="T_norm", target_col="lnk_norm"):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.text_col = text_col
        self.temp_col = temp_col
        self.target_col = target_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row[self.text_col])
        temp = float(row[self.temp_col])
        target = float(row[self.target_col])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "temperature": torch.tensor(temp, dtype=torch.float),
            "targets": torch.tensor(target, dtype=torch.float)
        }

# =====================
# 4. Tokenizer
# =====================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
special_tokens = [".", ">>","I","IIa","IIb","IIc","IId"]
num_added = tokenizer.add_tokens(special_tokens)
print(f"Added {num_added} special tokens")

# =====================
# 5. Attention Pooling
# =====================
class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states, mask):
        scores = self.attn(hidden_states).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        return pooled

# =====================
# 6. model define
# =====================
class MolFormerRegressor(nn.Module):#concat strategy
    def __init__(self, model_name, tokenizer, use_attention_pooling=True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        try:
            self.encoder.resize_token_embeddings(len(tokenizer))
        except Exception:
            pass

        hidden_size = getattr(self.encoder.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.encoder.config, "d_model", None)

        self.use_attention_pooling = use_attention_pooling
        if use_attention_pooling:
            self.pool = AttentionPooling(hidden_size)
        else:
            self.pool = None

        self.regressor = nn.Sequential(
            nn.Linear(hidden_size + 1, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        # self.regressor = nn.Sequential(
        #     nn.Linear(hidden_size + 1, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(512,256),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 1)
        # )#deeper mlp regressor
        # self.regressor = nn.Sequential(
        #     nn.Linear(hidden_size + 1, 769),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(769, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 1)
        # )#wider mlp regressor

    def forward(self, input_ids, attention_mask, temperature, output_attentions=False):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )
        hidden = outputs.last_hidden_state
        if self.use_attention_pooling:
            pooled = self.pool(hidden, attention_mask)
        else:
            pooled = hidden[:, 0, :]
        x = torch.cat([pooled, temperature.unsqueeze(1)], dim=1)
        logits = self.regressor(x).squeeze(-1)
        if output_attentions:
            return logits, outputs.attentions
        return logits
# class MolFormerRegressor(nn.Module):#(add strategy)
#     def __init__(self, model_name, tokenizer, use_attention_pooling=True):
#         super().__init__()
#         self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
#         self.encoder.resize_token_embeddings(len(tokenizer))
#         hidden_size = getattr(self.encoder.config, "hidden_size", getattr(self.encoder.config, "d_model", None))
#         self.pool = AttentionPooling(hidden_size) if use_attention_pooling else None
#         self.temp_encoder = nn.Sequential(nn.Linear(1, hidden_size), nn.Tanh())
#         self.regressor = nn.Sequential(
#             nn.Linear(hidden_size, 512), nn.ReLU(), nn.Dropout(0.5),
#             nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 1)
#         )
#
#     def forward(self, input_ids, attention_mask, temperature, output_attentions=False):
#         outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_attentions=output_attentions)
#         hidden = outputs.last_hidden_state
#         pooled = self.pool(hidden, attention_mask) if self.pool else hidden[:, 0, :]
#         temp_vec = self.temp_encoder(temperature.unsqueeze(1))
#         fused = pooled + temp_vec
#         logits = self.regressor(fused).squeeze(-1)
#         if output_attentions:
#             return logits, outputs.attentions
#         return logits
# =====================
# 7. Metrics
# =====================
def compute_metrics(preds, reals, scaler_y):
    preds = preds.cpu().numpy() if isinstance(preds, torch.Tensor) else preds
    reals = reals.cpu().numpy() if isinstance(reals, torch.Tensor) else reals
    preds_real = scaler_y.inverse_transform(preds.reshape(-1, 1)).ravel()
    reals_real = scaler_y.inverse_transform(reals.reshape(-1, 1)).ravel()
    rmse = np.sqrt(mean_squared_error(reals_real, preds_real))
    r2 = r2_score(reals_real, preds_real)
    return rmse, r2, preds_real, reals_real

# =====================
# 8. Train & Eval
# =====================
def train_one_epoch(model, loader, loss_fn, optimizer, scheduler):
    model.train()
    losses = []
    for batch in tqdm(loader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        temps = batch["temperature"].to(DEVICE)
        targets = batch["targets"].to(DEVICE)

        preds = model(input_ids, attention_mask, temps)
        loss = loss_fn(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

    return np.mean(losses)

def evaluate_model(model, loader, loss_fn, desc="Evaluating"):
    model.eval()
    losses, preds, reals = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            temps = batch["temperature"].to(DEVICE)
            targets = batch["targets"].to(DEVICE)

            out = model(input_ids, attention_mask, temps)
            loss = loss_fn(out, targets)
            losses.append(loss.item())
            preds.extend(out.cpu().numpy())
            reals.extend(targets.cpu().numpy())
    return np.mean(losses), np.array(preds), np.array(reals)

# =====================
# 9. Cross-Validation with TensorBoard
# =====================
kfold = GroupKFold(n_splits=N_SPLITS)
fold_results = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(df_full, groups=groups)):
    print(f"\n========== Fold {fold+1}/{N_SPLITS} ==========")
    set_seed(SEED)

    # create TensorBoard writer
    fold_log_dir = os.path.join(log_dir_root, f"fold_{fold+1}")
    writer = SummaryWriter(log_dir=fold_log_dir)
    print(f"TensorBoard logging to {fold_log_dir}")

    train_df = df_full.iloc[train_idx].copy()
    test_df = df_full.iloc[test_idx].copy()

    scaler_T = StandardScaler()
    scaler_y = StandardScaler()
    # scaler_T = MinMaxScaler()
    # scaler_y = MinMaxScaler()
    scaler_T.fit(train_df[["T"]])
    scaler_y.fit(train_df[["lnk"]].values.reshape(-1, 1))

    for df in [train_df, test_df]:
        df["T_norm"] = scaler_T.transform(df[["T"]])
        df["lnk_norm"] = scaler_y.transform(df[["lnk"]].values.reshape(-1, 1))

    train_loader = DataLoader(ReactionDataset(train_df, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(ReactionDataset(test_df, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    model = MolFormerRegressor(MODEL_NAME, tokenizer, use_attention_pooling=True).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader)*EPOCHS, eta_min=1e-7)
    loss_fn = nn.SmoothL1Loss()

    best_rmse, patience = 1e9, 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, scheduler)
        train_eval_loss, train_preds, train_reals = evaluate_model(model, train_loader, loss_fn, "Train Eval")
        test_loss, test_preds, test_reals = evaluate_model(model, test_loader, loss_fn, "Test Eval")

        train_rmse, train_r2, _, train_real_y = compute_metrics(train_preds, train_reals, scaler_y)
        test_rmse, test_r2, _, test_real_y = compute_metrics(test_preds, test_reals, scaler_y)

        train_nrmse = train_rmse / np.std(train_real_y)
        test_nrmse = test_rmse / np.std(test_real_y)

        # ===== output control =====
        print(f"Epoch {epoch}: "
              f"Train R²={train_r2:.4f}, RMSE={train_rmse:.4f}, NRMSE={train_nrmse:.4f} | "
              f"Test R²={test_r2:.4f}, RMSE={test_rmse:.4f}, NRMSE={test_nrmse:.4f}")

        # ===== TensorBoard Logging =====
        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Train/R2", train_r2, epoch)
        writer.add_scalar("Train/RMSE", train_rmse, epoch)
        writer.add_scalar("Train/NRMSE", train_nrmse, epoch)

        writer.add_scalar("Test/Loss", test_loss, epoch)
        writer.add_scalar("Test/R2", test_r2, epoch)
        writer.add_scalar("Test/RMSE", test_rmse, epoch)
        writer.add_scalar("Test/NRMSE", test_nrmse, epoch)

        # ===== Early Stopping =====
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            patience = 0
            torch.save(model.state_dict(), f"5fold_best_model_fold_{fold+1}.pth")
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    writer.close()
    fold_results.append({"fold": fold+1, "best_rmse": best_rmse})
