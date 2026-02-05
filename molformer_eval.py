import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# Try importing RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import Draw

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


# =====================
# 1. Configuration and Global Seed
# =====================
def set_seed(seed_value=42):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set reactions for visualization (Rank 1, 2, 3)
BEST_REACTIONS = [
    # (52, "ONONN(=O)O>>ONN(=O)O.[N]=O"),  # Rank 1
    # (16, "[NH4+][N-]([N+](=O)[O-])[N+](=O)[O-]-IIb>>[N]N(=O)[O-][NH4+].N(=O)[O]"),  # Rank 2
    # (12, "[NH4+][N-]([N+](=O)[O-])[N+](=O)[O-]-IIc>>[NH4+][N-]([N+](=O)[O-])[N+](=O)[O-]-IId")  # Rank 3
    # (48,"[H+][N-]([N+](=O)[O-])[N+](=O)[O-]-I.[H+][N-]([N+](=O)[O-])[N+](=O)[O-]-I>>[H+][N-]([N+](=O)[O-])[N+](=O)[O-]-IIa.[H+][N-]([N+](=O)[O-])[N+](=O)[O-]-IIa")#4
    # (51,"ONONN(=O)[O-]>>N(=O)N=O.N(=O)[O-]")#5
    # (58,"N(=O)N=O>>[N]=O.[N]=O")#6
    # (24,"[NH4+][N-]([N+](=O)[O-])[N+](=O)[O-]-IIa>>[NH4+].[N+](=O)([O-])[O-].[N-]=[N+]=O")#7
    # (18,"[NH4+][N-]([N+](=O)[O-])[N+](=O)[O-]-IIb>>[NH4+].[N-]([N+](=O)[O-])[N+](=O)[O-]")#8
    # (46,"[H+][N-]([N+](=O)[O-])[N+](=O)[O-]-I>>[N+](=O)(O)[O-].[N-]=[N+]=O")#9
    # (14,"[NH4+][N-]([N+](=O)[O-])[N+](=O)[O-]-IIc>>[NH4+].[N+](=O)([O-])[O-].[N-]=[N+]=O")#10

    # (29, "[NH4+][N-]([N+](=O)[O-])[N+](=O)[O-]-I>>N.H[N-]([N+](=O)[O-])[N+](=O)[O-]-I")  # worst1
    (5, "[NH]N(=O)O>>N[N](=O)O")  # worst2
    # (39,"[H+][N-]([N+](=O)[O-])[N+](=O)[O-]-IIb>>[H+][N-]([N+](=O)[O-])[N+](=O)[O-]-IIc") #3
    # (33,"[H+][N-]([N+](=O)[O-])[N+](=O)[O-]-IId>>[H+][N-]([N+](=O)[O-])[N+](=O)[O-]-IIa") #4
    # (27,"[NH4+][N-]([N+](=O)[O-])[N+](=O)[O-]-I>>[NH4+][N-]([N+](=O)[O-])[N+](=O)[O-]-IIa") #5
    # (1,"[N]N(=O)[O-][NH4+]>>[N-]=[N+]=O.N.[OH]") #6
    # (36,"[H+][N-]([N+](=O)[O-])[N+](=O)[O-]-IIc>>[H+][N-]([N+](=O)[O-])[N+](=O)[O-]-IId") #7
    # (54,"ONN(=O)O>>N=O.N(=O)[O]") #8
    # (30,"[NH4+].[N-]([N+](=O)[O-])[N+](=O)[O-]>>[NH4+].[N+](=O)([O-])[O-].[N-]=[N+]=O") #9
    # (7,"[NH]N(=O)O.O>>N[N](=O)O.O") #10
]
SEED = 42
set_seed(SEED)

MODEL_NAME = "/home/lcb/MoLFormer-XL-both-10pct"
FILE_PATH = "/home/lcb/data/adn.xlsx"
LOG_DIR_ROOT = "runs_loo"
MAX_LEN = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================
# 2. Model Definition (Supports Self-Attention Output)
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
        return pooled, weights


class MolFormerRegressor(nn.Module):
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

    def forward(self, input_ids, attention_mask, temperature):
        # output_attentions=True enables attention output
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        hidden = outputs.last_hidden_state

        # Extract the Self-Attention Matrix from the last layer
        # Shape: (Batch, Num_Heads, Seq_Len, Seq_Len)
        last_layer_attn = outputs.attentions[-1]

        # Average over all heads to get the aggregated attention matrix
        # Shape: (Batch, Seq_Len, Seq_Len)
        avg_attn_matrix = torch.mean(last_layer_attn, dim=1)

        attn_weights = None
        if self.use_attention_pooling:
            pooled, attn_weights = self.pool(hidden, attention_mask)
        else:
            pooled = hidden[:, 0, :]

        x = torch.cat([pooled, temperature.unsqueeze(1)], dim=1)
        logits = self.regressor(x).squeeze(-1)

        return logits, attn_weights, avg_attn_matrix


# =====================
# 3. Inference and Visualization Utility Functions
# =====================

def load_model_for_fold(fold_idx, tokenizer):
    model_path = os.path.join(f"loo_best_model_fold_{fold_idx}.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(LOG_DIR_ROOT, f"loo_best_model_fold_{fold_idx}.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = MolFormerRegressor(MODEL_NAME, tokenizer).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"Loaded model for Fold {fold_idx}")
    return model


def visualize_reaction(fold_idx, target_smiles, df_full, tokenizer, scaler_T, scaler_y):
    print(f"\n>>> Visualizing Fold {fold_idx}: {target_smiles}")

    # 1. Prepare data
    reaction_data = df_full[df_full['smiles'] == target_smiles].copy()
    reaction_data = reaction_data.sort_values(by='T')

    if len(reaction_data) == 0:
        return

    texts = [str(s) for s in reaction_data['smiles'].tolist()]
    temps_raw = reaction_data['T'].values
    lnks_raw = reaction_data['lnk'].values
    temps_norm = scaler_T.transform(temps_raw.reshape(-1, 1)).flatten()

    inputs = tokenizer(texts, padding="max_length", max_length=MAX_LEN, truncation=True, return_tensors="pt")
    input_ids = inputs['input_ids'].to(DEVICE)
    attention_mask = inputs['attention_mask'].to(DEVICE)
    temps_tensor = torch.tensor(temps_norm, dtype=torch.float).to(DEVICE)

    # 2. Model inference
    model = load_model_for_fold(fold_idx, tokenizer)
    with torch.no_grad():
        preds_norm, pooling_weights, self_attn_matrix = model(input_ids, attention_mask, temps_tensor)

    preds_real = scaler_y.inverse_transform(preds_norm.cpu().numpy().reshape(-1, 1)).ravel()
    rmse = np.sqrt(mean_squared_error(lnks_raw, preds_real))
    r2 = r2_score(lnks_raw, preds_real)

    # =====================
    # Plot 1: Arrhenius Plot (Keep original style)
    # =====================
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    sns.scatterplot(x=reaction_data['T'], y=lnks_raw, color='black', label='R55 Calculated', s=100, alpha=0.8,
                    edgecolor='w')
    sns.lineplot(x=reaction_data['T'], y=preds_real, color='#E63946', label='R55 Predicted', linewidth=3)

    metrics_text = f"RMSE = {rmse:.4f}\n$R^2$ = {r2:.4f}"
    plt.gca().text(0.05, 0.05, metrics_text, transform=plt.gca().transAxes,
                   fontsize=16, fontweight='bold', verticalalignment='bottom',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

    plt.xlabel("1000/T ($K^{-1}$)", fontsize=24, fontweight='bold')
    plt.ylabel("lnk", fontsize=24, fontweight='bold')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc='upper right', fontsize=14, frameon=True)
    plt.tight_layout()
    plt.savefig(f"arrhenius_fold_{fold_idx}.png", dpi=300)
    plt.show()

    # Prepare data for Heatmap
    sample_idx = len(reaction_data) // 2
    token_ids = input_ids[sample_idx].cpu().numpy()
    p_weights = pooling_weights[sample_idx].cpu().numpy()
    s_attn = self_attn_matrix[sample_idx].cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    valid_indices = []
    valid_tokens = []
    for i, t in enumerate(tokens):
        if t not in ['<pad>', '<s>', '</s>']:
            valid_indices.append(i)
            valid_tokens.append(t)

    s_attn_valid = s_attn[np.ix_(valid_indices, valid_indices)]
    p_weights_valid = p_weights[valid_indices]
    if len(p_weights_valid) > 0:
        p_weights_norm = (p_weights_valid - p_weights_valid.min()) / (
                    p_weights_valid.max() - p_weights_valid.min() + 1e-9)
    else:
        p_weights_norm = p_weights_valid

    # =====================
    # Plot 2: Attention Intensity (Bar Plot)
    # =====================
    plt.figure(figsize=(12, 4))
    bars = plt.bar(range(len(valid_tokens)), p_weights_norm, color=plt.cm.viridis(p_weights_norm))

    # ★ Style control: 12pt, monospace, bold
    plt.xticks(range(len(valid_tokens)), valid_tokens, rotation=90, fontsize=12, fontname='monospace',
               fontweight='bold')

    plt.ylabel("Attention Intensity", fontsize=14, fontweight='bold')
    plt.xlabel("SMILES Tokens", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"pooling_attention_fold_{fold_idx}.png", dpi=300)
    plt.show()

    # =====================
    # Plot 3: Token-Token Cross-Attention (Heatmap) - Style Aligned
    # =====================
    plt.figure(figsize=(10, 8))

    # Use the same viridis color scheme
    ax = sns.heatmap(s_attn_valid,
                     xticklabels=valid_tokens,
                     yticklabels=valid_tokens,
                     cmap="viridis",
                     square=True,
                     cbar_kws={"shrink": 0.8})

    # ★ Style control: Strictly aligned with Bar Plot (12pt, monospace, bold)
    plt.xticks(rotation=90, fontsize=12, fontname='monospace', fontweight='bold')
    plt.yticks(rotation=0, fontsize=12, fontname='monospace', fontweight='bold')

    plt.xlabel("Key Token", fontsize=14, fontweight='bold')
    plt.ylabel("Query Token", fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()

    # =====================
    # Plot 4: Molecule Highlighting
    # =====================
    if RDKIT_AVAILABLE:
        mol = Chem.MolFromSmiles(target_smiles)
        if mol:
            print("Generating Molecule structure...")
            img = Draw.MolToImage(mol, size=(600, 300))
            plt.figure(figsize=(8, 4))
            plt.imshow(img)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"molecule_fold_{fold_idx}.png", dpi=300, bbox_inches='tight')
            plt.show()


# =====================
# 4. Main Execution Logic
# =====================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
special_tokens = [".", ">>", "I", "IIa", "IIb", "IIc", "IId"]
tokenizer.add_tokens(special_tokens)

df_full = pd.read_excel(FILE_PATH, header=None)
df_full.columns = ["smiles", "T", "lnk"]
df_full["T"] = pd.to_numeric(df_full["T"], errors="coerce")
df_full["lnk"] = pd.to_numeric(df_full["lnk"], errors="coerce")
df_full = df_full.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
df_full = df_full.sort_values(by=['smiles', 'T']).reset_index(drop=True)

scaler_T = StandardScaler()
scaler_y = StandardScaler()
scaler_T.fit(df_full[["T"]])
scaler_y.fit(df_full[["lnk"]].values.reshape(-1, 1))


print(f"Starting Consistent Visualization (Seed={SEED})...")

for fold_idx, smiles in BEST_REACTIONS:
    try:
        visualize_reaction(fold_idx, smiles, df_full, tokenizer, scaler_T, scaler_y)
    except Exception as e:
        print(f"Error visualizing fold {fold_idx}: {e}")
        import traceback

        traceback.print_exc()