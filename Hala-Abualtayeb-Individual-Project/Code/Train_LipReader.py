import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
from torch.cuda.amp import autocast, GradScaler
import editdistance

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = os.getcwd()
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
TOKENIZER_PATH = os.path.join(MODEL_DIR, "char_tokenizer.pkl")

FRAME_HEIGHT = 64
FRAME_WIDTH = 64
MAX_FRAMES = 75
CHANNELS = 3
BATCH_SIZE = 8
ACCUMULATION_STEPS = 2
EPOCHS = 100
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

HIDDEN_DIM = 512
NUM_LSTM_LAYERS = 3
DROPOUT = 0.4
LABEL_SMOOTHING = 0.05
PATIENCE_LIMIT = 30

def pad_video(seq, max_frames=MAX_FRAMES):
    out = np.zeros((max_frames, FRAME_HEIGHT, FRAME_WIDTH, CHANNELS), dtype=np.float32)
    length = min(len(seq), max_frames)
    out[:length] = seq[:length]
    if length < max_frames and length > 0:
        for i in range(length, max_frames):
            out[i] = seq[-1]
    return out

def load_all_manifest_from_folders():

    file_pairs = []
    processed_folders = [
        "/home/ubuntu/DL/DL Project/Project/processed_data",
        "/home/ubuntu/DL/DL Project/processed_data"
    ]

    for base_folder in processed_folders:
        npy_files = sorted([f for f in os.listdir(base_folder) if f.endswith("_frames.npy")])
        txt_files = sorted([f for f in os.listdir(base_folder) if f.endswith("_text.txt")])

        for npy_file, txt_file in zip(npy_files, txt_files):
            pair = (os.path.join(base_folder, npy_file), os.path.join(base_folder, txt_file))
            file_pairs.append(pair)

    print(f"Total file pairs loaded: {len(file_pairs)}")
    return file_pairs

class VideoTextDataset(Dataset):
    def __init__(self, file_pairs, tokenizer=None, training=True):
        self.file_pairs = file_pairs
        self.tokenizer = tokenizer
        if len(self.file_pairs) == 0:
            raise RuntimeError("No valid video-text pairs found!")

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        video_path, text_path = self.file_pairs[idx]
        video = np.load(video_path)
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read().strip().lower()
        video = pad_video(video)
        video = torch.tensor(video, dtype=torch.float32).permute(0, 3, 1, 2)
        label = None
        if self.tokenizer:
            encode_text = lambda txt: [self.tokenizer.get(c, 0) for c in txt]
            label = torch.tensor(encode_text(text), dtype=torch.long)
        return video, label, text

def collate_fn(batch):
    videos, labels, texts = zip(*batch)
    videos = torch.stack(videos)
    targets = torch.cat(labels)
    target_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    input_lengths = torch.full((len(videos),), videos.shape[1], dtype=torch.long)
    return videos, targets, input_lengths, target_lengths, texts

def create_char_tokenizer(texts):
    chars = sorted(list(set("".join(texts))))
    mapping = {"<BLANK>": 0}
    for i, c in enumerate(chars, start=1):
        mapping[c] = i
    return mapping

class AttentionModule(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, lstm_output):
        weights = torch.softmax(self.attention(lstm_output), dim=1)
        return lstm_output * weights + lstm_output, weights

class lipreader(nn.Module):
    def __init__(self, num_chars, hidden_dim=HIDDEN_DIM, num_lstm_layers=NUM_LSTM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(CHANNELS, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout2d(0.2)
        )
        cnn_feat = 128 * (FRAME_HEIGHT // 8) * (FRAME_WIDTH // 8)
        self.rnn = nn.LSTM(cnn_feat, hidden_dim, num_lstm_layers, batch_first=True,
                           bidirectional=True, dropout=dropout if num_lstm_layers > 1 else 0)
        self.attention = AttentionModule(hidden_dim)
        self.fc = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Dropout(dropout),
                                nn.Linear(hidden_dim * 2, num_chars))

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.reshape(B, T, -1)
        out, _ = self.rnn(x)
        out, _ = self.attention(out)
        out = self.fc(out)
        return out.permute(1, 0, 2)  # T,B,C

class CTCLossWithLabelSmoothing(nn.Module):
    def __init__(self, blank=0, smoothing=LABEL_SMOOTHING):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank, zero_infinity=True)
        self.smoothing = smoothing

    def forward(self, logits, targets, input_lengths, target_lengths):
        logits = logits.float()
        loss = self.ctc_loss(logits, targets, input_lengths, target_lengths)
        if self.smoothing > 0:
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)
            smooth_loss = -log_probs.mean()
            loss = (1 - self.smoothing) * loss + self.smoothing * smooth_loss
        return loss

def decode_predictions(logits, tokenizer):
    idx_to_char = {v: k for k, v in tokenizer.items()}
    _, max_indices = torch.max(logits, dim=2)
    decoded = []
    for seq in max_indices.permute(1,0):
        chars, prev = [], None
        for idx in seq:
            idx = idx.item()
            if idx != 0 and idx != prev:
                chars.append(idx_to_char.get(idx,'?'))
            prev = idx
        decoded.append(''.join(chars))
    return decoded

def calculate_cer(preds, refs):
    total_distance = sum(editdistance.eval(p,r) for p,r in zip(preds,refs))
    total_length = sum(len(r) for r in refs)
    return total_distance / max(total_length,1)

def calculate_wer(preds, refs):
    total_distance = 0
    total_length = 0
    for p,r in zip(preds,refs):
        p_words = p.split()
        r_words = r.split()
        total_distance += editdistance.eval(p_words,r_words)
        total_length += len(r_words)
    return total_distance / max(total_length,1)

def train():
    print(f"Device: {DEVICE}, Batch Size: {BATCH_SIZE}, Accumulation Steps: {ACCUMULATION_STEPS}")

    file_pairs = load_all_manifest_from_folders()
    if len(file_pairs) == 0:
        raise RuntimeError("No data found! Please check your processed_data folders.")

    all_texts = []
    for _, text_path in file_pairs:
        with open(text_path, "r", encoding="utf-8") as f:
            all_texts.append(f.read().strip().lower())
    tokenizer = create_char_tokenizer(all_texts)
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"Vocabulary size: {len(tokenizer)}")

    pairs_tmp, pairs_test = train_test_split(file_pairs, test_size=0.15, random_state=42)
    pairs_train, pairs_val = train_test_split(pairs_tmp, test_size=0.1765, random_state=42)
    print(f"Training: {len(pairs_train)}, Validation: {len(pairs_val)}, Test: {len(pairs_test)}")

    test_pkl_path = os.path.join(MODEL_DIR, "test_pairs.pkl")
    with open(test_pkl_path, "wb") as f:
        pickle.dump(pairs_test, f)
    print(f"Saved test split to {test_pkl_path}")

    train_ds = VideoTextDataset(pairs_train, tokenizer, training=True)
    val_ds = VideoTextDataset(pairs_val, tokenizer, training=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=4, pin_memory=torch.cuda.is_available())

    model = lipreader(len(tokenizer)).to(DEVICE)
    criterion = CTCLossWithLabelSmoothing()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler() if torch.cuda.is_available() else None

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for videos, targets, input_lengths, target_lengths, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]"):
            videos, targets = videos.to(DEVICE), targets.to(DEVICE)
            logits = model(videos)
            loss = criterion(logits, targets, input_lengths.to(DEVICE), target_lengths.to(DEVICE))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        all_preds, all_refs = [], []
        with torch.no_grad():
            for videos, targets, input_lengths, target_lengths, texts in val_loader:
                videos, targets = videos.to(DEVICE), targets.to(DEVICE)
                logits = model(videos)
                loss = criterion(logits, targets, input_lengths.to(DEVICE), target_lengths.to(DEVICE))
                val_loss += loss.item()
                preds = decode_predictions(logits, tokenizer)
                all_preds.extend(preds)
                all_refs.extend(texts)
        avg_val_loss = val_loss / len(val_loader)
        cer = calculate_cer(all_preds, all_refs)
        wer = calculate_wer(all_preds, all_refs)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | CER: {cer:.4f} | WER: {wer:.4f}")

        rand_idx = random.randint(0, len(all_preds)-1)
        print(f"Sample prediction: Pred: {all_preds[rand_idx]} | Ref: {all_refs[rand_idx]}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE_LIMIT:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print("Training complete. Best model saved at", os.path.join(MODEL_DIR, "best_model.pth"))

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        import traceback
        print("Training failed:", e)
        traceback.print_exc()
