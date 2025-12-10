import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn

PROJECT_ROOT = os.getcwd()
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_FRAMES = 75
FRAME_HEIGHT = 64
FRAME_WIDTH = 64
CHANNELS = 3
BATCH_SIZE = 1

def pad_video(seq):
    out = np.zeros((MAX_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, CHANNELS), dtype=np.float32)
    length = min(len(seq), MAX_FRAMES)
    out[:length] = seq[:length]
    return out


def decode_predictions(logits, tokenizer):
    idx_to_char = {v: k for k, v in tokenizer.items()}
    _, max_indices = torch.max(logits, dim=2)
    decoded = []
    for sequence in max_indices.permute(1, 0):
        chars = []
        prev_idx = None
        for idx in sequence:
            idx = idx.item()
            if idx != 0 and idx != prev_idx:
                chars.append(idx_to_char.get(idx, '?'))
            prev_idx = idx
        decoded.append(''.join(chars))
    return decoded


def calculate_cer(predictions, references):
    import editdistance
    total_distance = 0
    total_length = 0
    for pred, ref in zip(predictions, references):
        dist = editdistance.eval(pred, ref)
        total_distance += dist
        total_length += len(ref)
    return total_distance / max(total_length, 1)


def calculate_wer(predictions, references):
    import editdistance
    total_distance = 0
    total_length = 0
    for pred, ref in zip(predictions, references):
        pred_words = pred.split()
        ref_words = ref.split()
        dist = editdistance.eval(pred_words, ref_words)
        total_distance += dist
        total_length += len(ref_words)
    return total_distance / max(total_length, 1)


def calculate_accuracy(predictions, references):
    correct = 0
    total = 0
    for pred, ref in zip(predictions, references):
        min_len = min(len(pred), len(ref))
        for i in range(min_len):
            if pred[i] == ref[i]:
                correct += 1
        total += max(len(pred), len(ref))
    return correct / max(total, 1)

class VideoOnlyDataset(Dataset):
    def __init__(self, file_pairs):
        self.file_pairs = file_pairs

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        frames_path, text_path = self.file_pairs[idx]
        video = np.load(frames_path)
        video = pad_video(video)
        video = torch.tensor(video, dtype=torch.float32).permute(0, 3, 1, 2)  # T, C, H, W
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        return video, text


def collate_fn(batch):
    videos, texts = zip(*batch)
    videos = torch.stack(videos)
    return videos, texts


class AttentionModule(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, lstm_output):
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        weighted = lstm_output * attention_weights
        return weighted + lstm_output


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
        return out.permute(1, 0, 2)



def test():
    tokenizer_path = os.path.join(MODEL_DIR, "char_tokenizer2.pkl")
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    test_split_path = os.path.join(MODEL_DIR,"test_split.pkl")
    with open(test_split_path, "rb") as f:
        test_pairs = pickle.load(f)

    test_ds = VideoOnlyDataset(test_pairs)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model_path = os.path.join(MODEL_DIR, "best_model.pth")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model = ImprovedCNNLSTMWithAttention(len(tokenizer))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    all_preds, all_refs = [], []

    with torch.no_grad():
        for videos, texts in tqdm(test_loader, desc="Testing"):
            videos = videos.to(DEVICE)
            logits = model(videos)

            preds = decode_predictions(logits, tokenizer)
            all_preds.extend(preds)
            all_refs.extend(texts)

    try:
        cer = calculate_cer(all_preds, all_refs)
        wer = calculate_wer(all_preds, all_refs)
        acc = calculate_accuracy(all_preds, all_refs)
    except ImportError:
        cer = wer = acc = None

    print(f"\nTest results on {len(all_preds)} samples")
    print(f"Example predictions vs references:")
    for p, r in zip(all_preds[:5], all_refs[:5]):
        print(f"  Pred: {p} | Ref: {r}")

    if cer is not None:
        print(f"\nMetrics:\n  CER: {cer:.4f}\n  WER: {wer:.4f}\n  Accuracy: {acc:.4f}")

if __name__ == "__main__":
    test()
