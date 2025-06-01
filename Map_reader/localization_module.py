import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from env import generate_diverse_path, mutate_walls_nearby,GridEnv
from guidance_module import compute_policy_field, visualize_policy

# ------------------------------
# Localization 모델 (MLP)
# ------------------------------
class LocalizationGRUModel(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=128, num_layers=1, map_height=10, map_width=10):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, map_height * map_width)

    def forward(self, x, h=None):
        out, h = self.gru(x, h)       # out: (B, T, H)
        logits = self.fc(out)         # logits: (B, T, H×W)
        return logits, h


class LocalizationSequenceDataset(Dataset):
    def __init__(self, base_map, goal, reachable_starts, mutate_fn, env_class,
                 num_variants=50, traj_per_variant=10, max_steps=50):
        self.height, self.width = base_map.shape
        self.sequences = []

        for _ in tqdm(range(num_variants), desc="Generating variants"):
            mutated_map, reachable_starts= mutate_fn(base_map, goal)
            for _ in range(traj_per_variant):
                env = env_class(mutated_map, goal, reachable_starts)
                obs = env.reset()
                done = False
                traj = []

                last_action = 0  # 초기 행동: '상'
                for _ in range(max_steps):
                    action = env.action_space.sample()
                    obs_flat = obs.flatten().astype(np.float32)
                    action_onehot = np.eye(4)[last_action]  # (4,)
                    input_vec = np.concatenate([action_onehot, obs_flat])  # shape: (13,)
                    y, x = env.agent_pos
                    label = y * self.width + x
                    traj.append((input_vec, label))
                    obs, _, done, _ = env.step(action)
                    last_action = action
                    if done:
                        break

                if len(traj) > 0:
                    seq_input, seq_label = zip(*traj)
                    self.sequences.append((
                        torch.tensor(seq_input, dtype=torch.float32),   # shape: (T, 13)
                        torch.tensor(seq_label, dtype=torch.long)       # shape: (T,)
                    ))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]  # returns (input_seq, label_seq)
    
def collate_fn(batch):
    x_seqs, y_seqs = zip(*batch)
    x_padded = nn.utils.rnn.pad_sequence(x_seqs, batch_first=True)
    y_padded = nn.utils.rnn.pad_sequence(y_seqs, batch_first=True, padding_value=-100)
    return x_padded, y_padded

# ------------------------------
# 학습 함수
# ------------------------------
def train_localization_gru(model, dataset, batch_size=8, epochs=10, lr=1e-3, device='cpu'):
    print(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)  # 패딩된 위치 무시

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)  # (B, T, 13)
            y_batch = y_batch.to(device)  # (B, T)

            optimizer.zero_grad()
            logits, _ = model(x_batch)    # (B, T, H×W)
            loss = criterion(logits.view(-1, logits.size(-1)), y_batch.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")
    return model

def test_localization_gru(model, dataset, batch_size=8, device='cpu'):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    model.eval()
    model = model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)       # (B, T, 13)
            y_batch = y_batch.to(device)       # (B, T)
            logits, _ = model(x_batch)         # (B, T, H*W)
            preds = logits.argmax(dim=-1)      # (B, T)

            mask = y_batch != -100
            correct += (preds[mask] == y_batch[mask]).sum().item()
            total += mask.sum().item()

    accuracy = correct / total if total > 0 else 0.0
    print(f"🔍 Test Accuracy: {accuracy*100:.2f}%")
    return accuracy

def visualize_localization_heatmap(model, dataset, map_height, map_width, device='cpu'):
    model.eval()
    model.to(device)

    error_map = np.zeros((map_height, map_width), dtype=np.int32)
    total_map = np.zeros((map_height, map_width), dtype=np.int32)

    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    with torch.no_grad():
        for x_seq, y_seq in loader:
            x_seq = x_seq.to(device)
            y_seq = y_seq.to(device)
            logits, _ = model(x_seq)  # (1, T, H*W)
            preds = logits.argmax(dim=-1).squeeze(0)  # (T,)
            labels = y_seq.squeeze(0)  # (T,)

            for pred, label in zip(preds, labels):
                if label == -100:
                    continue
                true_y, true_x = divmod(label.item(), map_width)
                total_map[true_y, true_x] += 1

                if pred.item() != label.item():
                    error_map[true_y, true_x] += 1

    # 오답률 계산
    error_rate = np.divide(error_map, total_map, out=np.zeros_like(error_map, dtype=float), where=total_map!=0)

    # 시각화
    plt.figure(figsize=(6, 5))
    plt.imshow(error_rate, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Error Rate')
    plt.title('Localization Error Heatmap')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().invert_yaxis()
    plt.show()


def train_localization_module(grid, goal, reachable_starts, visualize = True):
    
    dataset = LocalizationSequenceDataset(
        base_map=grid,
        goal=goal,
        reachable_starts=reachable_starts,
        mutate_fn=mutate_walls_nearby,
        env_class=GridEnv,
        num_variants=100,         # 더 줄이거나 늘릴 수 있음
        traj_per_variant=100,
        max_steps=30
    )

    # 모델 정의
    model = LocalizationGRUModel(input_dim=13, hidden_dim=128, map_height=10, map_width=10)

    # 학습
    trained_model = train_localization_gru(
        model,
        dataset,
        batch_size=8,
        epochs=20,
        lr=1e-3,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    test_dataset = LocalizationSequenceDataset(
        base_map=grid,
        goal=goal,
        reachable_starts=reachable_starts,
        mutate_fn=mutate_walls_nearby,
        env_class=GridEnv,
        num_variants=1,         # 더 줄이거나 늘릴 수 있음
        traj_per_variant=1000,
        max_steps=30
    )

    # 정확도 평가
    test_localization_gru(trained_model, test_dataset, batch_size=8, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    if visualize:
        visualize_localization_heatmap(
            model=trained_model,
            dataset=test_dataset,
            map_height=10,
            map_width=10,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    
    return trained_model

def main():
    grid, goal, reachable_starts = generate_diverse_path(height=10, width=10, wall_prob=0.3)
    trained_model = train_localization_module(grid, goal, reachable_starts, visualize= True)

if __name__ == "__main__":
    main()


