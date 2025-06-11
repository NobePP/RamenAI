import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import time

# --- 1. 設定項目 ---

# ★ データセットがある親フォルダのパスをここに指定してください
base_path = r"your_path"

# 画像のサイズとバッチサイズ
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20 # 学習回数

# デバイスの設定 (GPUが利用可能ならGPU(cuda)を、なければCPUを使用)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用デバイス: {device}")


# --- 2. データの読み込みと準備 ---

# データの前処理と拡張（Augmentation）
# 訓練データ用：リサイズ、ランダム反転・回転、テンソル変換、正規化
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNetの平均と標準偏差
])

# テストデータ用：リサイズ、テンソル変換、正規化のみ
test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# フォルダのパスを結合
train_dir = os.path.join(base_path, 'train')
test_dir = os.path.join(base_path, 'test')

# ImageFolderを使ってデータセットを作成
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

# DataLoaderを作成
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# クラス名とクラス数を取得
class_names = train_dataset.classes
NUM_CLASSES = len(class_names)
print("クラス名:", class_names)


# --- 3. CNNモデルの構築 ---

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        # Kerasと同様の構成でCNNを定義
        self.features = nn.Sequential(
            # ブロック1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # ブロック2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # ブロック3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 画像サイズを元に、Flatten後のサイズを計算
        # IMG_SIZE=224 -> 224/2=112 -> 112/2=56 -> 56/2=28
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes) # 出力層。活性化関数(Softmax)は損失関数に含まれる
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# モデルのインスタンス化とデバイスへの転送
model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
print(model)


# --- 4. 損失関数とオプティマイザの設定 ---
criterion = nn.CrossEntropyLoss() # Softmaxとクロスエントロピー損失を内包
### 変更点：オプティマイザにweight_decayを追加 ###
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) # weight_decayを追加


# --- 5. モデルの学習と評価 ---

# 結果を記録するためのリスト
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

start_time = time.time()

for epoch in range(EPOCHS):
    # --- 訓練モード ---
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 勾配をリセット
        optimizer.zero_grad()

        # 順伝播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 逆伝播
        loss.backward()

        # パラメータ更新
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    epoch_train_loss = running_loss / len(train_loader)
    epoch_train_acc = 100 * correct_train / total_train
    history['train_loss'].append(epoch_train_loss)
    history['train_acc'].append(epoch_train_acc)

    # --- 評価モード ---
    model.eval()
    running_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad(): # 勾配計算を無効化
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    epoch_val_loss = running_loss / len(test_loader)
    epoch_val_acc = 100 * correct_val / total_val
    history['val_loss'].append(epoch_val_loss)
    history['val_acc'].append(epoch_val_acc)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | "
          f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")


end_time = time.time()
print(f"\n--- 学習完了 ---")
print(f"学習時間: {(end_time - start_time)/60:.2f} 分")
print(f"最終的なテストデータの正解率: {epoch_val_acc:.2f}%")


# --- 6. 結果の可視化 ---

# X軸の範囲をエポック数（1から開始）に合わせて生成
epochs_range = range(1, EPOCHS + 1)

plt.figure(figsize=(14, 6)) # 少し横長にすると見やすいです

# --- 正解率のプロット ---
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history['train_acc'], label='Training Accuracy')
plt.plot(epochs_range, history['val_acc'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')

### 追加・変更点 ###
# X軸の目盛りを0からエポック数まで10刻みで表示（例: 0, 10, 20）
# EPOCHSが20なら、[0, 10, 20] という目盛りが作成されます
plt.xticks(range(0, EPOCHS + 1, 1)) 
# グリッド線を表示
plt.grid(True, linestyle='--', alpha=0.6)
# 100%のラインに補助線を追加
plt.axhline(y=100, color='red', linestyle='--', linewidth=1) 
# Y軸の範囲を調整（必要に応じて変更）
#plt.ylim(min(history['val_acc']) - 5, 101)


# --- 損失のプロット ---
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history['train_loss'], label='Training Loss')
plt.plot(epochs_range, history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

### 追加・変更点 ###
# X軸の目盛りを10刻みで表示
plt.xticks(range(0, EPOCHS + 1, 1))
# グリッド線を表示
plt.grid(True, linestyle='--', alpha=0.6)


# グラフのレイアウトを自動調整
plt.tight_layout()
# グラフを表示
plt.show()
