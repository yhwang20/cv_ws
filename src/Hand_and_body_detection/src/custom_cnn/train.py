import torch
from torch.utils.data import DataLoader
from resblock_class import CNN_plus_Transformer  # your hybrid model
from coco_dataset import COCOKP
import os
device = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset

curr_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(curr_dir, "..", "..", "..", "..", "..", "data")
train_img_set = os.path.join(data_dir, "train2017")
train_ann_file = os.path.join(data_dir, "annotations", "person_keypoints_train2017.json")
val_img_set = os.path.join(data_dir, "val2017")
val_ann_file = os.path.join(data_dir, "annotations", "person_keypoints_val2017.json")

train_ds = COCOKP(train_img_set, train_ann_file)
val_ds = COCOKP(val_img_set, val_ann_file)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=8)

# Model
model = CNN_plus_Transformer(num_classes=17).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()

# Training loop
for epoch in range(20):
    model.train()
    train_loss = 0
    for imgs, heatmaps in train_loader:
        imgs = imgs.to(device)
        heatmaps = heatmaps.to(device)

        preds = model(imgs)
        loss = criterion(preds, heatmaps)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f"Epoch {epoch}: Train Loss={train_loss/len(train_loader):.4f}")

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, heatmaps in val_loader:
            imgs = imgs.to(device)
            heatmaps = heatmaps.to(device)
            preds = model(imgs)
            val_loss += criterion(preds, heatmaps).item()
    print(f"Epoch {epoch}: Val Loss={val_loss/len(val_loader):.4f}")

torch.save(model.state_dict(), "cnn_mobilevit_pose.pth")
