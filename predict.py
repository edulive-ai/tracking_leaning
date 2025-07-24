# import pytorch_lightning as pl
# import torch
# checkpoint = torch.load('models/epoch=4-step=503725.ckpt')
# print(checkpoint)
#-----------------------------------------------------------------#
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from config import Config
from saint import PlusSAINTModule

# Hàm tạo dữ liệu kiểm tra giả lập với user_id
def create_mock_test_data(num_users=5, max_interactions=8, seq_len=10, max_exercise_id=100, max_category_id=20):
    # Tạo dữ liệu giả lập
    interactions = []
    for user_id in range(1, num_users + 1):
        num_inter = np.random.randint(3, max_interactions + 1)
        for _ in range(num_inter):
            interactions.append({
                "user_id": user_id,
                "exercise_id": np.random.randint(1, max_exercise_id + 1),
                "category_id": np.random.randint(0, max_category_id),
                "response_time": np.random.uniform(5.0, 60.0),
                "response": np.random.randint(0, 2)
            })
    
    # Lọc theo user_id
    input_data_list = []
    labels_list = []
    selected_user_ids = np.random.choice(range(1, num_users + 1), size=2, replace=False)
    
    for user_id in selected_user_ids:
        user_interactions = [inter for inter in interactions if inter["user_id"] == user_id]
        user_interactions = user_interactions[:seq_len]
        
        input_ids = torch.zeros(seq_len, dtype=torch.long)
        input_cat = torch.zeros(seq_len, dtype=torch.long)
        input_rtime = torch.zeros(seq_len, dtype=torch.float)
        labels = torch.full((seq_len,), 2, dtype=torch.long)
        
        for i, inter in enumerate(user_interactions):
            input_ids[i] = inter["exercise_id"]
            input_cat[i] = inter["category_id"]
            input_rtime[i] = inter["response_time"]
            labels[i] = inter["response"]
        
        input_data_list.append({
            "input_ids": input_ids.unsqueeze(0),
            "input_cat": input_cat.unsqueeze(0),
            "input_rtime": input_rtime.unsqueeze(0)
        })
        labels_list.append(labels.unsqueeze(0))
    
    input_data = {
        "input_ids": torch.cat([d["input_ids"] for d in input_data_list], dim=0).to(Config.device),
        "input_cat": torch.cat([d["input_cat"] for d in input_data_list], dim=0).to(Config.device),
        "input_rtime": torch.cat([d["input_rtime"] for d in input_data_list], dim=0).to(Config.device)
    }
    labels = torch.cat(labels_list, dim=0).to(Config.device)
    
    return input_data, labels, selected_user_ids

# Hàm dự đoán và đánh giá
def predict_and_evaluate(checkpoint_path):
    # Khởi tạo mô hình
    model = PlusSAINTModule()
    
    # Load state_dict từ checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=Config.device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.to(Config.device)

    # Tạo dữ liệu kiểm tra giả lập
    input_data, labels, user_ids = create_mock_test_data(
        num_users=5,
        max_interactions=8,
        seq_len=Config.MAX_SEQ,
        max_exercise_id=Config.TOTAL_EXE,
        max_category_id=Config.TOTAL_CAT
    )

    # Dự đoán
    with torch.no_grad():
        output = model(input_data, labels)
        target_mask = (input_data["input_ids"] != 0)
        preds = torch.sigmoid(output)
        preds = torch.masked_select(preds, target_mask)
        true_labels = torch.masked_select(labels, target_mask)

    # Tính AUC
    preds_np = preds.cpu().numpy()
    labels_np = true_labels.cpu().numpy()
    auc = roc_auc_score(labels_np, preds_np)
    
    print("User IDs:", user_ids)
    print("Predictions:", preds_np)
    print("True labels:", labels_np)
    print(f"Test AUC: {auc:.4f}")
    
    return auc, preds_np, labels_np, user_ids

if __name__ == "__main__":
    # Đường dẫn đến checkpoint
    checkpoint_path = "/home/batien/Desktop/SAINT_plus-Knowledge-Tracing-/tracking_leaning/models/epoch=4-step=503725.ckpt"
    
    # Dự đoán và đánh giá
    auc, predictions, true_labels, user_ids = predict_and_evaluate(checkpoint_path)