import os
import glob

# Mapping từ ID cũ (Roboflow) sang ID mới (Chuẩn hệ thống)
# Cũ: 0=H_landing, 1=circle, 2=semicircle
# Mới: 0=circle, 1=semicircle, 2=H_landing
ID_MAP = {
    "0": "2",  # H_landing chuyển thành 2
    "1": "0",  # circle chuyển thành 0
    "2": "1"   # semicircle chuyển thành 1
}

def remap_labels(label_dir):
    txt_files = glob.glob(os.path.join(label_dir, "*.txt"))
    
    for txt_file in txt_files:
        with open(txt_file, 'r') as file:
            lines = file.readlines()
            
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
                
            old_id = parts[0]
            if old_id in ID_MAP:
                # Thay thế class ID ở đầu dòng
                parts[0] = ID_MAP[old_id]
                new_lines.append(" ".join(parts) + "\n")
            else:
                new_lines.append(line)
                
        with open(txt_file, 'w') as file:
            file.writelines(new_lines)
            
    print(f"Đã xử lý xong {len(txt_files)} file trong {label_dir}")

# Chạy script cho cả 3 tập
for split in ["train/labels", "valid/labels", "test/labels"]:
    target_dir = split  # <--- Thay đổi ở dòng này, bỏ os.path.join("..", split)
    if os.path.exists(target_dir):
        remap_labels(target_dir)
    else:
        print(f"Không tìm thấy thư mục: {target_dir}")

print("Hoàn tất map lại nhãn! Bây giờ bạn có thể bắt đầu train YOLO11-seg.")