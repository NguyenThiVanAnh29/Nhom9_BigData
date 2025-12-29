import os
import subprocess

# ===================== CONFIG =====================
dataset_dir = "./datasets/PIE-bench"  # đường dẫn đến dataset
model = "instructpix2pix"             # có thể đổi sang magicbrush/instructdiffusion/mgie/ultraedit
num_random_candidates = 10
select_one_seed = True
output_dir = "./outputs"               # folder lưu kết quả

# Nếu dataset có instruction riêng cho từng ảnh, sửa ở đây:
# key = tên file ảnh, value = instruction
instruction_dict = {
    # Ví dụ:
    # "img1.png": "Replace the cat with a bear",
    # "img2.png": "Change the dog to a tiger",
}

# Nếu ảnh không có instruction riêng, sẽ dùng instruction mặc định này
default_instruction = "Replace the main object according to instruction"

# ===================== CREATE OUTPUT DIR =====================
os.makedirs(output_dir, exist_ok=True)

# ===================== LOOP QUA TẤT CẢ ẢNH =====================
for img_file in os.listdir(dataset_dir):
    if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
        input_path = os.path.join(dataset_dir, img_file)
        # lấy instruction cho từng ảnh
        instruction = instruction_dict.get(img_file, default_instruction)
        
        # tạo lệnh gọi inference.py
        cmd = [
            "python", "inference.py",
            "--run_type", "run_single_image",
            "--input_path", input_path,
            "--instruction", instruction,
            "--model", model,
            "--output_dir", output_dir
        ]
        if select_one_seed:
            cmd.append("--select_one_seed")
        if num_random_candidates > 0:
            cmd += ["--num_random_candidates", str(num_random_candidates)]
        
        print("\nRunning:", " ".join(cmd))
        subprocess.run(cmd)

print("\n✅ Hoàn tất chạy tất cả ảnh trong dataset!")
print(f"Tất cả outputs được lưu ở: {output_dir}")
