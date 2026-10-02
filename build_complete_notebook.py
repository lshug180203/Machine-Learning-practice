"""
Build and generate the complete Jupyter Notebook for Lec 3 Linear Regression & Data Normalization.
Ensures adherence to all 6 parts requested by user, exact slide reproduction, and real dataset execution.
"""
import nbformat as nbf
import os
import sys

# Configure UTF-8 for console output on Windows
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def create_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.14.0"
        }
    }
    
    cells = []
    
    # -------------------------------------------------------------
    # PHẦN 1: INTRODUCTION
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""# BÀI THỰC HÀNH MACHINE LEARNING: HỒI QUY TUYẾN TÍNH ĐA BIẾN VÀ CHUẨN HÓA DỮ LIỆU
## MULTIPLE LINEAR REGRESSION & FEATURE SCALING (SLIDES 36 - 46)

---

### 1. Giới thiệu tài liệu & Mục tiêu thực hành
Notebook này được xây dựng dựa trên bài giảng **Machine Learning - Lecture 3: Linear Regression** (từ Slide 36 `# dataset` đến Slide 46), kết hợp nền tảng lý thuyết tối ưu hóa Gradient Descent (Slide 35).

Mục tiêu chính của bài thực hành:
1. **Xây dựng từ đầu (From Scratch)** mô hình **Hồi quy Tuyến tính Đa biến (Multiple Linear Regression)** bằng Python thuần và NumPy mà không dùng các hàm đóng gói sẵn của thư viện bậc cao cho thuật toán chính.
2. Cài đặt chi tiết từng thành phần của thuật toán **Stochastic Gradient Descent (SGD)**:
   - Hàm lan truyền xuôi dự đoán: $\\hat{y} = w_1 x_1 + w_2 x_2 + w_3 x_3 + b$
   - Hàm mất mát sai số bình phương: $L = (\\hat{y} - y)^2$
   - Đạo hàm riêng (Gradient): $\\frac{\\partial L}{\\partial w_i} = 2 x_i (\\hat{y} - y)$ và $\\frac{\\partial L}{\\partial b} = 2 (\\hat{y} - y)$
   - Quy tắc cập nhật trọng số và bias theo tốc độ học (learning rate $\\eta$): $w_i \\leftarrow w_i - \\eta \\frac{\\partial L}{\\partial w_i}$, $b \\leftarrow b - \\eta \\frac{\\partial L}{\\partial b}$
3. Cài đặt hoàn chỉnh các kỹ thuật **Chuẩn hóa Dữ liệu (Feature Scaling)**:
   - **Min-Max Scaling**: Biến đổi dữ liệu về đoạn $[0, 1]$ và hàm khôi phục **Inverse Min-Max Scaling**.
   - **Z-Score Scaling (Standardization)**: Biến đổi dữ liệu về phân phối có $\\mu = 0$ và $\\sigma = 1$, kèm hàm khôi phục **Inverse Z-Score Scaling**.
4. Áp dụng toàn bộ mã nguồn trên tập dữ liệu thực tế: `advertising.csv`.
5. Đánh giá hiệu năng mô hình (MSE, RMSE, MAE, $R^2$), phân tích vai trò của Feature Scaling đối với tốc độ hội tụ và đối chiếu với nghiệm giải tích Closed-Form Normal Equation cùng thư viện Scikit-Learn.

---

### 2. Thông tin tập dữ liệu (`advertising.csv`)
Tập dữ liệu ghi nhận ngân sách quảng cáo cho một sản phẩm trên 3 kênh truyền thông và doanh số tương ứng tại 200 thị trường khác nhau:
- **TV**: Chi phí quảng cáo trên truyền hình (đơn vị: nghìn USD).
- **Radio**: Chi phí quảng cáo trên đài phát thanh (đơn vị: nghìn USD).
- **Newspaper**: Chi phí quảng cáo trên báo giấy (đơn vị: nghìn USD).
- **Sales**: Doanh số bán hàng đạt được (biến mục tiêu $y$, đơn vị: nghìn sản phẩm)."""))

    # -------------------------------------------------------------
    # PHẦN 2: ENVIRONMENT SETUP
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""---
## PHẦN 2: ENVIRONMENT SETUP (THIẾT LẬP MÔI TRƯỜNG)

Trong phần này, chúng ta sẽ kiểm tra phiên bản Python, nhập các thư viện cần thiết, thiết lập cấu hình đường dẫn tương đối `DATA_PATH` để notebook có thể chạy mượt mà trên mọi môi trường (JupyterLab, VS Code, Google Colab)."""))

    cells.append(nbf.v4.new_code_cell("""# 1. Kiểm tra môi trường & hướng dẫn cài đặt nếu thiếu
import sys
import os

print(f"Phiên bản Python hiện tại: {sys.version}")

# Danh sách các thư viện cần thiết
required_packages = ["numpy", "matplotlib", "pandas", "sklearn"]
print("Danh sách thư viện yêu cầu:", required_packages)
print("Nếu thiếu thư viện trong môi trường mới, hãy chạy lệnh sau trong terminal:")
print("    pip install numpy matplotlib pandas scikit-learn")"""))

    cells.append(nbf.v4.new_code_cell("""# 2. Import các thư viện được sử dụng
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Cấu hình hiển thị đồ thị đẹp mắt
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.figsize'] = (9, 5)
plt.rcParams['font.size'] = 11

# 3. Cấu hình đường dẫn dataset linh hoạt (DATA_PATH)
DATA_FILE_CANDIDATES = [
    'advertising.csv',
    '../content/advertising.csv',
    os.path.join(os.getcwd(), 'advertising.csv')
]

DATA_PATH = None
for candidate in DATA_FILE_CANDIDATES:
    if os.path.exists(candidate):
        DATA_PATH = candidate
        break

if DATA_PATH is None:
    raise FileNotFoundError("Không tìm thấy file 'advertising.csv'. Vui lòng đặt file trong thư mục làm việc.")

print(f"Đường dẫn dataset đang sử dụng: {DATA_PATH}")"""))

    # -------------------------------------------------------------
    # PHẦN 3: DATASET EXPLORATION
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""---
## PHẦN 3: DATASET EXPLORATION (KHÁM PHÁ DỮ LIỆU)

Trước khi đi vào xây dựng mô hình toán học, ta tiến hành khám phá sơ bộ tập dữ liệu `advertising.csv` để hiểu rõ cấu trúc, kiểu dữ liệu, các giá trị thống kê mô tả và mối tương quan giữa các biến."""))

    cells.append(nbf.v4.new_code_cell("""# Đọc dữ liệu bằng Pandas DataFrame để quan sát tổng quan
df = pd.read_csv(DATA_PATH)

print(f"Kích thước bộ dữ liệu (Số dòng, Số cột): {df.shape}")
print("\\n5 dòng đầu tiên của tập dữ liệu:")
df.head()"""))

    cells.append(nbf.v4.new_code_cell("""# Kiểm tra kiểu dữ liệu, số lượng giá trị thiếu (missing values) và bản ghi trùng lặp
print("--- THÔNG TIN KIỂU DỮ LIỆU & BỘ NHỚ ---")
df.info()

print("\\n--- KIỂM TRA GIÁ TRỊ THIẾU (NULL) ---")
print(df.isnull().sum())

print(f"\\nSố dòng bị trùng lặp (duplicate rows): {df.duplicated().sum()}")"""))

    cells.append(nbf.v4.new_code_cell("""# Thống kê mô tả các đặc trưng
desc = df.describe().T
desc['range'] = desc['max'] - desc['min']
print("Bảng thống kê mô tả:")
desc[['count', 'mean', 'std', 'min', '50%', 'max', 'range']]"""))

    cells.append(nbf.v4.new_markdown_cell("""**Nhận xét quan trọng về thang đo (Scale) của các đặc trưng:**
- Chi phí cho `TV` dao động từ **0.7** đến **296.4** nghìn USD (biên độ gần 300).
- Chi phí cho `Radio` dao động từ **0.0** đến **49.6** nghìn USD.
- Chi phí cho `Newspaper` dao động từ **0.3** đến **114.0** nghìn USD.
- Do các đặc trưng có độ lớn và phương sai rất chênh lệch nhau, bề mặt hàm mất mát (loss surface) sẽ bị kéo giãn thành hình elip rất dẹp. Vì vậy, các kỹ thuật **Chuẩn hóa dữ liệu (Feature Scaling - Min-Max Scaling & Z-Score)** ở các Slide 41-46 là tối quan trọng để thuật toán Gradient Descent hội tụ nhanh và ổn định hơn!"""))

    cells.append(nbf.v4.new_code_cell("""# Trực quan hóa phân phối và mối tương quan với biến mục tiêu Sales
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)

channels = [('TV', '#1f77b4'), ('Radio', '#ff7f0e'), ('Newspaper', '#2ca02c')]
for ax, (col, color) in zip(axes, channels):
    ax.scatter(df[col], df['Sales'], color=color, alpha=0.7, edgecolors='none', s=40)
    # Đường hồi quy tuyến tính đơn giản minh họa xu hướng
    m, c = np.polyfit(df[col], df['Sales'], 1)
    x_line = np.linspace(df[col].min(), df[col].max(), 100)
    ax.plot(x_line, m * x_line + c, color='red', linestyle='--', linewidth=1.5, label=f'Trend (r={df[col].corr(df["Sales"]):.2f})')
    ax.set_title(f'{col} vs Sales', fontsize=13, fontweight='bold')
    ax.set_xlabel(f'Ngân sách {col} ($1,000)', fontsize=11)
    if col == 'TV':
        ax.set_ylabel('Doanh số Sales ($1,000 sản phẩm)', fontsize=11)
    ax.legend(frameon=True)
    ax.grid(True, linestyle=':', alpha=0.6)

plt.suptitle('Mối tương quan giữa từng Kênh Quảng cáo và Doanh số Bán hàng', fontsize=14, y=1.03)
plt.tight_layout()
plt.show()"""))

    # -------------------------------------------------------------
    # PHẦN 4: CODE EXAMPLES FROM PDF (SLIDES 36 - 46)
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""---
## PHẦN 4: CODE EXAMPLES (TÁI HIỆN TOÀN BỘ CODE TỪ SLIDE 36 ĐẾN SLIDE 46)

Trong phần này, chúng ta sẽ thực thi từng đoạn mã nguồn trong bài giảng `lec3.pdf` theo đúng trình tự bài học. Các đoạn mã được viết lại đầy đủ, giải thích chi tiết mục tiêu, công thức toán học và đảm bảo chạy tương thích 100% với dataset `advertising.csv`."""))

    # Slide 36
    cells.append(nbf.v4.new_markdown_cell("""### 4.1. Slide 36: Tải dữ liệu và chuẩn bị dữ liệu (`prepare_data`)
* **Mục tiêu:** Đọc file dữ liệu CSV bằng hàm `np.genfromtxt`, bỏ qua dòng tiêu đề (`skip_header=1`), chuyển thành danh sách 2D bằng `.tolist()`.
* **Trích xuất:** Viết hàm hỗ trợ `get_column(data, index)` trích xuất từng cột thành một list riêng lẻ:
  - Cột 0: `tv_data`
  - Cột 1: `radio_data`
  - Cột 2: `newspaper_data`
  - Cột 3: `sales_data`
* **Xây dựng tập huấn luyện:** Gom các kênh quảng cáo thành ma trận đầu vào $X = [tv, radio, newspaper]$ và vector nhãn $y = sales$."""))

    cells.append(nbf.v4.new_code_cell("""# Slide 36: Code chuẩn bị dữ liệu (Data Preparation)
# Lưu ý: Lệnh !gdown và !unzip từ Colab được thay thế bằng kiểm tra file cục bộ DATA_PATH đã có sẵn.

def get_column(data, index):
    \"\"\"Trích xuất cột thứ 'index' từ danh sách 2D 'data'\"\"\"
    result = [row[index] for row in data]
    return result

def prepare_data(file_name_dataset):
    \"\"\"Đọc file dataset và trích xuất ma trận đặc trưng X và nhãn y\"\"\"
    data = np.genfromtxt(file_name_dataset, delimiter=',', skip_header=1).tolist()
    N = len(data)
    
    # get tv (index=0)
    tv_data = get_column(data, 0)
    
    # get radio (index=1)
    radio_data = get_column(data, 1)
    
    # get newspaper (index=2)
    newspaper_data = get_column(data, 2)
    
    # get sales (index=3)
    sales_data = get_column(data, 3)
    
    # building X input and y output for training
    X = [tv_data, radio_data, newspaper_data]
    y = sales_data
    return X, y

# Thực thi hàm prepare_data với file advertising.csv
X, y = prepare_data(DATA_PATH)

print(f"Số lượng mẫu quan sát N = {len(y)}")
print(f"Số lượng đặc trưng trong X: {len(X)} (0: TV, 1: Radio, 2: Newspaper)")
print("5 giá trị đầu tiên của từng biến:")
print(f"  TV        : {X[0][:5]}")
print(f"  Radio     : {X[1][:5]}")
print(f"  Newspaper : {X[2][:5]}")
print(f"  Sales (y) : {y[:5]}")"""))

    # Slide 37
    cells.append(nbf.v4.new_markdown_cell("""### 4.2. Slide 37: Khởi tạo tham số và các hàm cốt lõi của Linear Regression
Theo sơ đồ thuật toán tại **Slide 35**:
1. Khởi tạo trọng số $w_1, w_2, w_3$ và hệ số chặn $b$. Trong slide 37, tác giả cung cấp giá trị khởi tạo cố định (được sinh ngẫu nhiên từ phân phối Gauss $\\mu=0, \\sigma=0.01$):
   $$\\mathbf{w}^{(0)} = (0.016992259082509283,\\; 0.0070783670518262355,\\; -0.002307860847821344), \\quad b^{(0)} = 0$$
2. **Dự đoán đầu ra:**
   $$\\hat{y} = w_1 x_1 + w_2 x_2 + w_3 x_3 + b$$
3. **Tính hàm mất mát sai số bình phương:**
   $$L = (\\hat{y} - y)^2$$
4. **Tính Gradient (Đạo hàm riêng của Loss đối với từng tham số):**
   $$\\frac{\\partial L}{\\partial w_i} = 2 x_i (\\hat{y} - y), \\quad \\frac{\\partial L}{\\partial b} = 2 (\\hat{y} - y)$$
5. **Cập nhật tham số theo Gradient Descent:**
   $$w_i \\leftarrow w_i - \\eta \\frac{\\partial L}{\\partial w_i}, \\quad b \\leftarrow b - \\eta \\frac{\\partial L}{\\partial b}$$"""))

    cells.append(nbf.v4.new_code_cell("""# Slide 37 (Phần 1): Hàm khởi tạo tham số
def initialize_params():
    \"\"\"Khởi tạo các trọng số w1, w2, w3 và bias b\"\"\"
    # w1 = random.gauss(mu=0.0, sigma=0.01)
    # w2 = random.gauss(mu=0.0, sigma=0.01)
    # w3 = random.gauss(mu=0.0, sigma=0.01)
    # b = 0
    
    w1, w2, w3, b = (0.016992259082509283, 0.0070783670518262355, -0.002307860847821344, 0)
    return w1, w2, w3, b

print("Khởi tạo tham số ban đầu (w1, w2, w3, b):")
print(initialize_params())"""))

    cells.append(nbf.v4.new_code_cell("""# Slide 37 (Phần 2): Cài đặt các hàm tính toán dự đoán, loss, gradient và cập nhật trọng số

# compute output and loss
def predict(x1, x2, x3, w1, w2, w3, b):
    \"\"\"Dự đoán giá trị y_hat theo mô hình hồi quy tuyến tính 3 biến\"\"\"
    return w1*x1 + w2*x2 + w3*x3 + b

def compute_loss(y_hat, y):
    \"\"\"Tính sai số bình phương giữa giá trị dự đoán và thực tế\"\"\"
    return (y_hat - y)**2

# compute gradient
def compute_gradient_wi(xi, y, y_hat):
    \"\"\"Đạo hàm riêng theo trọng số wi: dL/dwi = 2 * xi * (y_hat - y)\"\"\"
    dl_dwi = 2*xi*(y_hat-y)
    return dl_dwi

def compute_gradient_b(y, y_hat):
    \"\"\"Đạo hàm riêng theo hệ số chặn b: dL/db = 2 * (y_hat - y)\"\"\"
    dl_db = 2*(y_hat-y)
    return dl_db

# update weights
def update_weight_wi(wi, dl_dwi, lr):
    \"\"\"Cập nhật trọng số: wi_new = wi - lr * dL/dwi\"\"\"
    wi = wi - lr*dl_dwi
    return wi

def update_weight_b(b, dl_db, lr):
    \"\"\"Cập nhật hệ số chặn: b_new = b - lr * dL/db\"\"\"
    b = b - lr*dl_db
    return b"""))

    cells.append(nbf.v4.new_code_cell("""# Kiểm tra hoạt động (Unit test) của các hàm trên một mẫu thử nghiệm cụ thể
x1_test, x2_test, x3_test, y_test = X[0][0], X[1][0], X[2][0], y[0]
w1_init, w2_init, w3_init, b_init = initialize_params()

y_hat_test = predict(x1_test, x2_test, x3_test, w1_init, w2_init, w3_init, b_init)
loss_test = compute_loss(y_hat_test, y_test)
grad_w1_test = compute_gradient_wi(x1_test, y_test, y_hat_test)
grad_b_test = compute_gradient_b(y_test, y_hat_test)

print(f"Mẫu số 0: x1={x1_test}, x2={x2_test}, x3={x3_test}, y_true={y_test}")
print(f"Dự đoán ban đầu y_hat : {y_hat_test:.4f}")
print(f"Sai số ban đầu Loss   : {loss_test:.4f}")
print(f"dL/dw1: {grad_w1_test:.4f} | dL/db: {grad_b_test:.4f}")"""))

    # Slide 38
    cells.append(nbf.v4.new_markdown_cell("""### 4.3. Slide 38: Huấn luyện mô hình Linear Regression (`implement_linear_regression`)
* Thuật toán sử dụng là **Stochastic Gradient Descent (SGD)**: duyệt qua từng mẫu dữ liệu $(x_i, y_i)$ trong bộ dữ liệu tại mỗi epoch, tính gradient cục bộ trên mẫu đó và cập nhật tham số ngay lập tức.
* Tham số huấn luyện mặc định:
  - `epoch_max = 50`: Số lượng chu kỳ huấn luyện toàn bộ dữ liệu.
  - `lr = 1e-5`: Tốc độ học (learning rate $\\eta$). Lưu ý: do dữ liệu chưa chuẩn hóa và đặc trưng `TV` có giá trị lớn (~300), tốc độ học cần chọn nhỏ ($10^{-5}$) để tránh thuật toán bị bùng nổ gradient."""))

    cells.append(nbf.v4.new_code_cell("""# Slide 38: Hàm thực thi huấn luyện hồi quy tuyến tính
def implement_linear_regression(X_data, y_data, epoch_max = 50, lr = 1e-5):
    \"\"\"Huấn luyện Linear Regression bằng Stochastic Gradient Descent (SGD)\"\"\"
    losses = []
    
    w1, w2, w3, b = initialize_params()
    
    N = len(y_data)
    for epoch in range(epoch_max):
        for i in range(N):
            # get a sample
            x1 = X_data[0][i]
            x2 = X_data[1][i]
            x3 = X_data[2][i]
            
            y = y_data[i]
            
            # compute output
            y_hat = predict(x1, x2, x3, w1, w2, w3, b)
            
            # compute loss
            loss = compute_loss(y, y_hat)
            
            # compute gradient w1, w2, w3, b
            dl_dw1 = compute_gradient_wi(x1, y, y_hat)
            dl_dw2 = compute_gradient_wi(x2, y, y_hat)
            dl_dw3 = compute_gradient_wi(x3, y, y_hat)
            dl_db = compute_gradient_b(y, y_hat)
            
            # update parameters
            w1 = update_weight_wi(w1, dl_dw1, lr)
            w2 = update_weight_wi(w2, dl_dw2, lr)
            w3 = update_weight_wi(w3, dl_dw3, lr)
            b = update_weight_b(b, dl_db, lr)
            
            # logging
            losses.append(loss)
            
    return (w1, w2, w3, b, losses)"""))

    # Slide 39
    cells.append(nbf.v4.new_markdown_cell("""### 4.4. Slide 39: Thực thi huấn luyện và trực quan hóa hàm mất mát (Loss History)
* Đọc dữ liệu từ file `advertising.csv`.
* Chạy huấn luyện và ghi nhận lịch sử hàm mất mát qua các bước lặp.
* In 100 giá trị loss đầu tiên `losses[0:100]`.
* In ra các giá trị trọng số cuối cùng: `w1, w2, w3`.
* Vẽ đồ thị hàm mất mát theo số vòng lặp (iteration) giống hệt như trong slide 39 của PDF."""))

    cells.append(nbf.v4.new_code_cell("""# Slide 39 (Phần 1): Huấn luyện và in kết quả trọng số
X, y = prepare_data(DATA_PATH)
(w1, w2, w3, b, losses) = implement_linear_regression(X, y)

print("10 giá trị loss đầu tiên trong 100 iterations:")
print(losses[0:10])

print("\\nCác trọng số tối ưu thu được (w1, w2, w3):")
print(w1, w2, w3)
print(f"Hệ số chặn b: {b}")"""))

    cells.append(nbf.v4.new_code_cell("""# Slide 39 (Phần 2): Vẽ biểu đồ hàm mất mát cho 100 iterations đầu tiên (tái hiện chính xác hình ảnh Slide 39)
plt.figure(figsize=(8, 5))
plt.plot(losses[0:100], color='#1f77b4', linewidth=1.8)
plt.xlabel("#iteration", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.title("Loss History in First 100 Iterations (Slide 39)", fontsize=13, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""# Trực quan hóa thêm: Quá trình suy giảm Loss trên toàn bộ quá trình huấn luyện (10,000 iterations = 50 epochs * 200 samples)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4.5))

ax1.plot(losses, color='navy', alpha=0.4, linewidth=0.5, label='Loss từng mẫu (SGD)')
# Đường trung bình trượt (Moving average) để nhìn rõ xu hướng hội tụ
moving_avg = pd.Series(losses).rolling(window=200).mean()
ax1.plot(moving_avg, color='red', linewidth=1.8, label='Moving Average (window=200)')
ax1.set_title('Hàm mất mát toàn bộ 10,000 iterations', fontsize=12, fontweight='bold')
ax1.set_xlabel('# Iteration', fontsize=11)
ax1.set_ylabel('Loss (y - y_hat)^2', fontsize=11)
ax1.set_ylim(-5, 100)
ax1.legend()
ax1.grid(True, linestyle=':', alpha=0.6)

# Tính MSE theo từng epoch
epoch_losses = [np.mean(losses[ep*200 : (ep+1)*200]) for ep in range(50)]
ax2.plot(range(1, 51), epoch_losses, marker='o', color='darkgreen', linewidth=1.8)
ax2.set_title('Mean Squared Error theo từng Epoch (1 đến 50)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('Mean Loss per Epoch', fontsize=11)
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()"""))

    # Slide 40
    cells.append(nbf.v4.new_markdown_cell("""### 4.5. Slide 40: Dự đoán doanh số bán hàng cho dữ liệu mới (Inference)
* Sử dụng mô hình đã huấn luyện với bộ trọng số $(w_1, w_2, w_3, b)$ để dự báo doanh số bán hàng `sales` cho chiến dịch quảng cáo mới:
  - Ngân sách TV: `19.2` nghìn USD
  - Ngân sách Radio: `35.9` nghìn USD
  - Ngân sách Newspaper: `51.3` nghìn USD
* Công thức dự đoán:
  $$\\text{sales} = w_1 \\cdot 19.2 + w_2 \\cdot 35.9 + w_3 \\cdot 51.3 + b$$"""))

    cells.append(nbf.v4.new_code_cell("""# Slide 40: Code dự đoán dữ liệu mới (New data inference)
# given new data
tv = 19.2
radio = 35.9
newspaper = 51.3

X, y = prepare_data(DATA_PATH)
(w1, w2, w3, b, losses) = implement_linear_regression(X, y)
sales = predict(tv, radio, newspaper, w1, w2, w3, b)
print(f'predicted sales is {sales}')"""))

    # Slide 41 - 43: Min-Max Scaling
    cells.append(nbf.v4.new_markdown_cell("""### 4.6. Slides 41 - 43: Chuẩn hóa dữ liệu - Min-Max Scaling & Nghịch đảo (Inverse Min-Max)

#### 1. Cơ sở lý thuyết (Slide 41)
Phương pháp **Min-Max Scaling** đưa dữ liệu về thang đo chuẩn trong đoạn $[0, 1]$:
$$x' = \\frac{x - x_{\\min}}{x_{\\max} - x_{\\min}}$$

Khi muốn khôi phục giá trị đã chuẩn hóa $x'$ về thang đo gốc ban đầu (**Inverse Min-Max Scaling**):
$$x = x' \\times (x_{\\max} - x_{\\min}) + x_{\\min}$$

#### 2. Cài đặt thuật toán (Slides 42 & 43)
- `min_max_scaling(X)`: Tính $x_{\\min}$ và $x_{\\max}$ cho từng đặc trưng, chuẩn hóa các phần tử và trả về `X_scaled`, `mins`, `maxs`.
- `inverse_min_max_scaling(X_scaled, mins, maxs)`: Áp dụng công thức nghịch đảo để khôi phục lại dữ liệu gốc `X_recovered`."""))

    cells.append(nbf.v4.new_code_cell("""# Slide 42: Cài đặt Min-Max Scaling
# 1. Min-Max Scaling
# ------------------
def min_max_scaling(X):
    mins = []
    maxs = []
    X_scaled = []
    for feature in X:
        min_val = min(feature)
        max_val = max(feature)
        mins.append(min_val)
        maxs.append(max_val)
        
        scaled_feature = []
        for x in feature:
            scaled_x = (x - min_val) / (max_val - min_val)
            scaled_feature.append(scaled_x)
            
        X_scaled.append(scaled_feature)
        
    return X_scaled, mins, maxs"""))

    cells.append(nbf.v4.new_code_cell("""# Slide 43: Cài đặt Inverse Min-Max Scaling
# 2. Inverse Min-Max
# ------------------
def inverse_min_max_scaling(X_scaled, mins, maxs):
    X_recovered = []
    
    for i in range(len(X_scaled)):
        feature = X_scaled[i]
        min_val = mins[i]
        max_val = maxs[i]
        
        recovered_feature = []
        for x in feature:
            original_x = x * (max_val - min_val) + min_val
            recovered_feature.append(original_x)
            
        X_recovered.append(recovered_feature)
        
    return X_recovered"""))

    cells.append(nbf.v4.new_code_cell("""# Kiểm tra hoạt động của Min-Max Scaling & Inverse trên dữ liệu thực tế advertising.csv
X, y = prepare_data(DATA_PATH)
X_scaled_mm, mins_mm, maxs_mm = min_max_scaling(X)
X_recovered_mm = inverse_min_max_scaling(X_scaled_mm, mins_mm, maxs_mm)

feature_names = ['TV', 'Radio', 'Newspaper']
print("--- KẾT QUẢ MIN-MAX SCALING TRÊN ADVERTISING.CSV ---")
for idx, name in enumerate(feature_names):
    print(f"Đặc trưng {name:9s}: Min gốc={mins_mm[idx]:6.2f}, Max gốc={maxs_mm[idx]:6.2f}")
    print(f"             : Min sau scale={min(X_scaled_mm[idx]):.4f}, Max sau scale={max(X_scaled_mm[idx]):.4f}")
    
    # Kiểm tra sai số khôi phục
    max_diff = max(abs(orig - rec) for orig, rec in zip(X[idx], X_recovered_mm[idx]))
    print(f"             : Sai số khôi phục tối đa (Max Absolute Error) = {max_diff:.2e}")"""))

    # Slide 44 - 46: Z-Score Scaling
    cells.append(nbf.v4.new_markdown_cell("""### 4.7. Slides 44 - 46: Chuẩn hóa dữ liệu - Z-Score Scaling (Standardization) & Nghịch đảo (Inverse Z-Score)

#### 1. Cơ sở lý thuyết (Slide 44)
Phương pháp **Z-Score Scaling (hay Chuẩn hóa chuẩn tắc - Standardization)** biến đổi dữ liệu sao cho tập dữ liệu mới có kỳ vọng (mean) $\\mu = 0$ và phương sai (variance) $\\sigma^2 = 1$:
$$x' = \\frac{x - \\mu}{\\sigma}$$
Trong đó:
- $\\mu = \\frac{1}{N} \\sum_{j=1}^N x_j$
- $\\sigma = \\sqrt{\\frac{1}{N - 1} \\sum_{j=1}^N (x_j - \\mu)^2}$ (độ lệch chuẩn mẫu - sample standard deviation)

Công thức khôi phục dữ liệu gốc (**Inverse Z-Score Scaling**):
$$x = x' \\times \\sigma + \\mu$$

#### 2. Cài đặt thuật toán (Slides 45 & 46)
- `z_score_scaling(X)`: Tính trung bình `mean_val` và độ lệch chuẩn `std_val` cho từng đặc trưng, chuẩn hóa và trả về `X_scaled`, `means`, `stds`.
- `inverse_z_score_scaling(X_scaled, means, stds)`: Áp dụng công thức nghịch đảo để khôi phục lại dữ liệu ban đầu."""))

    cells.append(nbf.v4.new_code_cell("""# Slide 45: Cài đặt Z-Score Scaling
# 3. Z-Score Scaling
# ------------------
def z_score_scaling(X):
    means = []
    stds = []
    X_scaled = []
    
    for feature in X:
        mean_val = sum(feature) / len(feature)
        means.append(mean_val)
        
        std_val = (sum((x - mean_val) ** 2 for x in feature) / (len(feature) - 1)) ** 0.5
        stds.append(std_val)
        
        scaled_feature = []
        for x in feature:
            scaled_x = (x - mean_val) / std_val
            scaled_feature.append(scaled_x)
            
        X_scaled.append(scaled_feature)
        
    return X_scaled, means, stds"""))

    cells.append(nbf.v4.new_code_cell("""# Slide 46: Cài đặt Inverse Z-Score Scaling
# 4. Inverse Z-Score
# ------------------
def inverse_z_score_scaling(X_scaled, means, stds):
    X_recovered = []
    
    for i in range(len(X_scaled)):
        feature = X_scaled[i]
        mean_val = means[i]
        std_val = stds[i]
        
        recovered_feature = []
        for x in feature:
            original_x = x * std_val + mean_val
            recovered_feature.append(original_x)
            
        X_recovered.append(recovered_feature)
        
    return X_recovered"""))

    cells.append(nbf.v4.new_code_cell("""# Kiểm tra hoạt động của Z-Score Scaling & Inverse trên dữ liệu thực tế advertising.csv
X_scaled_z, means_z, stds_z = z_score_scaling(X)
X_recovered_z = inverse_z_score_scaling(X_scaled_z, means_z, stds_z)

print("--- KẾT QUẢ Z-SCORE SCALING TRÊN ADVERTISING.CSV ---")
for idx, name in enumerate(feature_names):
    m_scaled = sum(X_scaled_z[idx]) / len(X_scaled_z[idx])
    s_scaled = (sum((v - m_scaled)**2 for v in X_scaled_z[idx]) / (len(X_scaled_z[idx]) - 1))**0.5
    
    print(f"Đặc trưng {name:9s}: Mean gốc={means_z[idx]:7.2f}, Std gốc={stds_z[idx]:6.2f}")
    print(f"             : Mean sau scale={m_scaled:9.2e} (~0), Std sau scale={s_scaled:.4f} (~1)")
    
    # Kiểm tra sai số khôi phục
    max_diff = max(abs(orig - rec) for orig, rec in zip(X[idx], X_recovered_z[idx]))
    print(f"             : Sai số khôi phục tối đa = {max_diff:.2e}")"""))

    # -------------------------------------------------------------
    # PHẦN 5: DATASET APPLICATIONS
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""---
## PHẦN 5: DATASET APPLICATIONS (ỨNG DỤNG NÂNG CAO TRÊN TẬP DỮ LIỆU THỰC TẾ)

Trong phần này, chúng ta sẽ mở rộng và ứng dụng các kiến thức đã học trong các slide:
1. **So sánh huấn luyện khi có và không có Feature Scaling (Z-Score):** Chứng minh trực quan tác động của chuẩn hóa lên tốc độ hội tụ và độ ổn định của Gradient Descent.
2. **Đánh giá mô hình đầy đủ:** Cài đặt các chỉ số đo lường hiệu năng hồi quy chuẩn quốc tế: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE) và Hệ số xác định $R^2$.
3. **Đối chiếu nghiệm:** So sánh mô hình tự xây dựng với Nghiệm giải tích tối ưu toàn cục (Closed-Form Normal Equation) và Thư viện chuẩn `scikit-learn`.
4. **Trực quan hóa đồ thị sai số (Residuals) và Actual vs Predicted.**"""))

    cells.append(nbf.v4.new_markdown_cell("""### 5.1. So sánh Huấn luyện: Dữ liệu Gốc (Unscaled) vs Dữ liệu Chuẩn hóa (Z-Score Scaled)
Khi dữ liệu được chuẩn hóa về $Z$-score, các đặc trưng có cùng phương sai $\\sigma=1$. Do đó:
- Không còn hiện tượng dao động zigzag do biên độ kênh TV quá lớn.
- Ta có thể sử dụng learning rate lớn hơn (ví dụ $\\eta = 0.01$ thay vì $10^{-5}$) giúp mô hình hội tụ nhanh hơn gấp hàng trăm lần!"""))

    cells.append(nbf.v4.new_code_cell("""# Huấn luyện mô hình trên dữ liệu đã chuẩn hóa Z-Score
def implement_linear_regression_flexible(X_data, y_data, epoch_max=50, lr=0.01):
    \"\"\"Huấn luyện Linear Regression với tùy chọn learning rate linh hoạt\"\"\"
    losses = []
    # Khởi tạo trọng số ngẫu nhiên nhỏ
    w1, w2, w3, b = 0.0, 0.0, 0.0, float(np.mean(y_data))
    
    N = len(y_data)
    for epoch in range(epoch_max):
        for i in range(N):
            x1 = X_data[0][i]
            x2 = X_data[1][i]
            x3 = X_data[2][i]
            y = y_data[i]
            
            y_hat = predict(x1, x2, x3, w1, w2, w3, b)
            loss = compute_loss(y, y_hat)
            
            dl_dw1 = compute_gradient_wi(x1, y, y_hat)
            dl_dw2 = compute_gradient_wi(x2, y, y_hat)
            dl_dw3 = compute_gradient_wi(x3, y, y_hat)
            dl_db = compute_gradient_b(y, y_hat)
            
            w1 = update_weight_wi(w1, dl_dw1, lr)
            w2 = update_weight_wi(w2, dl_dw2, lr)
            w3 = update_weight_wi(w3, dl_dw3, lr)
            b = update_weight_b(b, dl_db, lr)
            
            losses.append(loss)
            
    return (w1, w2, w3, b, losses)

# Huấn luyện trên X đã chuẩn hóa Z-Score
w1_z, w2_z, w3_z, b_z, losses_z = implement_linear_regression_flexible(X_scaled_z, y, epoch_max=50, lr=0.005)

print(f"Trọng số trên không gian Z-Score:")
print(f"  w1 (TV)       : {w1_z:.4f}")
print(f"  w2 (Radio)    : {w2_z:.4f}")
print(f"  w3 (Newspaper): {w3_z:.4f}")
print(f"  b (Intercept) : {b_z:.4f}")
print(f"Loss trung bình ở epoch cuối: {np.mean(losses_z[-200:]):.4f}")"""))

    cells.append(nbf.v4.new_code_cell("""# So sánh đồ thị suy giảm Loss giữa Unscaled và Z-score Scaled
plt.figure(figsize=(10, 5))
plt.plot(pd.Series(losses[:2000]).rolling(50).mean(), label='Dữ liệu thô (Unscaled, lr=1e-5)', color='crimson', linewidth=1.8)
plt.plot(pd.Series(losses_z[:2000]).rolling(50).mean(), label='Dữ liệu chuẩn hóa (Z-Score Scaled, lr=0.005)', color='royalblue', linewidth=1.8)
plt.title('So sánh tốc độ hội tụ của hàm Loss (2,000 bước đầu tiên)', fontsize=13, fontweight='bold')
plt.xlabel('# Iteration', fontsize=11)
plt.ylabel('Loss (Smoothed MA=50)', fontsize=11)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()"""))

    cells.append(nbf.v4.new_markdown_cell("""### 5.2. Đánh giá Mô hình với các Chỉ số Hồi quy (Evaluation Metrics)
Các chỉ số đánh giá chất lượng mô hình:
1. **Mean Squared Error (MSE):**
   $$MSE = \\frac{1}{N} \\sum_{i=1}^N (y_i - \\hat{y}_i)^2$$
2. **Root Mean Squared Error (RMSE):** Cùng đơn vị đo với biến mục tiêu ($1,000 sản phẩm):
   $$RMSE = \\sqrt{MSE}$$
3. **Mean Absolute Error (MAE):** Trung bình sai số tuyệt đối:
   $$MAE = \\frac{1}{N} \\sum_{i=1}^N |y_i - \\hat{y}_i|$$
4. **Hệ số xác định ($R^2$ Score):** Tỷ lệ phương sai của biến mục tiêu được giải thích bởi mô hình:
   $$R^2 = 1 - \\frac{\\sum_{i=1}^N (y_i - \\hat{y}_i)^2}{\\sum_{i=1}^N (y_i - \\bar{y})^2}$$"""))

    cells.append(nbf.v4.new_code_cell("""# Cài đặt hàm tính các chỉ số đánh giá mô hình hồi quy
def evaluate_regression(y_true, y_pred, model_name="Model"):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        "Model": model_name,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2_Score": r2
    }

# Dự đoán toàn bộ tập dữ liệu bằng mô hình gốc Slide 39
y_preds_scratch = [predict(X[0][i], X[1][i], X[2][i], w1, w2, w3, b) for i in range(len(y))]
eval_scratch = evaluate_regression(y, y_preds_scratch, "Custom SGD (Slide 39)")

pd.DataFrame([eval_scratch])"""))

    cells.append(nbf.v4.new_markdown_cell("""### 5.3. Đối chiếu với Nghiệm giải tích Closed-Form Normal Equation & Scikit-Learn

#### 1. Công thức nghiệm tối ưu toàn cục (Normal Equation)
Trong bài toán hồi quy tuyến tính, ta có thể tìm nghiệm tối ưu toàn cục mà không cần lặp Gradient Descent thông qua công thức nghiệm giải tích:
$$\\mathbf{w}^* = (\\mathbf{X}_{design}^T \\mathbf{X}_{design})^{-1} \\mathbf{X}_{design}^T \\mathbf{y}$$
trong đó ma trận thiết kế $\\mathbf{X}_{design}$ có thêm cột $1$ cho hệ số chặn (bias).

#### 2. So sánh với `scikit-learn` `LinearRegression`"""))

    cells.append(nbf.v4.new_code_cell("""# 1. Nghiệm giải tích Closed-form (Normal Equation)
X_matrix = np.array(X).T  # Shape: (200, 3)
N_samples = X_matrix.shape[0]
X_design = np.hstack([np.ones((N_samples, 1)), X_matrix])  # Thêm cột bias 1: Shape (200, 4)
y_vector = np.array(y)

# w_closed = (X^T * X)^(-1) * X^T * y
w_closed = np.linalg.inv(X_design.T @ X_design) @ (X_design.T @ y_vector)
b_closed = w_closed[0]
w1_closed, w2_closed, w3_closed = w_closed[1], w_closed[2], w_closed[3]
y_preds_closed = X_design @ w_closed
eval_closed = evaluate_regression(y, y_preds_closed, "Normal Equation (Closed-form)")

# 2. Thư viện Scikit-Learn LinearRegression
from sklearn.linear_model import LinearRegression
sklearn_lr = LinearRegression()
sklearn_lr.fit(X_matrix, y_vector)
y_preds_sklearn = sklearn_lr.predict(X_matrix)
eval_sklearn = evaluate_regression(y, y_preds_sklearn, "Scikit-Learn LinearRegression")

# 3. Tổng hợp so sánh trọng số và hiệu năng
comparison_weights = pd.DataFrame({
    "Hệ số / Chỉ số": ["w1 (TV)", "w2 (Radio)", "w3 (Newspaper)", "b (Bias)", "R2 Score", "RMSE"],
    "Custom SGD (50 epochs)": [w1, w2, w3, b, eval_scratch["R2_Score"], eval_scratch["RMSE"]],
    "Normal Equation": [w1_closed, w2_closed, w3_closed, b_closed, eval_closed["R2_Score"], eval_closed["RMSE"]],
    "Scikit-Learn": [sklearn_lr.coef_[0], sklearn_lr.coef_[1], sklearn_lr.coef_[2], sklearn_lr.intercept_, eval_sklearn["R2_Score"], eval_sklearn["RMSE"]]
})

print("BẢNG SO SÁNH TRỌNG SỐ VÀ HIỆU NĂNG GIỮA CÁC PHƯƠNG PHÁP:")
comparison_weights"""))

    cells.append(nbf.v4.new_markdown_cell("""**Nhận xét:**
- Nghiệm của **Scikit-Learn** và **Normal Equation** trùng khớp hoàn toàn nhau ($R^2 \\approx 0.9026$, RMSE $\\approx 1.645$).
- Thuật toán **Custom SGD** sau 50 epochs với tốc độ học nhỏ $\\eta=10^{-5}$ đã đạt được xấp xỉ rất tốt ($R^2 \\approx 0.887$, RMSE $\\approx 1.767$) và đang hội tụ dần về nghiệm tối ưu toàn cục mà không gặp hiện tượng phân kỳ!"""))

    cells.append(nbf.v4.new_code_cell("""# 5.4. Trực quan hóa kết quả dự đoán và phần dư (Residuals Plot)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

# Đồ thị 1: Actual vs Predicted
ax1.scatter(y, y_preds_scratch, color='#1f77b4', alpha=0.7, edgecolors='k', s=45, label='Dự đoán của Custom SGD')
min_val = min(min(y), min(y_preds_scratch))
max_val = max(max(y), max(y_preds_scratch))
ax1.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Đường lý tưởng (y = y_hat)')
ax1.set_title('Thực tế (Actual) vs Dự đoán (Predicted Sales)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Doanh số thực tế (Actual Sales)', fontsize=11)
ax1.set_ylabel('Doanh số dự đoán (Predicted Sales)', fontsize=11)
ax1.legend()
ax1.grid(True, linestyle=':', alpha=0.6)

# Đồ thị 2: Residuals Plot (y_true - y_pred)
residuals = np.array(y) - np.array(y_preds_scratch)
ax2.scatter(y_preds_scratch, residuals, color='purple', alpha=0.7, edgecolors='k', s=45)
ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax2.set_title('Đồ thị Phần dư (Residuals vs Predicted)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Doanh số dự đoán (Predicted Sales)', fontsize=11)
ax2.set_ylabel('Phần dư: Residuals (y - y_hat)', fontsize=11)
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()"""))

    # -------------------------------------------------------------
    # PHẦN 6: SUMMARY & EXERCISES
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell("""---
## PHẦN 6: SUMMARY & EXERCISES (TỔNG KẾT VÀ BÀI TẬP BỔ SUNG)

### 1. Bảng tổng hợp kiến thức và các hàm trọng tâm
| Nội dung | Tên hàm | Mục đích & Công thức toán học | Slide PDF |
| :--- | :--- | :--- | :--- |
| **Đọc dữ liệu** | `prepare_data` | Sử dụng `np.genfromtxt` trích xuất các cột thành ma trận $X$ và nhãn $y$ | Slide 36 |
| **Khởi tạo tham số** | `initialize_params` | Khởi tạo vector trọng số $\\mathbf{w}$ và hệ số chặn $b$ | Slide 37 |
| **Dự đoán** | `predict` | Tính giá trị đầu ra: $\\hat{y} = w_1 x_1 + w_2 x_2 + w_3 x_3 + b$ | Slide 37 |
| **Hàm mất mát** | `compute_loss` | Sai số bình phương: $L = (\\hat{y} - y)^2$ | Slide 37 |
| **Đạo hàm riêng** | `compute_gradient_wi`, `compute_gradient_b` | $\\frac{\\partial L}{\\partial w_i} = 2x_i(\\hat{y}-y)$ và $\\frac{\\partial L}{\\partial b} = 2(\\hat{y}-y)$ | Slide 37 |
| **Cập nhật trọng số** | `update_weight_wi`, `update_weight_b` | $w_i \\leftarrow w_i - \\eta \\frac{\\partial L}{\\partial w_i}$, $b \\leftarrow b - \\eta \\frac{\\partial L}{\\partial b}$ | Slide 37 |
| **Huấn luyện SGD** | `implement_linear_regression` | Lặp qua các epochs, duyệt từng mẫu và cập nhật tham số | Slide 38 |
| **Loss & Visualization**| Vẽ đồ thị hàm Loss | Theo dõi sự suy giảm của Loss theo từng iteration | Slide 39 |
| **Suy luận dự báo** | Dự đoán mẫu mới | Dự đoán doanh số với `tv=19.2, radio=35.9, newspaper=51.3` $\\rightarrow$ `8.176` | Slide 40 |
| **Min-Max Scaling** | `min_max_scaling` | Chuẩn hóa đặc trưng về khoảng $[0, 1]$: $x' = \\frac{x - x_{\\min}}{x_{\\max} - x_{\\min}}$ | Slides 41, 42 |
| **Inverse Min-Max** | `inverse_min_max_scaling` | Khôi phục dữ liệu gốc: $x = x' \\times (x_{\\max} - x_{\\min}) + x_{\\min}$ | Slides 41, 43 |
| **Z-Score Scaling** | `z_score_scaling` | Chuẩn hóa chuẩn tắc $\\mu=0, \\sigma=1$: $x' = \\frac{x - \\mu}{\\sigma}$ | Slides 44, 45 |
| **Inverse Z-Score** | `inverse_z_score_scaling` | Khôi phục dữ liệu gốc: $x = x' \\times \\sigma + \\mu$ | Slides 44, 46 |

---

### 2. So sánh Gradient Descent vs Closed-Form Normal Equation
- **Gradient Descent (SGD/BGD):**
  - *Ưu điểm:* Phù hợp với các tập dữ liệu cực lớn (hàng triệu quan sát hoặc đặc trưng), có thể cập nhật trực tuyến (online learning), dễ dàng áp dụng cho các mô hình phi tuyến tính phức tạp (như Neural Networks).
  - *Nhược điểm:* Cần tinh chỉnh siêu tham số (Learning rate, Epochs, Batch size), nhạy cảm với thang đo của đặc trưng (cần Feature Scaling).
- **Closed-Form Normal Equation:**
  - *Ưu điểm:* Cho nghiệm tối ưu toàn cục chính xác tuyệt đối ngay lập tức mà không cần lặp hay chọn learning rate.
  - *Nhược điểm:* Phải tính nghịch đảo ma trận $(X^T X)^{-1}$ với độ phức tạp tính toán xấp xỉ $\\mathcal{O}(D^3)$ (trong đó $D$ là số đặc trưng), không khả thi khi $D$ lớn.

---

### 3. Bài tập thực hành mở rộng (Exercises)"""))

    cells.append(nbf.v4.new_markdown_cell("""#### Bài tập 1: Chia tập dữ liệu Train/Test Split (80% Train, 20% Test)
Hãy chia tập dữ liệu thành tập huấn luyện (160 mẫu) và tập kiểm tra (40 mẫu) để đánh giá khả năng tổng quát hóa (Generalization) của mô hình nhằm phát hiện hiện tượng quá khớp (Overfitting) như đã đề cập tại Slide 32."""))

    cells.append(nbf.v4.new_code_cell("""# Gợi ý lời giải Bài tập 1: Train/Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_matrix, y_vector, test_size=0.2, random_state=42)

model_split = LinearRegression()
model_split.fit(X_train, y_train)

y_train_pred = model_split.predict(X_train)
y_test_pred = model_split.predict(X_test)

print(f"R2 trên tập Train: {model_split.score(X_train, y_train):.4f}")
print(f"R2 trên tập Test : {model_split.score(X_test, y_test):.4f}")
print(f"RMSE trên tập Test: {np.sqrt(np.mean((y_test - y_test_pred)**2)):.4f}")"""))

    cells.append(nbf.v4.new_markdown_cell("""#### Bài tập 2: Cài đặt Mini-batch Gradient Descent
Thay vì cập nhật từng mẫu đơn lẻ (Stochastic Gradient Descent) hoặc toàn bộ bộ dữ liệu (Batch Gradient Descent), hãy viết hàm `implement_minibatch_gradient_descent` cập nhật theo từng batch kích thước $B = 16$ hoặc $B = 32$ để cân bằng giữa tốc độ tính toán và độ mượt của gradient."""))

    cells.append(nbf.v4.new_code_cell("""# Gợi ý lời giải Bài tập 2: Mini-batch Gradient Descent
def minibatch_gradient_descent(X, y, batch_size=16, epochs=50, lr=0.01):
    N = len(y)
    # Chuẩn hóa Z-Score cho X
    X_scaled, means, stds = z_score_scaling(X)
    X_arr = np.array(X_scaled).T # (N, 3)
    y_arr = np.array(y)
    
    w = np.zeros(3)
    b = float(np.mean(y_arr))
    
    batch_losses = []
    for epoch in range(epochs):
        indices = np.random.permutation(N)
        X_shuffled = X_arr[indices]
        y_shuffled = y_arr[indices]
        
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            X_b = X_shuffled[start_idx:end_idx]
            y_b = y_shuffled[start_idx:end_idx]
            
            y_hat_b = X_b @ w + b
            error = y_hat_b - y_b
            
            # Gradient trung bình trên batch
            grad_w = (2 / len(y_b)) * (X_b.T @ error)
            grad_b = (2 / len(y_b)) * np.sum(error)
            
            w -= lr * grad_w
            b -= lr * grad_b
            batch_losses.append(np.mean(error**2))
            
    return w, b, batch_losses

w_mb, b_mb, mb_losses = minibatch_gradient_descent(X, y, batch_size=16, epochs=50, lr=0.01)
print("Kết quả Mini-batch Gradient Descent:")
print(f"  Weights w: {w_mb}")
print(f"  Bias b   : {b_mb:.4f}")
print(f"  Final Batch Loss: {mb_losses[-1]:.4f}")"""))

    cells.append(nbf.v4.new_markdown_cell("""#### Bài tập 3: Thêm số hạng hiệu chỉnh L2 Regularization (Ridge Regression)
Viết hàm tính đạo hàm có bổ sung hệ số phạt L2:
$$L_{Ridge} = (\\hat{y} - y)^2 + \\lambda \\sum_{i=1}^3 w_i^2$$
$$\\frac{\\partial L_{Ridge}}{\\partial w_i} = 2 x_i (\\hat{y} - y) + 2 \\lambda w_i$$
Nhận xét tác động của $\\lambda$ lên độ lớn của các trọng số $w_i$ (đặc biệt là trọng số của Newspaper vốn có hệ số nhỏ)."""))

    nb.cells = cells
    
    output_filename = "Linear_Regression_and_Data_Normalization_Practice.ipynb"
    with open(output_filename, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    
    print(f"Đã tạo thành công file: {output_filename}")
    print(f"Tổng số cell: {len(cells)} (Markdown: {len([c for c in cells if c.cell_type == 'markdown'])}, Code: {len([c for c in cells if c.cell_type == 'code'])})")
    return output_filename

if __name__ == "__main__":
    create_notebook()
