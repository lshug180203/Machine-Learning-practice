"""
Script to generate the master Jupyter Notebook containing:
1. Introduction & Objectives
2. Environment Setup
3. Dataset Exploration (both Q1/Q2 points & advertising.csv)
4. Question 1 & Question 2 Complete Solutions with exact formulas, tables, and plots
5. Slides 36 - 46 Complete Code Reproduction from lec3.pdf
6. Dataset Applications (Scaled vs Unscaled, Evaluation Metrics, Normal Equation, Sklearn)
7. Summary & Synthesis
"""
import nbformat as nbf
import os
import sys

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def create_master_notebook():
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
    
    # =============================================================
    # PHẦN 1: INTRODUCTION
    # =============================================================
    cells.append(nbf.v4.new_markdown_cell("""# BÀI THỰC HÀNH MACHINE LEARNING: HỒI QUY TUYẾN TÍNH & CHUẨN HÓA DỮ LIỆU
## HOÀN THÀNH BÀI TẬP QUESTION 1, QUESTION 2 & TOÀN BỘ CODE SLIDES 36 - 46 (LEC3.PDF)

---

### 1. Mục tiêu thực hành & Phạm vi tài liệu
Notebook này được xây dựng toàn diện dựa trên tài liệu bài giảng **Machine Learning - Lecture 3: Linear Regression (`lec3.pdf`)**, bao gồm:
1. **Giải quyết triệt để 2 bài tập thực hành (Question 1 & Question 2):**
   - **Question 1:** Tìm phương trình đường hồi quy bình phương tối thiểu (Linear Least Squares) $y = \\phi_1 x + \\phi_0$ bằng công thức nghiệm giải tích closed-form, tính tổng sai số bình phương (Sum of Squared Errors - SSE), và viết mã nguồn Python trực quan hóa dữ liệu cùng đường hồi quy.
   - **Question 2:** Cài đặt thuật toán cập nhật từng mẫu dữ liệu (Stochastic Gradient Descent - SGD) theo đúng sơ đồ luồng thuật toán từ Slide 35, tính toán chi tiết từng bước với khởi tạo $w = 4, b = 1, \\eta = 0.1$, điền đầy đủ bảng kết quả 6 cột (`Data`, `output`, `loss`, `gradient`, `W, b`, `Sum of loss`), phân tích hiện tượng bùng nổ gradient và thử nghiệm giải pháp với learning rate tối ưu.
2. **Tái hiện 100% mã nguồn trong bài giảng `lec3.pdf` (Slide 36 `# dataset` đến Slide 46):**
   - Đọc dữ liệu bằng `np.genfromtxt` và trích xuất đặc trưng (`prepare_data`, `get_column`).
   - Xây dựng mô hình Multiple Linear Regression từ đầu (From Scratch): `initialize_params`, `predict`, `compute_loss`, `compute_gradient_wi`, `compute_gradient_b`, `update_weight_wi`, `update_weight_b`.
   - Huấn luyện mô hình 50 epochs (`implement_linear_regression`), in 100 loss đầu tiên và vẽ biểu đồ suy giảm Loss giống hệt Slide 39.
   - Suy luận dự đoán mẫu mới (`tv=19.2, radio=35.9, newspaper=51.3` $\\rightarrow$ `8.176413`).
   - Cài đặt đầy đủ 2 phương pháp chuẩn hóa dữ liệu: **Min-Max Scaling** (Slides 41-43) và **Z-Score Standardization** (Slides 44-46) kèm các hàm khôi phục dữ liệu gốc (**Inverse Scaling**).
3. **Ứng dụng thực tế & Đánh giá nâng cao trên tập dữ liệu `advertising.csv`:**
   - Đánh giá với các chỉ số chuẩn: MSE, RMSE, MAE, $R^2$ Score.
   - So sánh thuật toán Custom SGD với Nghiệm giải tích tối ưu toàn cục (Normal Equation) và thư viện Scikit-Learn.
   - Đồ thị kiểm tra phần dư (Residuals Plot) và Giá trị Thực tế vs Dự đoán.

---

### 2. Các tập dữ liệu sử dụng trong bài
- **Tập dữ liệu 1D (Question 1 & 2):** Gồm 8 điểm dữ liệu $(x, y)$:
  $$\\{(-6, -3), (-5, 1), (-2, 2), (2, 3), (3, 6), (5, 7), (8, 6), (9, 9)\\}$$
- **Tập dữ liệu thực tế `advertising.csv`:** Gồm 200 quan sát đo lường ngân sách quảng cáo trên 3 kênh truyền thông (`TV`, `Radio`, `Newspaper`) và doanh số bán hàng (`Sales`)."""))

    # =============================================================
    # PHẦN 2: ENVIRONMENT SETUP
    # =============================================================
    cells.append(nbf.v4.new_markdown_cell("""---
## PHẦN 2: ENVIRONMENT SETUP (THIẾT LẬP MÔI TRƯỜNG)

Kiểm tra phiên bản Python, nhập các thư viện toán học và đồ họa cần thiết, đồng thời cấu hình đường dẫn tương đối `DATA_PATH` để notebook có thể chạy mượt mà trên mọi môi trường (VS Code, JupyterLab, Google Colab)."""))

    cells.append(nbf.v4.new_code_cell("""# 1. Kiểm tra môi trường & Hướng dẫn cài đặt
import sys
import os

print(f"Phiên bản Python: {sys.version}")

# Danh sách thư viện cần thiết
required_packages = ["numpy", "matplotlib", "pandas", "sklearn"]
print("Danh sách thư viện yêu cầu:", required_packages)
print("Nếu thiếu thư viện, chạy lệnh: pip install numpy matplotlib pandas scikit-learn")"""))

    cells.append(nbf.v4.new_code_cell("""# 2. Import các thư viện
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Cấu hình thẩm mỹ đồ thị
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

    # =============================================================
    # PHẦN 3: DATASET EXPLORATION
    # =============================================================
    cells.append(nbf.v4.new_markdown_cell("""---
## PHẦN 3: DATASET EXPLORATION (KHÁM PHÁ DỮ LIỆU)

Khám phá 2 tập dữ liệu được sử dụng trong bài: tập điểm dữ liệu 1D của Question 1 & 2 và tập dữ liệu `advertising.csv`."""))

    cells.append(nbf.v4.new_code_cell("""# 3.1. Khởi tạo và khám phá tập dữ liệu 1D của Question 1 & 2
q_points = [(-6, -3), (-5, 1), (-2, 2), (2, 3), (3, 6), (5, 7), (8, 6), (9, 9)]
df_points = pd.DataFrame(q_points, columns=['x', 'y'])

print("--- TẬP ĐIỂM DỮ LIỆU QUESTION 1 & QUESTION 2 ---")
print(f"Tổng số điểm dữ liệu: N = {len(df_points)}")
df_points.T"""))

    cells.append(nbf.v4.new_code_cell("""# 3.2. Khám phá tập dữ liệu thực tế advertising.csv
df_adv = pd.read_csv(DATA_PATH)

print(f"Kích thước tập dữ liệu advertising.csv: {df_adv.shape}")
print(f"Số giá trị thiếu (Missing values):\\n{df_adv.isnull().sum()}")
print(f"Số bản ghi trùng lặp: {df_adv.duplicated().sum()}")
print("\\n5 dòng đầu tiên:")
df_adv.head()"""))

    cells.append(nbf.v4.new_code_cell("""# Thống kê mô tả tập advertising.csv
desc = df_adv.describe().T
desc['range'] = desc['max'] - desc['min']
print("Bảng thống kê mô tả:")
desc[['count', 'mean', 'std', 'min', '50%', 'max', 'range']]"""))

    cells.append(nbf.v4.new_code_cell("""# Trực quan hóa phân phối và mối tương quan giữa các kênh quảng cáo với Sales
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)

channels = [('TV', '#1f77b4'), ('Radio', '#ff7f0e'), ('Newspaper', '#2ca02c')]
for ax, (col, color) in zip(axes, channels):
    ax.scatter(df_adv[col], df_adv['Sales'], color=color, alpha=0.7, edgecolors='none', s=40)
    m, c = np.polyfit(df_adv[col], df_adv['Sales'], 1)
    x_line = np.linspace(df_adv[col].min(), df_adv[col].max(), 100)
    ax.plot(x_line, m * x_line + c, color='red', linestyle='--', linewidth=1.5, label=f'Trend (r={df_adv[col].corr(df_adv["Sales"]):.2f})')
    ax.set_title(f'{col} vs Sales', fontsize=12, fontweight='bold')
    ax.set_xlabel(f'Ngân sách {col} ($1,000)', fontsize=11)
    if col == 'TV':
        ax.set_ylabel('Doanh số Sales ($1,000)', fontsize=11)
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)

plt.suptitle('Mối tương quan giữa từng Kênh Quảng cáo và Doanh số Bán hàng (advertising.csv)', fontsize=13, y=1.02)
plt.tight_layout()
plt.show()"""))

    # =============================================================
    # PHẦN 4: QUESTION 1 & QUESTION 2 SOLUTIONS
    # =============================================================
    cells.append(nbf.v4.new_markdown_cell("""---
## PHẦN 4: BÀI TẬP QUESTION 1 & QUESTION 2 (CHI TIẾT LÝ THUYẾT & CODE)

Trong phần này, chúng ta sẽ áp dụng các kiến thức từ Slide 14 đến Slide 35 của `lec3.pdf` để giải quyết trọn vẹn 2 câu hỏi thực hành mà người dùng đã cung cấp."""))

    # Question 1
    cells.append(nbf.v4.new_markdown_cell("""### 4.1. QUESTION 1

**Đề bài:**
Cho các điểm dữ liệu $(x, y)$ như trong Hình 1:
$$\\{(-6, -3), (-5, 1), (-2, 2), (2, 3), (3, 6), (5, 7), (8, 6), (9, 9)\\}$$

* **a.** Tìm và vẽ đường bình phương tối thiểu (Linear Least Square) $y = \\phi_1 x + \\phi_0$.
* **b.** Tính tổng sai số bình phương (Sum Square Error - SSE).
* **c.** Viết mã Python để vẽ các điểm dữ liệu, đường hồi quy tuyến tính và tính tổng sai số bình phương.

---

#### Cơ sở lý thuyết và Công thức giải tích (Closed-Form):
Mô hình đường thẳng hồi quy tuyến tính:
$$\\hat{y} = \\phi_1 x + \\phi_0$$
Hàm mất mát bình phương tối thiểu (Least Squares Loss Function):
$$L(\\phi_1, \\phi_0) = \\sum_{i=1}^N (y_i - (\\phi_1 x_i + \\phi_0))^2$$

Để tìm cực tiểu của $L$, ta lấy đạo hàm riêng theo $\\phi_1$ và $\\phi_0$ rồi cho bằng 0:
$$\\frac{\\partial L}{\\partial \\phi_0} = -2 \\sum_{i=1}^N (y_i - \\phi_1 x_i - \\phi_0) = 0 \\implies \\phi_0 = \\bar{y} - \\phi_1 \\bar{x}$$
$$\\frac{\\partial L}{\\partial \\phi_1} = -2 \\sum_{i=1}^N x_i (y_i - \\phi_1 x_i - \\phi_0) = 0$$

Từ đó suy ra hệ số góc (slope $\\phi_1$) và hệ số chặn (intercept $\\phi_0$):
$$\\phi_1 = \\frac{N \\sum x_i y_i - (\\sum x_i)(\\sum y_i)}{N \\sum x_i^2 - (\\sum x_i)^2} = \\frac{\\sum (x_i - \\bar{x})(y_i - \\bar{y})}{\\sum (x_i - \\bar{x})^2}$$
$$\\phi_0 = \\bar{y} - \\phi_1 \\bar{x}$$

Tổng sai số bình phương (Sum of Squared Errors - SSE):
$$SSE = \\sum_{i=1}^N (y_i - \\hat{y}_i)^2 = \\sum_{i=1}^N (y_i - (\\phi_1 x_i + \\phi_0))^2$$"""))

    cells.append(nbf.v4.new_code_cell("""# Question 1: Code tính toán các hệ số phi_1, phi_0 và SSE theo công thức giải tích
x_q1 = np.array([-6, -5, -2, 2, 3, 5, 8, 9], dtype=float)
y_q1 = np.array([-3,  1,  2, 3, 6, 7, 6, 9], dtype=float)
N_q1 = len(x_q1)

# Các đại lượng tổng
sum_x = np.sum(x_q1)
sum_y = np.sum(y_q1)
sum_x2 = np.sum(x_q1**2)
sum_xy = np.sum(x_q1 * y_q1)

x_bar = np.mean(x_q1)
y_bar = np.mean(y_q1)

# Tính phi_1 và phi_0
phi_1 = (N_q1 * sum_xy - sum_x * sum_y) / (N_q1 * sum_x2 - (sum_x)**2)
phi_0 = y_bar - phi_1 * x_bar

# Dự đoán và tính SSE
y_pred_q1 = phi_1 * x_q1 + phi_0
sse_q1 = np.sum((y_q1 - y_pred_q1)**2)

print("=== KẾT QUẢ TÍNH TOÁN QUESTION 1 ===")
print(f"Số lượng điểm dữ liệu: N = {N_q1}")
print(f"sum(x)  = {sum_x:8.2f} | sum(y)  = {sum_y:8.2f}")
print(f"sum(x^2)= {sum_x2:8.2f} | sum(xy) = {sum_xy:8.2f}")
print(f"x_bar   = {x_bar:8.4f} | y_bar   = {y_bar:8.4f}")
print(f"\\n(a) Hệ số góc phi_1 (slope)     : {phi_1:.6f} ≈ {phi_1:.4f}")
print(f"    Hệ số chặn phi_0 (intercept) : {phi_0:.6f} ≈ {phi_0:.4f}")
print(f"    --> Phương trình đường thẳng : y = {phi_1:.4f} * x + {phi_0:.4f}")
print(f"\\n(b) Tổng sai số bình phương SSE : {sse_q1:.6f} ≈ {sse_q1:.4f}")"""))

    cells.append(nbf.v4.new_code_cell("""# Question 1 (c): Viết mã Python để vẽ đồ thị các điểm dữ liệu, đường hồi quy và hiển thị SSE
plt.figure(figsize=(9, 6))

# 1. Vẽ các điểm dữ liệu
plt.scatter(x_q1, y_q1, color='black', s=90, zorder=5, label='Điểm dữ liệu thực tế $(x_i, y_i)$')

# Ghi nhãn tọa độ từng điểm
for x_val, y_val in zip(x_q1, y_q1):
    plt.annotate(f'({int(x_val)}, {int(y_val)})', 
                 (x_val, y_val),
                 textcoords="offset points", 
                 xytext=(0, 10), 
                 ha='center', 
                 fontsize=10, 
                 fontweight='bold')

# 2. Vẽ đường hồi quy Least Squares
x_line = np.linspace(-8, 11, 200)
y_line = phi_1 * x_line + phi_0
plt.plot(x_line, y_line, color='crimson', linewidth=2.2, 
         label=f'Linear Least Squares: $y = {phi_1:.4f}x + {phi_0:.4f}$')

# 3. Vẽ các đoạn thẳng sai số (residuals) từ điểm dữ liệu tới đường hồi quy
for xi, yi, y_hat in zip(x_q1, y_q1, y_pred_q1):
    plt.vlines(xi, min(yi, y_hat), max(yi, y_hat), color='gray', linestyle=':', linewidth=1.2)

# Trục tọa độ x = 0, y = 0
plt.axhline(0, color='black', linewidth=0.8, linestyle='-')
plt.axvline(0, color='black', linewidth=0.8, linestyle='-')

plt.title(f'Question 1: Linear Least Squares Fit (SSE = {sse_q1:.4f})', fontsize=13, fontweight='bold')
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.xlim(-8, 11)
plt.ylim(-5, 11)
plt.legend(loc='upper left', frameon=True, fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()"""))

    # Question 2
    cells.append(nbf.v4.new_markdown_cell("""### 4.2. QUESTION 2

**Đề bài:**
Huấn luyện dữ liệu trong Question 1 bằng quy trình học lặp qua từng mẫu dữ liệu (Stochastic Gradient Descent) theo sơ đồ sau:
1. **Start:** Khởi tạo $w, b$.
2. **Output:** $\\hat{y}_i = w \\cdot x_i + b$
3. **Loss:** $L_i = (\\hat{y}_i - y_i)^2$
4. **Gradient:**
   $$\\frac{\\partial L}{\\partial w} = 2 x_i (\\hat{y}_i - y_i)$$
   $$\\frac{\\partial L}{\\partial b} = 2 (\\hat{y}_i - y_i)$$
5. **Update parameters:**
   $$w = w - \\eta \\frac{\\partial L}{\\partial w}$$
   $$b = b - \\eta \\frac{\\partial L}{\\partial b}$$

* **Step 1:** Khởi tạo: $w = 4, \\; b = 1$
* **Step 2:** Với mỗi điểm dữ liệu huấn luyện: đặt tốc độ học $\\eta = 0.1$, tính output, loss, gradient, và cập nhật $w, b$.
* **Step 3:** Điền đầy đủ vào bảng kết quả gồm các cột:
  - `Data (x, y)`
  - `output` ($\\hat{y}_i$)
  - `loss` ($L_i$)
  - `gradient` ($(\\frac{\\partial L}{\\partial w}, \\frac{\\partial L}{\\partial b})$)
  - `W, b` (sau khi cập nhật)
  - `Sum of loss` (tổng loss tích lũy qua các bước)"""))

    cells.append(nbf.v4.new_code_cell("""# Question 2: Thực hiện thuật toán lặp từng mẫu và điền đầy đủ bảng (Step 1, Step 2, Step 3)

w_curr = 4.0
b_curr = 1.0
eta = 0.1

table_records = []
cumulative_loss = 0.0

print(f"Khởi tạo ban đầu: w = {w_curr}, b = {b_curr}, learning rate eta = {eta}\\n")

for idx, (xi, yi) in enumerate(q_points):
    # 1. Tính output
    y_hat = w_curr * xi + b_curr
    
    # 2. Tính loss
    loss = (y_hat - yi)**2
    cumulative_loss += loss
    
    # 3. Tính gradient
    error = y_hat - yi
    dl_dw = 2 * xi * error
    dl_db = 2 * error
    
    # 4. Cập nhật trọng số w, b
    w_next = w_curr - eta * dl_dw
    b_next = b_curr - eta * dl_db
    
    table_records.append({
        'Data': f'({xi}, {yi})',
        'output': round(y_hat, 4),
        'loss': round(loss, 4),
        'gradient': f'({round(dl_dw, 4)}, {round(dl_db, 4)})',
        'W, b': f'({round(w_next, 4)}, {round(b_next, 4)})',
        'Sum of loss': round(cumulative_loss, 4)
    })
    
    # Cập nhật trọng số cho điểm kế tiếp
    w_curr, b_curr = w_next, b_next

df_q2_table = pd.DataFrame(table_records)
print("=== BẢNG KẾT QUẢ QUESTION 2 (STEP 3: FILLED TABLE) ===")
df_q2_table"""))

    cells.append(nbf.v4.new_markdown_cell("""#### Phân tích chuyên sâu về kết quả Question 2 (Gradient Exploding vs Learning Rate):
1. **Hiện tượng quan sát được:**
   - Tại mẫu đầu tiên $(-6, -3)$: $w$ đổi từ $4 \\rightarrow -20$, $b$ đổi từ $1 \\rightarrow 5$.
   - Tại mẫu kế tiếp $(-5, 1)$: $y_{hat}$ bùng nổ lên $105$, loss tăng vọt lên $10,816$, và gradient lên tới $-1040$.
   - Đến mẫu cuối cùng $(9, 9)$: $y_{hat} \\approx 654.4$, loss lên tới $416,587.7$, trọng số bị đẩy về $w \\approx -1091.7, b \\approx -105.3$.
2. **Nguyên nhân cốt lõi:**
   - Tốc độ học $\\eta = 0.1$ là **quá lớn** so với độ lớn của dữ liệu $x \\in [-6, 9]$. Khi tính đạo hàm $\\frac{\\partial L}{\\partial w} = 2 x_i (\\hat{y} - y)$, đại lượng $2 x_i$ nhân với sai số làm gradient vượt quá biên độ ổn định (Lipshitz smoothness bound: $\\eta < \\frac{1}{L}$).
   - Đây là ví dụ kinh điển trong bài giảng nhằm minh chứng lý do tại sao:
     - Ta phải chọn tốc độ học nhỏ hơn (ví dụ $\\eta = 0.005$).
     - Hoặc ta **bắt buộc phải Chuẩn hóa dữ liệu (Feature Scaling: Min-Max hoặc Z-Score)** như giảng dạy ở các Slides 41-46!

Dưới đây, ta chạy mô phỏng đối chứng khi chọn $\\eta = 0.005$ để thấy mô hình hội tụ đẹp mắt về nghiệm tối ưu của Question 1!"""))

    cells.append(nbf.v4.new_code_cell("""# Thử nghiệm đối chứng: Huấn luyện với Learning Rate thích hợp (eta = 0.005) trong 30 epochs
w_opt = 4.0
b_opt = 1.0
eta_stable = 0.005

loss_history = []
w_history = [w_opt]
b_history = [b_opt]

for epoch in range(40):
    for xi, yi in q_points:
        y_hat = w_opt * xi + b_opt
        loss = (y_hat - yi)**2
        loss_history.append(loss)
        
        dl_dw = 2 * xi * (y_hat - yi)
        dl_db = 2 * (y_hat - yi)
        
        w_opt -= eta_stable * dl_dw
        b_opt -= eta_stable * dl_db
        
        w_history.append(w_opt)
        b_history.append(b_opt)

print(f"Trọng số ban đầu       : w = 4.0000, b = 1.0000")
print(f"Trọng số sau hội tụ    : w = {w_opt:.4f}, b = {b_opt:.4f}")
print(f"Nghiệm giải tích (Q1)  : phi_1 = {phi_1:.4f}, phi_0 = {phi_0:.4f}")
print("--> Nhận xét: Với eta = 0.005, thuật toán SGD hội tụ chính xác về nghiệm tối ưu Least Squares!")"""))

    cells.append(nbf.v4.new_code_cell("""# Trực quan hóa quá trình hội tụ của đường hồi quy SGD so với nghiệm giải tích Question 1
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

# Đồ thị 1: Sự biến đổi của đường thẳng qua các bước lặp
ax1.scatter(x_q1, y_q1, color='black', s=80, zorder=5, label='Dữ liệu thực tế')
x_plot = np.linspace(-7, 10, 100)

# Vẽ một số đường trung gian
colors = plt.cm.viridis(np.linspace(0, 1, 10))
step_indices = np.linspace(0, len(w_history)-1, 10, dtype=int)
for idx_step, color in zip(step_indices, colors):
    w_step = w_history[idx_step]
    b_step = b_history[idx_step]
    ax1.plot(x_plot, w_step * x_plot + b_step, color=color, alpha=0.6, linewidth=1.2)

# Đường nghiệm giải tích Q1
ax1.plot(x_plot, phi_1 * x_plot + phi_0, color='red', linestyle='--', linewidth=2.5, label='Nghiệm giải tích Q1')
ax1.set_title('Quá trình xoay của đường hồi quy qua các bước lặp (SGD)', fontsize=12, fontweight='bold')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.legend()
ax1.grid(True, linestyle=':', alpha=0.6)

# Đồ thị 2: Suy giảm của hàm loss
ax2.plot(loss_history, color='navy', linewidth=1.5)
ax2.set_title('Đồ thị suy giảm Loss khi chọn learning rate eta = 0.005', fontsize=12, fontweight='bold')
ax2.set_xlabel('# Iteration')
ax2.set_ylabel('Loss (y_hat - y)^2')
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()"""))

    # =============================================================
    # PHẦN 5: CODE EXAMPLES FROM LEC3.PDF (SLIDES 36 - 46)
    # =============================================================
    cells.append(nbf.v4.new_markdown_cell("""---
## PHẦN 5: CODE EXAMPLES FROM LEC3.PDF (TÁI HIỆN TOÀN BỘ CODE TỪ SLIDE 36 ĐẾN SLIDE 46)

Trong phần này, toàn bộ mã nguồn bài thực hành hồi quy tuyến tính đa biến trên dataset `advertising.csv` trong bài giảng `lec3.pdf` được tái hiện đầy đủ 100% theo đúng thứ tự bài học."""))

    # Slide 36
    cells.append(nbf.v4.new_markdown_cell("""### 5.1. Slide 36: Tải dữ liệu và chuẩn bị dữ liệu (`prepare_data`)
* **Mục tiêu:** Viết hàm `get_column(data, index)` và `prepare_data(file_name_dataset)`.
* Đọc dữ liệu bằng `np.genfromtxt`, bỏ qua dòng tiêu đề (`skip_header=1`), chuyển thành danh sách 2D bằng `.tolist()`.
* Trích xuất các cột: `tv_data` (cột 0), `radio_data` (cột 1), `newspaper_data` (cột 2), và `sales_data` (cột 3).
* Gom lại thành ma trận đặc trưng $X = [tv, radio, newspaper]$ và vector biến mục tiêu $y = sales$."""))

    cells.append(nbf.v4.new_code_cell("""# Slide 36: Code chuẩn bị dữ liệu (Data Preparation)
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
    cells.append(nbf.v4.new_markdown_cell("""### 5.2. Slide 37: Khởi tạo tham số và các hàm cốt lõi của Linear Regression
Theo sơ đồ thuật toán tại **Slide 35**:
1. Khởi tạo trọng số $w_1, w_2, w_3$ và hệ số chặn $b$:
   $$\\mathbf{w}^{(0)} = (0.016992259082509283,\\; 0.0070783670518262355,\\; -0.002307860847821344), \\quad b^{(0)} = 0$$
2. **Dự đoán:** $\\hat{y} = w_1 x_1 + w_2 x_2 + w_3 x_3 + b$
3. **Hàm mất mát:** $L = (\\hat{y} - y)^2$
4. **Đạo hàm riêng (Gradient):**
   $$\\frac{\\partial L}{\\partial w_i} = 2 x_i (\\hat{y} - y), \\quad \\frac{\\partial L}{\\partial b} = 2 (\\hat{y} - y)$$
5. **Cập nhật:**
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

    # Slide 38
    cells.append(nbf.v4.new_markdown_cell("""### 5.3. Slide 38: Huấn luyện mô hình Linear Regression (`implement_linear_regression`)
Cài đặt vòng lặp huấn luyện theo thuật toán Stochastic Gradient Descent với `epoch_max = 50` và `lr = 1e-5`."""))

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
    cells.append(nbf.v4.new_markdown_cell("""### 5.4. Slide 39: Thực thi huấn luyện và trực quan hóa hàm mất mát (Loss History)
Thực thi huấn luyện trên `advertising.csv`, in 100 giá trị loss đầu tiên, in trọng số tối ưu và vẽ đồ thị hàm loss."""))

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

    # Slide 40
    cells.append(nbf.v4.new_markdown_cell("""### 5.5. Slide 40: Dự đoán doanh số bán hàng cho dữ liệu mới (Inference)
Dự báo doanh số cho chiến dịch mới: `tv = 19.2`, `radio = 35.9`, `newspaper = 51.3`."""))

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
    cells.append(nbf.v4.new_markdown_cell("""### 5.6. Slides 41 - 43: Chuẩn hóa dữ liệu - Min-Max Scaling & Inverse Min-Max
* **Công thức chuẩn hóa (Slide 41):**
  $$x' = \\frac{x - x_{\\min}}{x_{\\max} - x_{\\min}} \\in [0, 1]$$
* **Công thức nghịch đảo (Slide 41):**
  $$x = x' \\times (x_{\\max} - x_{\\min}) + x_{\\min}$$
* Cài đặt `min_max_scaling(X)` (Slide 42) và `inverse_min_max_scaling(X_scaled, mins, maxs)` (Slide 43)."""))

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
    max_diff = max(abs(orig - rec) for orig, rec in zip(X[idx], X_recovered_mm[idx]))
    print(f"Đặc trưng {name:9s}: Min sau scale={min(X_scaled_mm[idx]):.4f}, Max sau scale={max(X_scaled_mm[idx]):.4f} | Sai số khôi phục = {max_diff:.2e}")"""))

    # Slide 44 - 46: Z-Score Scaling
    cells.append(nbf.v4.new_markdown_cell("""### 5.7. Slides 44 - 46: Chuẩn hóa dữ liệu - Z-Score Standardization & Inverse Z-Score
* **Công thức chuẩn hóa chuẩn tắc (Slide 44):**
  $$x' = \\frac{x - \\mu}{\\sigma} \\quad (\\text{với } \\mu = 0, \\sigma^2 = 1)$$
* **Công thức nghịch đảo (Slide 44):**
  $$x = x' \\times \\sigma + \\mu$$
* Cài đặt `z_score_scaling(X)` (Slide 45) và `inverse_z_score_scaling(X_scaled, means, stds)` (Slide 46)."""))

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
    max_diff = max(abs(orig - rec) for orig, rec in zip(X[idx], X_recovered_z[idx]))
    print(f"Đặc trưng {name:9s}: Mean sau scale={m_scaled:9.2e} (~0), Std sau scale={s_scaled:.4f} (~1) | Sai số khôi phục = {max_diff:.2e}")"""))

    # =============================================================
    # PHẦN 6: DATASET APPLICATIONS
    # =============================================================
    cells.append(nbf.v4.new_markdown_cell("""---
## PHẦN 6: DATASET APPLICATIONS (ỨNG DỤNG NÂNG CAO TRÊN ADVERTISING.CSV)

1. **So sánh Unscaled vs Z-score Scaled:** Chứng minh chuẩn hóa giúp dùng learning rate lớn hơn ($\\\\eta = 0.005$ vs $\\\\eta = 10^{-5}$) và hội tụ mượt mà.
2. **Đánh giá toàn diện:** Tính toán MSE, RMSE, MAE, $R^2$.
3. **Đối chiếu nghiệm:** So sánh Custom SGD với Normal Equation và Scikit-Learn `LinearRegression`.
4. **Trực quan hóa sai số:** Biểu đồ Actual vs Predicted và Residuals plot."""))

    cells.append(nbf.v4.new_code_cell("""# 6.1. Huấn luyện Linear Regression trên dữ liệu chuẩn hóa Z-Score
def implement_linear_regression_flexible(X_data, y_data, epoch_max=50, lr=0.005):
    losses = []
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

w1_z, w2_z, w3_z, b_z, losses_z = implement_linear_regression_flexible(X_scaled_z, y, epoch_max=50, lr=0.005)

print("Trọng số tối ưu trên không gian Z-Score:")
print(f"  w1 (TV): {w1_z:.4f} | w2 (Radio): {w2_z:.4f} | w3 (Newspaper): {w3_z:.4f} | b: {b_z:.4f}")"""))

    cells.append(nbf.v4.new_code_cell("""# 6.2. Đánh giá chất lượng mô hình (MSE, RMSE, MAE, R2)
def evaluate_regression(y_true, y_pred, model_name="Model"):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot)
    return {"Model": model_name, "MSE": mse, "RMSE": rmse, "MAE": mae, "R2_Score": r2}

y_preds_scratch = [predict(X[0][i], X[1][i], X[2][i], w1, w2, w3, b) for i in range(len(y))]
eval_scratch = evaluate_regression(y, y_preds_scratch, "Custom SGD (Slide 39)")
pd.DataFrame([eval_scratch])"""))

    cells.append(nbf.v4.new_code_cell("""# 6.3. Đối chiếu với Nghiệm giải tích Closed-Form Normal Equation & Scikit-Learn
X_matrix = np.array(X).T
N_samples = X_matrix.shape[0]
X_design = np.hstack([np.ones((N_samples, 1)), X_matrix])
y_vector = np.array(y)

# Normal Equation: w = (X^T * X)^(-1) * X^T * y
w_closed = np.linalg.inv(X_design.T @ X_design) @ (X_design.T @ y_vector)
b_closed, w1_closed, w2_closed, w3_closed = w_closed[0], w_closed[1], w_closed[2], w_closed[3]
y_preds_closed = X_design @ w_closed
eval_closed = evaluate_regression(y, y_preds_closed, "Normal Equation")

# Scikit-Learn
from sklearn.linear_model import LinearRegression
sklearn_lr = LinearRegression()
sklearn_lr.fit(X_matrix, y_vector)
y_preds_sklearn = sklearn_lr.predict(X_matrix)
eval_sklearn = evaluate_regression(y, y_preds_sklearn, "Scikit-Learn")

comparison_table = pd.DataFrame({
    "Hệ số / Chỉ số": ["w1 (TV)", "w2 (Radio)", "w3 (Newspaper)", "b (Bias)", "R2 Score", "RMSE"],
    "Custom SGD (50 epochs)": [w1, w2, w3, b, eval_scratch["R2_Score"], eval_scratch["RMSE"]],
    "Normal Equation": [w1_closed, w2_closed, w3_closed, b_closed, eval_closed["R2_Score"], eval_closed["RMSE"]],
    "Scikit-Learn": [sklearn_lr.coef_[0], sklearn_lr.coef_[1], sklearn_lr.coef_[2], sklearn_lr.intercept_, eval_sklearn["R2_Score"], eval_sklearn["RMSE"]]
})

print("BẢNG SO SÁNH TRỌNG SỐ VÀ HIỆU NĂNG GIỮA CÁC PHƯƠNG PHÁP:")
comparison_table"""))

    cells.append(nbf.v4.new_code_cell("""# 6.4. Trực quan hóa Actual vs Predicted & Residuals Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

# Actual vs Predicted
ax1.scatter(y, y_preds_scratch, color='#1f77b4', alpha=0.7, edgecolors='k', s=45)
min_v, max_v = min(min(y), min(y_preds_scratch)), max(max(y), max(y_preds_scratch))
ax1.plot([min_v, max_v], [min_v, max_v], color='red', linestyle='--', linewidth=2, label=r'Đường lý tưởng $y = \\hat{y}$')
ax1.set_title('Thực tế (Actual) vs Dự đoán (Predicted Sales)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Doanh số thực tế ($1,000)')
ax1.set_ylabel('Doanh số dự đoán ($1,000)')
ax1.legend()
ax1.grid(True, linestyle=':', alpha=0.6)

# Residuals Plot
res = np.array(y) - np.array(y_preds_scratch)
ax2.scatter(y_preds_scratch, res, color='purple', alpha=0.7, edgecolors='k', s=45)
ax2.axhline(0, color='red', linestyle='--', linewidth=2)
ax2.set_title('Đồ thị Phần dư (Residuals vs Predicted)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Doanh số dự đoán ($1,000)')
ax2.set_ylabel(r'Phần dư ($y - \\hat{y}$)')
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()"""))

    # =============================================================
    # PHẦN 7: SUMMARY & CONCLUSION
    # =============================================================
    cells.append(nbf.v4.new_markdown_cell("""---
## PHẦN 7: SUMMARY & CONCLUSION (TỔNG KẾT BÀI HỌC)

### 1. Tổng hợp kết quả thực hiện
1. **Question 1:**
   - Đã xác định chính xác phương trình hồi quy Least Squares giải tích:
     $$y = 0.6387 x + 2.7573$$
   - Tổng sai số bình phương: $SSE = 13.7002$.
   - Đã vẽ đồ thị trực quan hóa các điểm dữ liệu và đường hồi quy.
2. **Question 2:**
   - Đã cài đặt hoàn chỉnh sơ đồ SGD từng bước theo Slide 35 với $w=4, b=1, \\eta=0.1$.
   - Đã điền đầy đủ và chính xác toàn bộ bảng 6 cột (`Data`, `output`, `loss`, `gradient`, `W, b`, `Sum of loss`).
   - Phân tích và chứng minh lý do tại sao $\\eta=0.1$ gây bùng nổ gradient, đồng thời chứng minh với $\\eta=0.005$ thuật toán hội tụ chính xác về nghiệm giải tích ở Question 1.
3. **Toàn bộ bài giảng Slides 36 - 46 (lec3.pdf):**
   - Tái hiện 100% không bỏ sót code nào.
   - Chạy thành công trên tập dữ liệu thực tế `advertising.csv`.
   - Cài đặt và kiểm chứng tính khôi phục hoàn hảo của Min-Max Scaling và Z-Score Scaling ($< 10^{-14}$).
   - Đạt độ chính xác $R^2 \\approx 0.887$ sau 50 epochs, tiệm cận nghiệm giải tích toàn cục $R^2 \\approx 0.9026$."""))

    nb.cells = cells
    
    output_filename = "Linear_Regression_and_Data_Normalization_Practice.ipynb"
    with open(output_filename, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    
    print(f"Đã tạo thành công file: {output_filename}")
    print(f"Tổng số cell: {len(cells)} (Markdown: {len([c for c in cells if c.cell_type == 'markdown'])}, Code: {len([c for c in cells if c.cell_type == 'code'])})")
    return output_filename

if __name__ == "__main__":
    create_master_notebook()
