# Machine Learning Practice & Coursework 🚀

Tài liệu học tập, thực hành và bài tập môn Machine Learning được tổ chức khoa học theo từng chủ đề bài giảng (Lectures), Google Colab thực hành, bộ dữ liệu (Datasets) và mã nguồn tiện ích (Scripts).

---

## 📁 Cấu Trúc Thư Mục Dự Án (Repository Structure)

```text
Machine Learning/
├── Cheasheet/                                              # Cheatsheet tóm tắt kiến thức Machine Learning
│   ├── cheatsheet ML.png
│   └── super-cheatsheet-machine-learning.pdf
│
├── dataset/                                                # Toàn bộ tập dữ liệu (CSV) thực hành
│   ├── advertising.csv                                     # Dataset doanh số & ngân sách quảng cáo
│   ├── pwt_subset.csv                                      # Penn World Table trích xuất
│   └── test_pwt.csv                                        # Dữ liệu kinh tế vĩ mô Penn World Table
│
├── Google Colab/                                           # Bài giảng & Thực hành Python, Pandas, Matplotlib, EDA
│   ├── Data Mining Project - EDA on E-Commerce Behaviour Data.ipynb
│   ├── Lect 2 Google Colab.pdf
│   ├── Lect 2 Python.pdf
│   ├── Matplotlib.pdf
│   ├── Pandas.pdf
│   ├── dataset(in).csv
│   ├── dataset(in).xlsx
│   ├── matplotlib_all_examples.ipynb                      # Toàn bộ bài thực hành và ví dụ trực quan hóa Matplotlib
│   └── pandas_all_examples.ipynb                          # Toàn bộ ví dụ xử lý dữ liệu với Pandas
│
├── Lec1 Introduction/                                      # Bài mở đầu tổng quan
│   └── lec1.pdf
│
├── Lec2 Supervised learning/                               # Bài giảng và Thực hành Học có giám sát (Supervised Learning)
│   ├── 2_1_Supervised_Learning.ipynb                       # Thực hành Supervised Learning cơ bản
│   ├── Linear_Regression_and_Data_Normalization_Advertising.ipynb # Bài tập Hồi quy đa biến trên advertising.csv
│   ├── Linear_Regression_and_Data_Normalization_Practice.ipynb    # Bài tập Hồi quy tuyến tính câu 1 & 2
│   └── Supervised learning.pdf                             # Slide bài giảng Supervised Learning
│
├── scripts/                                                # Các kịch bản Python hỗ trợ tạo, kiểm tra & xuất notebook
│   ├── build_complete_notebook.py
│   ├── build_q1_q2_notebook.py
│   ├── create_pandas_notebook.py
│   ├── execute_and_verify_notebook.py
│   ├── generate_final_notebook.py
│   ├── rerun_and_export_pdfs.py
│   └── verify_linear_regression_notebook.py
│
├── .gitignore                                              # Cấu hình bỏ qua file tạm, môi trường ảo & file build
├── README.md                                               # Hướng dẫn dự án
└── requirements.txt                                        # Các thư viện phụ thuộc Python
```

---

## 🛠 Hướng Dẫn Cài Đặt Môi Trường (Setup)

1. **Khởi tạo và kích hoạt môi trường ảo (Virtual Environment):**
   ```bash
   python -m venv .venv
   # Trên Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   # Trên Linux/macOS:
   source .venv/bin/activate
   ```

2. **Cài đặt các gói thư viện cần thiết:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Khởi chạy Jupyter Notebook hoặc JupyterLab:**
   ```bash
   jupyter notebook
   ```

---

## 📚 Tóm Tắt Nội Dung Chính

- **Google Colab & Nền Tảng Khoa Học Dữ Liệu:** Làm quen với môi trường Colab, các cấu trúc dữ liệu với `Pandas` (Series, DataFrame, Indexing, GroupBy, Reshaping, Time Series) và vẽ biểu đồ trực quan hóa dữ liệu với `Matplotlib`.
- **Học Có Giám Sát (Supervised Learning):**
  - Hồi quy tuyến tính đơn biến (Simple Linear Regression) & Đa biến (Multiple Linear Regression).
  - Thuật toán Gradient Descent (Batch Gradient Descent) và giải thuật Closed-form Solution (Normal Equation).
  - Chuẩn hóa dữ liệu (Feature Scaling): Min-Max Normalization và Z-Score Standardization cùng quy trình nghịch đảo (Inverse Scaling).
- **Bộ Dữ Liệu Thực Tế:** Ứng dụng phân tích và mô hình hóa trên tập dữ liệu `advertising.csv` và dữ liệu chuỗi thời gian `test_pwt.csv`.
