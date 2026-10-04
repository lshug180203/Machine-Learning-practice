# Machine Learning Practice & Coursework 🚀

Tài liệu học tập, thực hành và bài tập môn Machine Learning được tổ chức khoa học theo từng chủ đề bài giảng (Lectures), Google Colab thực hành, bộ dữ liệu (Datasets) và mã nguồn tiện ích (Scripts).

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
