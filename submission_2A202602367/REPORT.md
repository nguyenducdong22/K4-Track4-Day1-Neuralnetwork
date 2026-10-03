# Báo cáo Lab Day 1 — <Họ tên> — <MSSV>

> Tối đa khoảng 4 trang. **Chỉ viết cho những chủ đề bạn đã thử**; xoá các mục không dùng. Mọi con số phải trỏ về một `exp_id` trong `experiments.xlsx`. Mọi "A tốt hơn B" phải cân nhắc độ nhiễu seed (mục 2). Xoá các dòng hướng dẫn `>` khi nộp.

## 1. Thiết lập

- Môi trường (Colab/Kaggle, GPU, phiên bản PyTorch):
- Dữ liệu: Forest CoverType; `train` 464 809 / `eval` 116 203 theo `split_metadata.csv`. Validation: 20% của train (phân tầng, seed 42) → 371 847 train / 92 962 val.
- Model: `M-base` (54→256→128→7, 47 879 tham số). Baseline: loss, optimizer, lr, batch, epochs, init.
- Mốc tham chiếu: accuracy "đoán lớp đa số" trên val = ___ (≈ 0,4876).
- Các chủ đề đã thử: ☐ loss ☐ optimizer ☐ hyper-parameter ☐ dropout ☐ clipping ☐ mixed precision ☐ init

## 2. Kiểm tra ban đầu và độ nhiễu

| Kiểm tra | Kết quả |
|---|---|
| Số tham số / shape logits | ___ / (B, 7) |
| Loss bước 0 (so với ln 7 = 1,946) | ___ |
| Quá khớp 20 mẫu: loss cuối | ___ |
| Mọi tham số có gradient khác 0 | ☐ có |
| Baseline, số seed đã chạy | ___ |
| Baseline: val acc (TB ± σ) | ___ ± ___ |
| Baseline: val macro-F1 (TB ± σ) | ___ ± ___ |

**Ngưỡng nhiễu dùng trong báo cáo:** 2σ = ___ (val macro-F1). *(Nếu chỉ có 1 seed: ghi rõ không đo được nhiễu và đây là hạn chế.)*

## 3. Kết quả theo chủ đề

> Mỗi chủ đề đã thử: (a) dự đoán trước khi chạy, (b) kết quả (số + ảnh + `exp_id`), (c) giải thích cơ chế, (d) khác biệt có vượt nhiễu không. Tất cả dựa trên **val**, không dựa trên eval.

### 3.1 Hàm mất mát — CE vs MSE
- Dự đoán:
- Kết quả (`exp_id`; ảnh):
- Giải thích (gradient của CE vs MSE; chú ý không so loss trực tiếp):

### 3.2 Bộ tối ưu hoá
- Dự đoán:
- Bảng nhỏ: mỗi bộ tối ưu ở lr tốt nhất của nó (exp_id, lr, val macro-F1, best epoch):
- Độ nhạy với lr (bộ nào ổn định hơn? lr nào gây dao động/chậm?) — ảnh chồng:
- Giải thích:

### 3.3 Hyper-parameter
- Yếu tố đã đổi (lr/batch/độ rộng/độ sâu/weight decay/epoch). Có tính đến số bước cập nhật và thời gian mỗi epoch:

### 3.4 Dropout
- Khoảng cách train–val loss thay đổi ra sao? Mô hình của bạn có thực sự đang quá khớp không?

### 3.5 Gradient clipping
- Ở lr bình thường, clipping có kích hoạt không (dựa trên `grad_norm`)?
- Ở lr cao: clipping có cứu được huấn luyện không?

### 3.6 Mixed precision
- Thời gian/epoch, bộ nhớ cực đại, độ chính xác: FP32 vs FP16 vs BF16.
- Giải thích kết quả (kể cả khi không nhanh hơn).

### 3.7 Khởi tạo tham số
- Độ lệch chuẩn kích hoạt theo lớp và loss bước 0 của các cách khởi tạo.
- Giải thích vì sao `zeros` (và `normal` nếu có) cho kết quả như vậy.

## 4. Đánh giá cuối trên tập eval

> Chỉ làm sau khi chọn cấu hình bằng val. Số lấy từ `eval_result.json` (do `scripts/evaluate.py` tạo), không tự tính lại.

| Cấu hình | Seed nộp | val macro-F1 | **eval macro-F1** | eval accuracy |
|---|---|---|---|---|
| Baseline | | | | |
| Cấu hình cuối cùng | | | | |

- Cấu hình cuối cùng gồm những gì và vì sao (chọn dựa trên val)?
- Cải thiện so với baseline trên eval có vượt nhiễu không? (nếu chạy nhiều seed: trung bình ± σ của eval macro-F1)
- Val và eval có gần nhau không? Nếu lệch nhiều, nghĩ vì sao.

### 4.1 Phân tích lỗi theo lớp

| Lớp | support | precision | recall | F1 |
|---|---|---|---|---|
| 0 | | | | |
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |
| 6 | | | | |

- Lớp khó nhất là lớp ___ (F1 = ___). Nó hay bị nhầm với lớp ___ (xem ma trận nhầm lẫn; chèn ảnh nếu có).
- Lý giải (số mẫu ít? đặc trưng giống lớp khác? mất cân bằng?) và một cách cải thiện bạn sẽ thử.

## 5. Trả lời các câu hỏi dẫn dắt

> Trả lời những câu liên quan tới chủ đề bạn đã thử; câu 6 luôn bắt buộc.

1. **Bộ tối ưu nào "thắng" khi mỗi cái được chỉnh lr công bằng? Khi lr không được chỉnh thì kết luận thay đổi ra sao?**
2. **Dropout có giúp không khi mô hình chưa quá khớp? Khi nào thì nên dùng?**
3. **Gradient clipping giải quyết vấn đề gì? Quan sát nào của bạn chứng minh điều đó?**
4. **Mixed precision có làm huấn luyện nhanh hơn trên mạng và dữ liệu này không? Vì sao (không)?**
5. **Vì sao khởi tạo toàn số 0 hỏng? Khởi tạo He khác Xavier ở điểm nào và khi nào điều đó quan trọng?**
6. **Quay lại câu hỏi của bài học:** một mạng có loss không giảm sau 2 000 bước. Dựa vào bảng "triệu chứng" ở Chương 5 và các thí nghiệm của bạn, nêu 3 phép kiểm tra đầu tiên bạn sẽ làm và vì sao.

## 6. Hạn chế và điều bất ngờ

- Kết quả nào khác với dự đoán của bạn? Vì sao?
- Điều gì trong thiết kế thí nghiệm có thể làm kết luận sai (ví dụ số seed ít, cùng số epoch nhưng khác số bước, lr chưa tối ưu)?
- Nếu có thêm thời gian, bạn sẽ chạy gì tiếp?

## 7. Phụ lục

- Danh sách file đã nộp.
- Thời gian chạy ước tính tổng cộng.
