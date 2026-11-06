# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=… ; `class_name`=… ; `rank`=1 ; `score`=… ; `taxonomy_name`=…
- `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K

2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
- Record này nói về cả bức ảnh. Vì đây là tác vụ Classification (Phân loại ảnh), mô hình gán một nhãn tổng thể (global label) mô tả chủ đề chính hoặc ngữ cảnh chung cho toàn bộ khung hình, chứ không xác định vị trí của từng vật thể riêng lẻ.

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
- Danh sách tên lớp (class list) đi kèm theo taxonomy/checkpoint.

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
- class_id: Dùng cho máy tính xử lý (lưu trữ dạng số, tối ưu bộ nhớ/tốc độ).
class_name: Dùng cho con người đọc hiểu trực quan.
taxonomy_name: Xác định không gian tên (namespace) và bộ dữ liệu chuẩn mà ID/tên lớp đó thuộc về.

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
- Guideline phải định nghĩa rõ quy tắc ưu tiên (priority hierarchy) hoặc tiêu chí xác định chủ thể chính (dominant subject)

6. Vì sao không được lấy `score` của model làm ground truth?
- Vì chỉ thể hiện độ tự tin (confidence degree) mang tính thống kê của mô hình toán học tại thời điểm hiện tại, không đảm bảo tính đúng đắn tuyệt đối. Ground truth phải đại diện cho thực thể khách quan (truth) do con người kiểm chứng và xác nhận theo guideline.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…
- `class_name`=bus ; `score`=0.912557 ; `bbox_xyxy`=[93.17, 187.95, 223.01, 320.91] ; `bbox_width`=129.84 ; `bbox_height`=132.96

2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
- Góc trên-trái (Top-Left): Nằm tại tọa độ X = 93.17(px) và Y = 187.95(px). 
Góc dưới-phải (Bottom-Right): Nằm tại tọa độ X = 9223.01(px) và Y = 320.91(px).
rộng=129.84(223.01-98.17) pixel.
cao=132.96(320.91-187.95) pixel

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
- Ngưỡng yolo11n.pt: 0.2; số vật thể phát hiện: 80.
Ngưỡng yolo11n.pt: 0.6; số vật thể phát hiện: 25.

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
- Công việc phát sinh của Reviewer khi hạ ngưỡng confidence: Khi hạ ngưỡng (nhận nhiều box hơn), reviewer phải tốn nhiều thời gian và công sức hơn để lọc bỏ các dự đoán sai (false positives/báo động giả), xóa các box trùng lặp/chồng lấp quá mức, và chỉnh sửa lại ranh giới box cho ôm sát vật thể.

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
- Khung hình chữ nhật phải bao phủ toàn bộ phần thấy được của vật thể (Tight Bounding Box), sát viền ngoài cùng tối đa có thể, không bỏ sót phần viền và hạn chế thừa vùng trống không thuộc về vật thể.

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
- bowl 46 bị che khuất hoặc cắt mép.
"Vẽ box ôm lấy phần nhìn thấy nếu bị che > 20%, hoặc mở rộng box ước lượng nếu bị che nhẹ"). Ngược lại, nếu vật thể bị che khuất quá dị biệt, không thể nhận diện rõ ranh giới, sẽ đưa lên escalation (hỏi Lab Coach) để thiết lập tiền lệ cho cả đội gán nhãn.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…
- `instance_id`=traffic-002 ; `class_name`=car ; `score`=0.891092 ; số điểm polygon=`polygon_point_count`=92 ; 3–5 điểm đầu của `polygon_xy`=(497.0, 287.0), (497.0, 287.0), (492.0, 287.0), (491.0, 287.0), (488.0, 284.0).

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
- Polygon cung cấp chi tiết ranh giới hình học chính xác (pixel-level contour) của vật thể, phân biệt rõ phần thuộc về đối tượng và phần nền, giải quyết triệt để các góc rỗng/vùng thừa mà khung hình chữ nhật (Bounding Box) buộc phải bao bọc.

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
- Nó không phải class ID vì class ID dùng để phân loại bản chất vật thể (ví dụ: tất cả các ghế đều chung class_id = chair).
Nó không phải tracking ID vì nó không có giá trị liên kết theo thời gian qua nhiều frame video khác nhau, chỉ có giá trị duy nhất trong bức ảnh hiện tại.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
- "Đường viền polygon phải bám sát theo ranh giới thực tế của vật thể với độ chính xác đến từng pixel, loại bỏ hoàn toàn phần nền lọt vào giữa các điểm nối và không kéo đường thẳng cắt ngang qua thân vật thể."

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
- Ranh giới mờ giữa chai lọ và bóng của nó trên mặt bàn đá.
Quyết định: Dùng Guideline nếu trong guideline đã có định nghĩa rõ ràng về việc loại bỏ vùng bóng đổ (shadow). Dùng Escalation nếu vật thể và môi trường xung quanh hòa lẫn vào nhau khiến mắt thường không thể phân định đâu là mép thật của vật thể.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một chuỗi văn bản/mã đại diện cho nhãn của cả ảnh (class_id, class_name). | Ảnh chứa nhiều chủ đề có độ phân giải ngang nhau. | Xem toàn bộ ảnh và chọn 1 nhãn phù hợp nhất theo quy tắc ưu tiên trong guideline. | Kiểm tra nhãn được chọn có vi phạm quy tắc ưu tiên không; đánh giá tính nhất quán giữa các ảnh. |
| Phát hiện vật thể | Bảng danh sách các ô hình chữ nhật kèm nhãn: [class_id, x_min, y_min, x_max, y_max] | Box bị vẽ rộng thừa nền, bỏ sót đối tượng nhỏ ở xa, hoặc box bị trùng lặp (duplicate) | Kéo thả khung chữ nhật bao bọc vừa khít từng vật thể và gán đúng nhãn cho từng box. | Kiểm tra độ ôm sát của box, phát hiện box bị thừa/thiếu, kiểm tra nhãn gán đúng loại chưa. |
| Instance segmentation | Danh sách các tọa độ điểm tạo thành đường cong kín gán cho từng cá thể: [instance_id, class_id, [x1, y1, x2, y2, ...]]| Đường viền polygon bị nét cắt thô (thiếu điểm), lọt vùng nền vào trong mask, hoặc vật bị che dính mask vào nhau. | Chấm các điểm tọa độ bao quanh chính xác ranh giới của từng vật thể độc lập. | Kiểm tra độ mịn/chính xác của đường biên mask, đảm bảo phân tách đúng từng instance riêng biệt. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
- Không lưu trữ, chia sẻ hay công khai dữ liệu của bài lab ra bên ngoài phạm vi môi trường thực hành được quy định, đồng thời cam kết xóa bỏ mọi thông tin cá nhân trước khi nộp sản phẩm. 

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
- Khi gặp dữ liệu nhạy cảm, vi phạm điều khoản hoặc nằm ngoài phạm vi bài lab, tôi sẽ lập tức dừng làm việc với file đó và báo cáo trực tiếp cho Lab Coach / Giảng viên.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ x] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [x ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS