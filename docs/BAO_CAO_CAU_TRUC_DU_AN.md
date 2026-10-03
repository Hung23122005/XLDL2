# Báo cáo cấu trúc và chức năng dự án AI Model Profiling

Tài liệu mô tả mã nguồn và dữ liệu đang có trong repo XLDL2 tại ngày 03/10/2026. Dự án mở rộng PRoof thành framework quản lý Agent, điều phối profiling và cung cấp REST API. Các bảng bên dưới dùng đường dẫn tương đối từ thư mục gốc repo.

Phạm vi gồm mã nguồn, cấu hình, model, report và tài liệu dự án. Các file bên trong môi trường `venv`, dữ liệu Git và cache Python được giải thích theo nhóm vì không phải mã nguồn nghiệp vụ. Việc có backend hoặc script trong repo không đồng nghĩa backend đó đã được cài đặt và kiểm thử trên máy hiện tại.

## Tổng quan thư mục

```text
XLDL2/
├── api/              REST API và lớp giao tiếp với tiến trình Agent
├── orchestration/    Tiếp nhận request, chọn Agent và điều phối lifecycle
├── registry/         Quản lý danh sách và metadata Agent đang hoạt động
├── agent/            Lifecycle, runtime, predictor và profiler
│   └── predictors/   Triển khai suy luận ONNX Runtime và PyTorch
├── artifact/         Quản lý trọng số, chuyển đổi và tái sử dụng artifact
├── configs/          Cấu hình môi trường thực thi của Agent
├── manifests/        Khai báo model, input, output và artifact
├── models/           Trọng số model và metadata nguồn gốc
├── reports/          Kết quả profiling JSON và báo cáo HTML đã sinh
├── model/            Lõi phân tích và backend của PRoof
│   ├── analyze/      Phân tích đồ thị, FLOPs và bộ nhớ
│   └── backend/      ONNX Runtime, TensorRT, OpenVINO, NART
├── dataviewer/       Công cụ chuyển report PRoof thành báo cáo trực quan
├── test/             Script thí nghiệm và tiện ích của PRoof
├── docs/             Tài liệu kiến trúc và báo cáo dự án
├── tmp/              File xem trước và báo cáo tạm
├── venv/             Môi trường Python cục bộ
├── __pycache__/      Cache bytecode Python
└── .git/             Lịch sử và metadata Git
```

Lưu ý: `model/` chứa **mã nguồn PRoof**, còn `models/` chứa **file model**. `registry/` quản lý Agent; `artifact/registry.py` quản lý phép chuyển đổi định dạng. Hai Registry có nhiệm vụ khác nhau.

## Luồng hoạt động

```text
HTTP POST /profile
  → api.main tạo ProfilingRequest
  → Orchestrator.execute
  → Registry tìm các Agent đúng backend/device, READY và chưa load model
  → Orchestrator chọn Agent
  → Agent.load_model
      → ArtifactManager lấy hoặc tạo artifact phù hợp
      → Predictor load artifact vào runtime
  → Agent.profile
      → ONNX Runtime: PRoofProfiler gọi main.py bằng subprocess
      → PyTorch: BasicProfiler đo các lần predict
  → Agent.unload_model
  → ProfilingResult.to_dict → HTTP response
```

API không tự chọn Agent hoặc thực hiện chuyển đổi/model inference. `ProcessAgent` chỉ chuyển lời gọi sang tiến trình con, nơi Agent thật thực thi. API xử lý tuần tự để phù hợp với Orchestrator đồng bộ hiện tại. Mỗi Agent có tiến trình riêng nhằm tránh xung đột CUDA giữa ONNX Runtime và PyTorch.

`Predictor` tái sử dụng model/session trong các lần predict. Riêng PRoof chạy ở subprocess có session profiling riêng; không nên diễn giải toàn bộ hệ thống là chỉ có một session ONNX duy nhất.

## Lớp API

| File | Chức năng |
|---|---|
| `api/__init__.py` | Khai báo package API. |
| `api/main.py` | Tạo FastAPI app; schema `ProfileBody`; startup tạo/start/register bốn Agent; shutdown stop Agent. Cung cấp `/health`, `/agents`, `/profile`; chuyển request sang Orchestrator và trả kết quả. |
| `api/runtime.py` | `ProcessAgent` chuyển lời gọi bằng Pipe tới Agent chạy trong multiprocessing spawn. Worker thực thi lifecycle thật; xử lý Ctrl+C để Uvicorn điều phối shutdown. Danh sách bốn config API sử dụng nằm ở đây. |

HTTP 400 tương ứng `ValueError`; HTTP 503 tương ứng `NoEligibleAgentError`; lỗi thực thi profiling trả HTTP 500. Lỗi cấu trúc body do FastAPI/Pydantic kiểm tra dùng cơ chế validation mặc định, thường là HTTP 422. `/docs` là Swagger tự sinh, không phải Web UI quản lý riêng.

## Điều phối và Registry

| File | Chức năng |
|---|---|
| `orchestration/__init__.py` | Xuất `Orchestrator`, `ProfilingRequest`, `ProfilingResult`. |
| `orchestration/request.py` | Dataclass request bất biến, kiểm tra trường đầu vào và tạo UUID. Dataclass result chứa Agent, model, report, artifact metadata, thời gian và duration; hỗ trợ `to_dict()`. |
| `orchestration/orchestrator.py` | Validate manifest; lấy ứng viên từ Registry; chọn Agent phù hợp; kiểm tra lại trạng thái; gọi load/profile/unload; bảo toàn lỗi workload khi cleanup cũng lỗi. |
| `orchestration/errors.py` | Định nghĩa `NoEligibleAgentError`, kế thừa `RuntimeError`, để API nhận biết trường hợp không có Agent sẵn sàng. |
| `registry/__init__.py` | Xuất `AgentRecord` và `AgentRegistry`. |
| `registry/models.py` | Snapshot metadata `AgentRecord`; kiểm tra dữ liệu; chuyển Path, datetime và enum thành dữ liệu JSON. |
| `registry/registry.py` | Lưu tham chiếu Agent trong bộ nhớ; register/unregister/get/list/find/clear; chống trùng ID; đọc trạng thái sống. `update_agent_status` xác nhận trạng thái thực tế, không ép thay đổi lifecycle. |

Registry hiện không lưu database và không điều phối máy từ xa. Agent `ERROR`, `BUSY`, `STOPPED` hoặc đã load model không được chọn cho request mới.

## Agent và suy luận

| File | Chức năng |
|---|---|
| `agent/__init__.py` | Điểm vào package Agent. |
| `agent/agent.py` | Class `Agent` và `AgentStatus`; sở hữu Predictor, Profiler, ArtifactManager. Quản lý start/load/predict/profile/unload/stop, trạng thái và metadata; có CLI chạy lifecycle. |
| `agent/config.py` | `AgentConfig` đọc YAML và kiểm tra backend/device/provider; xác định repo root từ vị trí file. |
| `agent/hardware.py` | Thu thập CPU, GPU, runtime version và provider khả dụng theo backend. |
| `agent/manifest.py` | Đọc/validate manifest, phân giải đường dẫn trong repo, kiểm tra input/output/artifact và sinh input NumPy giả. |
| `agent/predictor.py` | Alias tương thích cho cách import Predictor ONNX ở phiên bản ban đầu. |
| `agent/profilers.py` | `PRoofProfiler` gọi PRoof qua Python subprocess, kiểm tra report rồi bổ sung artifact metadata. `BasicProfiler` đo PyTorch với warmup và các lần predict; factory chọn profiler theo backend. |
| `agent/predictors/__init__.py` | Xuất các thành phần Predictor/factory. |
| `agent/predictors/base.py` | Giao diện chung cho Predictor và các thao tác lifecycle/synchronization. |
| `agent/predictors/factory.py` | Tạo Predictor đúng backend/device/provider. |
| `agent/predictors/onnxruntime_predictor.py` | Load ONNX vào InferenceSession, lấy input name từ session, tái sử dụng session cho predict và giải phóng khi unload; kiểm tra provider. |
| `agent/predictors/pytorch_predictor.py` | Load trọng số PyTorch, chuyển model/input tới thiết bị tương ứng, suy luận và đưa output về CPU; hỗ trợ đồng bộ CUDA. |
| `agent/test_lifecycle.py` | Test lifecycle và hành vi Agent/ONNX ban đầu. |
| `agent/test_backends.py` | Test cấu hình backend, trạng thái, lựa chọn Predictor và hành vi lỗi runtime. |

`BasicProfiler` đo phạm vi predict API, bao gồm chuyển input/output giữa thiết bị; kết quả không mặc nhiên tương đương phạm vi đo nội bộ của PRoof. Khi so sánh cần đọc `measurement_scope`, đơn vị và loại profiler.

## Artifact và model

| File | Chức năng |
|---|---|
| `artifact/__init__.py` | Xuất thành phần quản lý artifact. |
| `artifact/manager.py` | `ArtifactManager` tìm/tạo artifact; lưu trọng số nguồn; kiểm tra checksum và signature; tái sử dụng artifact hợp lệ; trả metadata nguồn gốc. |
| `artifact/registry.py` | `ConversionRegistry` ánh xạ cặp source → target tới converter. Mặc định có PyTorch → ONNX; không tự suy ra chuyển đổi ngược. |
| `artifact/converter.py` | Tạo/load model nguồn torchvision, export ONNX và kiểm tra output ONNX so với model nguồn PyTorch. |
| `artifact/test_artifacts.py` | Kiểm thử tạo/tái sử dụng artifact, metadata và tính nhất quán chuyển đổi. |
| `models/resnet18_v1/pytorch/resnet18.pth` | Trọng số nguồn PyTorch của logical model `resnet18_v1`. |
| `models/resnet18_v1/pytorch/resnet18.pth.json` | Metadata model ID, signature và checksum của trọng số nguồn. |
| `models/resnet18_v1/onnx/resnet18.onnx` | Model ONNX được chuyển đổi từ trọng số nguồn. |
| `models/resnet18_v1/onnx/resnet18.onnx.json` | Metadata ONNX: checksum, signature, nguồn sinh và kết quả validation. |

Manifest hiện dùng `weights: null`; không nên mô tả các trọng số này là model pretrained dùng để đánh giá độ chính xác ImageNet. Mục tiêu hiện tại là kiểm chứng thực thi và đo hiệu năng.

## Cấu hình Agent và manifest

| File | Chức năng |
|---|---|
| `configs/agent_cpu.yaml` | Config ONNX CPU cũ, ID `agent_cpu`; mặc định của `Agent()` khi không truyền config. |
| `configs/agent_gpu.yaml` | Config ONNX GPU cũ, ID `agent_gpu`. |
| `configs/agent_onnx_cpu.yaml` | Agent ONNX CPU với `CPUExecutionProvider`. |
| `configs/agent_onnx_gpu.yaml` | Agent ONNX GPU với `CUDAExecutionProvider`. |
| `configs/agent_pytorch_cpu.yaml` | Agent PyTorch chạy thiết bị CPU. |
| `configs/agent_pytorch_gpu.yaml` | Agent PyTorch chạy thiết bị CUDA. |
| `manifests/resnet18.yaml` | Manifest logical model dùng chung: ID, nguồn PyTorch, input/output và đường dẫn hai artifact. Phù hợp luồng Registry/Orchestrator/API hiện tại. |
| `manifests/resnet18_cpu.yaml` | Manifest ONNX kiểu cũ, trỏ `resnet18_clean.onnx`, report mặc định `reports/resnet18_cpu.json`. |
| `manifests/resnet18_onnx.yaml` | Manifest ONNX local riêng với input/output ResNet18. |
| `manifests/resnet18_pytorch.yaml` | Manifest PyTorch riêng, nguồn torchvision ResNet18 và weights null. |

Config mô tả **nơi và cách chạy**, manifest mô tả **model và dữ liệu**. Backend/device thực thi do AgentConfig quyết định, không dựa vào các gợi ý runtime cũ trong manifest.

## Lõi PRoof

| File | Chức năng |
|---|---|
| `main.py` | CLI PRoof; nhận model, backend, batch size, output, subjects và các tùy chọn; chạy PerfContext và xuất report. Khác với `api/main.py`. |
| `context.py` | `PerfContext`, `RooflineContext`, `ModelContext` tổ chức phân tích, benchmark model và phép đo roofline. |
| `datatype.py` | Các dataclass biểu diễn dữ liệu report, tensor shape, thông số model, layer và benchmark. |
| `util.py` | Xác định/tạo thư mục tạm PRoof từ `PROOF_TMPDIR` hoặc thư mục tạm hệ thống. |
| `model/roofline.py` | Sinh model ONNX phục vụ phép đo giới hạn tính toán/bộ nhớ roofline. |
| `model/analyze/__init__.py` | Class `Analyze`: đọc đồ thị và cung cấp truy vấn toán tử, tensor, FLOPs, memory, quan hệ trước/sau. |
| `model/analyze/model.py` | Tổng hợp phân tích layer của model ONNX thành dữ liệu report. |
| `model/analyze/op.py` | Biểu diễn toán tử và các phép tính chi phí theo loại operator. |
| `model/analyze/flops.py` | Bảng quy đổi số FLOPs cho phép toán cơ bản và phức tạp. |
| `model/analyze/graph.py` | Thông tin tensor và quan hệ producer/consumer giữa các node. |
| `model/analyze/shape.py` | Đọc/suy luận shape và thay đổi batch dimension. |
| `model/analyze/fuse.py` | Biểu diễn/ phân tích các nhóm toán tử được fusion để đối chiếu layer runtime. |
| `model/analyze/util.py` | Tiện ích chuyển shape ONNX và tìm phần tử theo tên. |
| `model/backend/__init__.py` | Giao diện backend PRoof và tra cứu backend bằng lazy import. |
| `model/backend/cache.py` | Mô phỏng cache giữa các operator để ước lượng lưu lượng bộ nhớ. |
| `model/backend/onnxruntime/__init__.py` | Backend PRoof ONNX Runtime: tạo session, benchmark end-to-end và profiling layer. |
| `model/backend/trtexec/__init__.py` | Backend TensorRT thông qua trtexec; chuẩn bị engine, đo hiệu năng và liên hệ layer với phân tích ONNX. |
| `model/backend/trtexec/ncu.py` | Tích hợp Nsight Compute, đọc chỉ số kernel để tính FLOPs và memory I/O. |
| `model/backend/trtexec/nsys_myelin.py` | Dùng Nsight Systems để phân tích các thành phần Myelin của TensorRT. |
| `model/backend/openvino/__init__.py` | Điểm vào backend OpenVINO. |
| `model/backend/openvino/backend.py` | Triển khai backend OpenVINO. |
| `model/backend/nart/__init__.py` | Backend NART; README đánh dấu tùy chọn này deprecated. |
| `model/backend/nart/_nart_run.cpp` | Phần chạy NART viết bằng C++. |
| `model/backend/nart/Makefile` | Quy tắc biên dịch runner NART. |

Các backend PRoof kể trên rộng hơn các backend Agent hiện hỗ trợ. Agent hiện có ONNX Runtime và PyTorch; không tự động có Agent TensorRT/OpenVINO chỉ vì PRoof có backend tương ứng.

## Hiển thị báo cáo

| File | Chức năng |
|---|---|
| `dataviewer/main.py` | CLI đọc report JSON PRoof rồi sinh báo cáo HTML vào thư mục output. |
| `dataviewer/gen_html.py` | Render báo cáo từ template và sao chép các asset cần thiết. |
| `dataviewer/gen_text.py` | File dự kiến sinh báo cáo text; hiện chỉ có TODO, chưa triển khai. |
| `dataviewer/read_value.py` | Trích giá trị theo đường dẫn key từ một hoặc nhiều report. |
| `dataviewer/roofline_image.py` | Script vẽ hình roofline bằng matplotlib. |
| `dataviewer/roofline_image2.py` | Biến thể vẽ roofline và phân bố thời gian/hiệu năng với nhiều biểu đồ. |
| `dataviewer/requirements.txt` | Dependencies riêng của công cụ hiển thị. |
| `dataviewer/html_templates/index.html` | Template trang tổng quan. |
| `dataviewer/html_templates/model.html` | Template thông tin và kết quả model. |
| `dataviewer/html_templates/layer-META_BS.html` | Template báo cáo layer theo batch size. |
| `dataviewer/html_templates/assets/common.css` | Style dùng chung. |
| `dataviewer/html_templates/assets/common.js` | JavaScript dùng chung cho báo cáo. |
| `dataviewer/html_templates/assets/roofline-chart.js` | JavaScript biểu đồ roofline. |

Đây là công cụ báo cáo HTML của PRoof, không phải frontend quản lý REST API vừa thêm. Report PyTorch BasicProfiler có schema khác report PRoof; không mặc nhiên đưa trực tiếp vào mọi template PRoof được.

## Report và dữ liệu sinh ra

| File hoặc nhóm file | Chức năng |
|---|---|
| `reports/resnet18_onnx_cpu.json` | Kết quả profiling ResNet18 qua ONNX Runtime CPU. |
| `reports/resnet18_onnx_gpu.json` | Kết quả profiling ResNet18 qua ONNX Runtime GPU. |
| `reports/resnet18_pytorch_cpu.json` | Kết quả BasicProfiler trên PyTorch CPU. |
| `reports/resnet18_pytorch_gpu.json` | Kết quả BasicProfiler trên PyTorch GPU. |
| `reports/resnet18_cpu.json` | Report của luồng ONNX CPU ban đầu. |
| `reports/resnet18_html/index.html` | Trang tổng quan báo cáo HTML đã sinh. |
| `reports/resnet18_html/model.html` | Trang model đã sinh. |
| `reports/resnet18_html/layer-1.html` | Trang layer cho batch size 1. |
| `reports/resnet18_html/assets/common.css` | CSS được sao chép cho báo cáo. |
| `reports/resnet18_html/assets/common.js` | JavaScript dùng chung của báo cáo. |
| `reports/resnet18_html/assets/roofline-chart.js` | JavaScript biểu đồ của báo cáo. |
| `report_clean.json`, `report_test.json`, `report_resnet18.json`, `report_resnet18_gpu.json` | Các report lưu tại root từ các lần thử trước. Không dùng tên file để kết luận đó là kết quả mới nhất; cần đối chiếu nội dung và lần chạy. |
| `resnet18.onnx`, `resnet18_clean.onnx` | Model ONNX tại root phục vụ các luồng thử/manifest cũ. |
| `resnet18_clean.onnx.data` | File dữ liệu trọng số external ONNX đi kèm khi model tham chiếu tới nó. Không coi là file cache có thể xóa tùy ý. |
| `tmp/compare_agents_preview.png` | Ảnh xem trước của notebook so sánh. |
| `tmp/report/index.html`, `tmp/report/model.html`, `tmp/report/layer-1.html` | Bản báo cáo HTML trong thư mục tạm. |
| `tmp/report/assets/common.css`, `tmp/report/assets/common.js`, `tmp/report/assets/roofline-chart.js` | Các asset đi kèm bản báo cáo tạm. |

Report có thể bị thay thế khi chạy lại cùng đường dẫn. `duration_ms` của ProfilingResult là thời gian luồng load/profile/cleanup, không phải latency một lần inference.

## File hỗ trợ ở thư mục gốc

| File | Chức năng |
|---|---|
| `README.md` | Hướng dẫn PRoof gốc: mục đích, cài đặt, các backend và sử dụng. |
| `requirements.txt` | Dependencies của repo, đã bổ sung FastAPI và Uvicorn; runtime CPU/GPU vẫn phụ thuộc môi trường cài đặt thực tế. |
| `.gitignore` | Quy định file/thư mục không đưa vào Git. |
| `ort_runtime.py` | Chuẩn bị DLL CUDA cho ONNX Runtime trong tiến trình hiện tại, không sửa PATH hệ thống. |
| `export_resnet18.py` | Script export torchvision ResNet18 sang `resnet18_clean.onnx`, opset 13 và dynamic batch. |
| `demo_registry.py` | Start/register các Agent từ config, in metadata Registry và dừng Agent để kiểm tra trạng thái. |
| `demo_orchestrator.py` | Demo một request hoặc `--all`; bốn trường hợp được chạy trong subprocess riêng; kiểm tra report và kết quả. |
| `test_agents.py` | Script chạy bốn môi trường Agent và báo lỗi từng trường hợp. |
| `test_registry.py` | Unit test Registry và FakeAgent dùng chung cho một số test. |
| `test_orchestrator.py` | Unit test request/result, lựa chọn Agent, cleanup, lỗi và interruption. |
| `test_api.py` | Test HTTP bằng TestClient, Registry/Orchestrator thật và fake Agent; kiểm tra response, mã lỗi, startup/shutdown. Cần thư viện httpx trong môi trường test hiện tại. |
| `compare_agents.ipynb` | Notebook so sánh các môi trường Agent và kết quả profiling; là công cụ phân tích, không nằm trên luồng API. |

## Các script thí nghiệm PRoof

Thư mục `test/` chủ yếu là script thí nghiệm, export và phân tích của PRoof. Khác với các unit test `test_api.py`, `test_registry.py` và `test_orchestrator.py` ở root; không nên chạy toàn bộ script thử nghiệm như một bộ unit test thông thường.

| File | Chức năng |
|---|---|
| `test/bert-export.py` | Thử nghiệm export/chạy model BERT với ONNX. |
| `test/timm_export_single.py` | Export một model từ timm. |
| `test/timm_export_all.py` | Export nhiều model từ timm. |
| `test/onnx_create_model.py` | Tạo model ONNX thử nghiệm. |
| `test/onnx_shape_infer.py` | Thử suy luận shape và thay đổi batch dimension ONNX. |
| `test/onnxruntime_test.py` | Thử chạy trực tiếp ONNX Runtime. |
| `test/nart_basic.py` | Thử nghiệm NART cơ bản. |
| `test/trt_inspector.py` | Xem thông tin layer của engine TensorRT. |
| `test/trt_img2dat.py` | Chuẩn bị dữ liệu ảnh đầu vào cho thử nghiệm TensorRT. |
| `test/trt_output_to_label.py` | Chuyển output phân loại TensorRT sang nhãn. |
| `test/data/imagenet_classes.txt` | Danh sách nhãn ImageNet dùng cho giải mã output. |
| `test/data/cinema.jpg` | Ảnh mẫu phục vụ thử nghiệm đầu vào phân loại. |
| `test/data/drum.jpg` | Ảnh mẫu phục vụ thử nghiệm đầu vào phân loại. |
| `test/data/goldfish.jpg` | Ảnh mẫu phục vụ thử nghiệm đầu vào phân loại. |
| `test/model/narttrt-config-fp16.json` | Cấu hình NART/TensorRT cho thực thi FP16. |
| `test/util.py` | Tiện ích in cấu trúc dictionary kèm kiểu dữ liệu. |
| `test/helper_table1.py` | Script hỗ trợ tổng hợp bảng kết quả thí nghiệm. |
| `test/helper_fig2_single.py` | Script hỗ trợ vẽ hình so sánh roofline trên các phần cứng. |
| `test/helper_figure3.py` | Script hỗ trợ vẽ hình kết quả thí nghiệm. |
| `test/orin-nx-nvpmodel.conf` | Cấu hình chế độ nguồn cho thiết bị Orin NX. |
| `test/expr-4.2.sh` | Kịch bản thí nghiệm nhóm 4.2, có tích hợp TensorRT/Nsight Compute. |
| `test/expr-4.4.sh` | Kịch bản thí nghiệm nhóm 4.4 của PRoof. |
| `test/expr-4.3-4090-fp16.sh` | Kịch bản RTX 4090, FP16. |
| `test/expr-4.3-4090-int8.sh` | Kịch bản RTX 4090, INT8. |
| `test/expr-4.3-a100-fp16.sh` | Kịch bản A100, FP16. |
| `test/expr-4.3-a100-int8.sh` | Kịch bản A100, INT8. |
| `test/expr-4.3-intel-xeon-gold-6330.sh` | Kịch bản Intel Xeon Gold 6330. |
| `test/expr-4.3-orin-nx-fp16.sh` | Kịch bản Orin NX, FP16. |
| `test/expr-4.3-orin-nx-int8.sh` | Kịch bản Orin NX, INT8. |
| `test/expr-4.3-raspberry-pi-4b.sh` | Kịch bản Raspberry Pi 4B. |
| `test/expr-4.3-ultra9-185h-npu-fp16.bat` | Kịch bản NPU Intel Core Ultra 9 185H, FP16. |
| `test/expr-4.3-ultra9-185h-npu-int8.bat` | Kịch bản NPU Intel Core Ultra 9 185H, INT8. |
| `test/expr-4.3-xavier-nx-fp16.sh` | Kịch bản Xavier NX, FP16. |
| `test/expr-4.3-xavier-nx-int8.sh` | Kịch bản Xavier NX, INT8. |

Các tên phần cứng/precision trong bảng phản ánh mục tiêu của script, không phải danh sách phần cứng đã được kiểm thử trong phiên phát triển hiện tại.

## Tài liệu và thư mục hệ thống

| File hoặc thư mục | Chức năng |
|---|---|
| `docs/BAO_CAO_CAU_TRUC_DU_AN.md` | Tài liệu tổng hợp cấu trúc và chức năng đang đọc. |
| `docs/superpowers/specs/2026-10-03-registry-orchestrator-design.md` | Thiết kế Registry/Orchestrator tại thời điểm lập spec; có thể không chứa mọi bổ sung triển khai sau đó. |
| `docs/superpowers/plans/2026-10-03-registry-orchestrator.md` | Kế hoạch triển khai và ghi nhận kiểm chứng Registry/Orchestrator. |
| `venv/` | Python executable, scripts và các thư viện cài cục bộ; không phải thành phần nghiệp vụ tự viết. |
| `.git/` | Dữ liệu quản lý phiên bản; không chỉnh sửa như mã nguồn ứng dụng. |
| `__pycache__/` và các thư mục cùng tên trong package | File `.pyc` do Python tự sinh để tăng tốc import. |

## Cách chạy các thành phần chính

Từ thư mục gốc repo trên PowerShell:

```powershell
# REST API
.\venv\Scripts\python.exe -m uvicorn api.main:app --reload

# Demo Registry
.\venv\Scripts\python.exe demo_registry.py

# Demo bốn môi trường qua Orchestrator
.\venv\Scripts\python.exe demo_orchestrator.py --all --manifest manifests/resnet18.yaml

# Agent ONNX CPU độc lập
.\venv\Scripts\python.exe agent\agent.py --agent-config configs/agent_onnx_cpu.yaml --manifest manifests/resnet18.yaml

# Bộ unit test framework
.\venv\Scripts\python.exe -m unittest test_api test_registry test_orchestrator agent.test_lifecycle agent.test_backends artifact.test_artifacts -q
```

Swagger nằm tại `http://127.0.0.1:8000/docs`. API hiện có `GET /health`, `GET /agents` và `POST /profile`. Framework chưa có database, Web UI quản lý, hàng đợi job, điều phối phân tán hoặc scheduler chạy song song.
