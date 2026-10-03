# Thiết kế Registry và Orchestrator

Ngày: 2026-10-03. Phạm vi: điều phối profiling cục bộ, đồng bộ, in-memory trên framework PRoof hiện tại.

## Mục tiêu và tiêu chí thành công

Thêm lớp nhận request và chọn execution environment mà không gắn một Agent với một model cố định. Một request chọn đúng backend, device và Agent READY; Agent tiếp tục dùng ArtifactManager, Predictor và profiler hiện có. Kết quả trả về chỉ ra Agent, logical model, artifact và report thực tế. Không tự fallback khi runtime lỗi.

Không triển khai REST API, Web UI, database, remote Agent, lịch chạy, thực thi đồng thời, load balancing hoặc converter mới. Code và model cũ được giữ. Chạy trực tiếp trong repo hiện tại; không đưa thay đổi đang có của người dùng vào phạm vi refactor này.

## Hiện trạng ảnh hưởng đến thiết kế

- Agent có get_info(), get_status(), supports(); load_model và profile tự quản lý BUSY/READY/ERROR.
- Agent.unload_model() hiện reset ERROR về READY khi runtime đã start. Cần tùy chọn giữ ERROR trong cleanup của Orchestrator, nhưng giữ mặc định cũ cho caller hiện tại.
- get_info() dùng hardware; Registry sẽ ánh xạ sang hardware_info trong record, không buộc đổi tên field cũ.
- PyTorch CUDA 13 và ONNX CUDA 12 có thể xung đột DLL trong một tiến trình. Demo benchmark cần tiến trình riêng theo backend/case; không coi Registry in-memory là cơ chế cách ly runtime.
- Manifest logical model có source pytorch vẫn hợp lệ cho ONNX qua ArtifactManager. Orchestrator không dùng so sánh framework trực tiếp để từ chối conversion hợp lệ.

## Các module và trách nhiệm

| Module | Trách nhiệm |
| --- | --- |
| registry/models.py | AgentRecord, metadata snapshot có to_dict() |
| registry/registry.py | AgentRegistry, lưu tham chiếu Agent theo ID và truy vấn metadata hiện tại |
| orchestration/request.py | ProfilingRequest và ProfilingResult, validation và JSON-compatible serialization |
| orchestration/orchestrator.py | Chọn Agent và điều phối load/profile/unload |
| demo_registry.py | Start, đăng ký, liệt kê và stop các cấu hình demo |
| demo_orchestrator.py | Gửi request và tổng hợp kết quả bốn case trong các tiến trình riêng |

Mỗi package có __init__.py export public API. Thêm test riêng cho Registry và Orchestrator. Thay đổi Agent chỉ ở cleanup lỗi và bổ sung capabilities.model_formats.

## Registry

Registry dùng dict giữ thứ tự đăng ký: agent_id -> Agent. Record được tạo mới từ agent.get_info() khi đọc; không lấy snapshot cũ làm nguồn trạng thái. Dict/list metadata trả về là bản sao để caller không sửa nội bộ Agent.

AgentRecord gồm agent_id, backend, device, provider, status, hardware_info, capabilities và model_loaded. capabilities.model_formats lấy từ metadata Agent, ví dụ ['onnx'] hoặc ['pytorch']; không đoán theo tên agent_id. Giữ các capability hiện có.

API:

- register_agent(agent) -> AgentRecord: kiểm tra các field từ get_info(); trùng ID raise ValueError, không thay Agent âm thầm.
- unregister_agent(agent_id) -> None: bỏ đăng ký, không stop Agent; ID không tồn tại raise KeyError.
- get_agent(agent_id) -> Agent: trả instance để Orchestrator gọi; ID không tồn tại raise KeyError.
- list_agents() -> list[AgentRecord]: snapshot hiện tại theo thứ tự đăng ký.
- find_agents(backend=None, device=None, status=None) -> list[AgentRecord]: match chính xác; không filter thêm nếu giá trị None. Status được kiểm tra theo AgentStatus.
- update_agent_status(agent_id, status) -> AgentRecord: đồng bộ metadata với trạng thái Agent thực tế. Giá trị không hợp lệ hoặc không trùng get_status() bị từ chối; không cho Registry tự biến Agent STOPPED thành READY. API này dùng để xác nhận/refresh sau lifecycle, không duy trì state machine thứ hai.
- clear() -> None: bỏ mọi đăng ký; caller vẫn chịu trách nhiệm stop Agent.

Agent không tự biết Registry. Demo/caller gọi start rồi register; sau stop record sẽ thể hiện STOPPED vì metadata đọc trực tiếp. Khi start lỗi, demo báo lỗi rõ và không đăng ký Agent đó như READY.

## Request và result

ProfilingRequest là dataclass bất biến gồm:

- manifest_path: str hoặc pathlib.Path, bắt buộc và không rỗng.
- backend, device: string không rỗng; backend được match với Registry, không hard-code danh sách trong Orchestrator.
- action: mặc định 'profile'; mọi action khác bị từ chối trước khi chọn/load Agent.
- report_path: str hoặc pathlib.Path hoặc None, không nhận chuỗi rỗng.

Ví dụ:

```python
ProfilingRequest(
    manifest_path='manifests/resnet18.yaml',
    backend='onnxruntime',
    device='gpu',
    action='profile',
    report_path='reports/resnet18_onnx_gpu.json',
)
```

execute nhận ProfilingRequest; validation chạy trước mọi workload. Resolve manifest theo quy tắc repo root của Manifest hiện tại. Report override tiếp tục dùng Agent.profile(output_path=...).

ProfilingResult gồm agent_id, backend, device, action, model_id, report_path và artifact metadata. Sao chép metadata trước unload. to_dict() chuyển Path thành string để dùng được với json.dumps và API sau này. Chỉ trả result sau khi profile và cleanup đều thành công; không tạo report giả khi lỗi.

## Selection và flow execute

1. Validate request và đọc/validate Manifest trước khi thay đổi Agent.
2. Query find_agents(backend=request.backend, device=request.device, status='READY').
3. select_agent(candidates, request) là method riêng; chọn record đầu tiên có model_loaded=False. Agent đang giữ model của caller không được unload hoặc chiếm dụng.
4. Không có ứng viên đủ điều kiện: raise RuntimeError có backend, device và yêu cầu READY/unloaded. Không fallback backend/device.
5. Lấy instance bằng get_agent(); xác nhận lại READY và chưa load trước khi bắt đầu. Không tuyên bố thread-safe hoặc atomic reservation ở phiên bản đồng bộ này.
6. Gọi agent.load_model(manifest); artifact resolution/conversion hoàn toàn do Agent và ArtifactManager thực hiện.
7. Gọi agent.profile(output_path=request.report_path); không gọi predict riêng trong action profile. Profiler tự thực hiện workload đo hiện có.
8. Sao chép model_id, artifact metadata, report path.
9. Trong finally, unload model do request này quản lý. Đồng bộ record sau cleanup.
10. Trả ProfilingResult.

Orchestrator không tự tạo ONNX/PyTorch model, không tự export, không tự đo thời gian và không tự start Agent do Registry trả về.

## Trạng thái và lỗi

Reuse lifecycle Agent, không ép BUSY từ Registry trước load_model vì _operation() yêu cầu READY. Các lời gọi load/profile tự chuyển READY -> BUSY -> READY; không hỗ trợ dispatch đồng thời nên khoảng READY giữa hai lời gọi không tạo reservation race được hỗ trợ.

Thêm unload_model(*, preserve_error=False). Mặc định giữ hành vi hiện tại; preserve_error=True giữ ERROR nếu Agent đã ERROR, vẫn giải phóng model và reset model_path/manifest/metadata. Orchestrator dùng preserve_error=True sau lỗi, nên Agent không được chọn lại cho đến khi caller stop/start.

Lỗi validate manifest hoặc không tìm thấy Agent không đổi trạng thái Agent nào. Lỗi load/profile được raise lại, Agent giữ ERROR sau cleanup. Nếu cleanup cũng lỗi, giữ exception gốc làm lỗi chính và gắn thông tin cleanup rõ ràng; không che traceback nguyên nhân ban đầu. Nếu chỉ cleanup lỗi, raise cleanup error, không trả success. get_info() phải vẫn phản ánh trạng thái sau lỗi.

## Demo và cách ly runtime

Demo Registry tạo Agent từ bốn file cấu hình hiện tại, không tự viết bốn record. In metadata sau start/register; báo runtime unavailable riêng; finally stop các instance đã tạo. Không suy luận provider trong danh sách đồng nghĩa inference chắc chắn chạy được.

Demo Orchestrator có chế độ một case (backend/device/report override) và chế độ --all. Một case tạo/register các config tương ứng backend đó, rồi execute request theo Registry. Chế độ --all chạy bốn case tuần tự bằng sys.executable trong bốn subprocess, tổng hợp PASS/FAILED cùng report hoặc error, và trả exit code khác 0 nếu có case lỗi. Mỗi case vẫn dùng AgentRegistry và Orchestrator thật, không gọi CLI Agent để bỏ qua selection.

Các report dùng đường dẫn mặc định theo model/backend/device hoặc override. Dùng manifest chung manifests/resnet18.yaml. Không xóa artifact để chạy demo; reuse cache.

Lệnh dự kiến:

```powershell
.\venv\Scripts\python.exe demo_registry.py
.\venv\Scripts\python.exe demo_orchestrator.py --backend onnxruntime --device gpu --manifest manifests/resnet18.yaml
.\venv\Scripts\python.exe demo_orchestrator.py --all --manifest manifests/resnet18.yaml
```

## Kiểm thử và nghiệm thu

Unit tests Registry: đăng ký từ get_info; trùng ID; ID không tồn tại; filter từng field và kết hợp; snapshot không cho sửa metadata gốc; trạng thái live sau stop; update status hợp lệ và không hợp lệ; unregister/clear không stop ngầm.

Unit tests Orchestrator dùng fake Agent đúng contract: chọn chính xác backend/device; chọn đầu tiên trong nhiều match; bỏ qua BUSY/ERROR/STOPPED và Agent đã load; không có match; request sai; report override; result serialization; unload sau thành công và sau load/profile lỗi; giữ ERROR; cleanup lỗi không che lỗi gốc. Test conversion-compatible logical manifest đi qua Agent mà không bị Orchestrator từ chối.

Regression: chạy agent.test_lifecycle, agent.test_backends, artifact.test_artifacts. Bổ sung test cleanup preserve_error mà không đổi hành vi mặc định.

Integration: chạy demo_registry và demo_orchestrator --all; xác minh agent_id được chọn theo metadata, report JSON tồn tại và có đúng model_id/artifact; các Agent được unload/stop; không fake GPU nếu unavailable. Ghi rõ kết quả thực tế từng case, không yêu cầu mọi máy đều có GPU.

## Khả năng mở rộng

AgentRecord và ProfilingResult có serialization; request validation và execute tạo ranh giới cho REST API/Web UI sau này. Có thể thay select_agent mà không sửa Registry hoặc Predictor. In-memory Registry không phải database và conversion registry trong artifact/registry.py là subsystem khác. Chưa thêm mạng, persistence, hàng đợi hoặc scheduling.
