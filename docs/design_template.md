# Design Template — Multi-Agent Research System

**Student**: Ho Dac Toan — 2A202600057  
**Lab**: 20

---

## Problem

Người dùng đặt một câu hỏi nghiên cứu phức tạp (ví dụ: _"Research GraphRAG
state-of-the-art and write a 500-word summary"_). Hệ thống phải:

1. Tìm kiếm thông tin thực từ web (không hallucinate)
2. Phân tích và cấu trúc thông tin tìm được
3. Tổng hợp câu trả lời cuối có trích dẫn rõ ràng
4. Kiểm tra lại tính chính xác của câu trả lời trước khi trả về người dùng

Một single LLM call không thể làm tốt tất cả bốn việc cùng lúc: câu trả lời
sẽ ngắn, thiếu nguồn, hoặc hallucinate thông tin không có trong training data.

---

## Why multi-agent?

| Limitation của single-agent | Giải pháp multi-agent |
|-----------------------------|----------------------|
| Không có real-time web access | Researcher agent gọi Tavily API |
| Một prompt không thể vừa tìm, vừa phân tích, vừa viết tốt | Mỗi agent chuyên một nhiệm vụ |
| Không thể tự fact-check output của mình | Critic agent độc lập kiểm tra |
| Context window giới hạn cho task dài | State phân tán qua nhiều agent |
| Không có cấu trúc rõ cho debug | `route_history` và `trace` trong ResearchState |

**Kết quả thực tế**: Quality score tăng từ **4.8 → 8.3/10** (+73%) với multi-agent.
Chi phí tăng 4.7×, latency tăng 3.6× — đây là trade-off chấp nhận được cho
research tasks nhưng không phù hợp với real-time chatbot.

---

## Agent Roles

| Agent | Responsibility | Input | Output | Failure mode |
|-------|---------------|-------|--------|--------------|
| **Supervisor** | Điều phối routing: quyết định agent nào chạy tiếp, khi nào dừng | `ResearchState` (kiểm tra `research_notes`, `analysis_notes`, `final_answer`, `iteration`) | Cập nhật `route_history`, tăng `iteration` | Vòng lặp vô hạn nếu không enforce `max_iterations`; xử lý: route `"done"` khi `iteration >= max` |
| **Researcher** | Tìm kiếm web, tổng hợp research notes có citation | `request.query`, `request.max_sources` | `state.sources` (list[SourceDocument]), `state.research_notes` (markdown với [N] refs) | Tavily API lỗi → trả `[]` (fallback graceful); LLM timeout → tenacity retry ×3 |
| **Analyst** | Phân tích research notes thành cấu trúc có thể dùng được | `state.research_notes` | `state.analysis_notes` (KEY CLAIMS / VIEWPOINTS / WEAK EVIDENCE / GAPS) | Nếu `research_notes` rỗng → phân tích trên empty string, output vô nghĩa |
| **Writer** | Tổng hợp final answer có inline citations | `state.research_notes`, `state.analysis_notes`, `state.sources` | `state.final_answer` (~500 words + References section) | Quên dùng inline `[N]` → citation coverage = 0 (đã xảy ra ở Q3) |
| **Critic** | Fact-check final answer, cho VERDICT score 0-10 | `state.final_answer`, `state.sources[:300chars]` | `AgentResult` với `verdict` metadata; không sửa `final_answer` | LLM tự do cũng có thể hallucinate verdict; không có ground truth để verify |

---

## Shared State

`ResearchState` (Pydantic BaseModel) — single source of truth qua toàn bộ graph.

| Field | Type | Lý do cần |
|-------|------|-----------|
| `request` | `ResearchQuery` | Query gốc của user, được tất cả agents dùng |
| `iteration` | `int` | Supervisor đếm để enforce `max_iterations` |
| `route_history` | `list[str]` | Debug routing path; graph dùng `route_history[-1]` để conditional edge |
| `sources` | `list[SourceDocument]` | Researcher điền; Writer dùng để tạo References; Critic dùng để fact-check |
| `research_notes` | `str | None` | None = Researcher chưa chạy; Supervisor dùng để quyết định next route |
| `analysis_notes` | `str | None` | None = Analyst chưa chạy; tương tự |
| `final_answer` | `str | None` | None = Writer chưa chạy; populated = done |
| `agent_results` | `list[AgentResult]` | Full history của mỗi agent: content + metadata (tokens, cost, verdict) |
| `trace` | `list[dict]` | Custom span events để debug từng bước |
| `errors` | `list[str]` | Accumulate lỗi để benchmark đếm failure rate |

**Design decision**: Không dùng immutable updates (mỗi agent mutates state in-place rồi return).
LangGraph node trả về `model_dump()` — LangGraph merge vào state hiện tại.
Vì chúng ta return full dict, merge tương đương replace-all.

---

## Routing Policy

```
START
  │
  ▼
[Supervisor]──── iteration >= max_iterations ────────────────► "done" ──► END
  │
  ├── research_notes is None ──────────────────────────────► "researcher"
  │                                                              │
  │                                                              ▼
  │                                                         [Researcher]
  │                                                              │
  │                                                              └──► [Supervisor]
  │
  ├── analysis_notes is None (research_notes exists) ─────► "analyst"
  │                                                              │
  │                                                              ▼
  │                                                          [Analyst]
  │                                                              │
  │                                                              └──► [Supervisor]
  │
  └── final_answer is None (both notes exist) ─────────────► "writer"
                                                                 │
                                                                 ▼
                                                             [Writer]
                                                                 │
                                                                 ▼
                                                             [Critic] ──► END
```

**Luồng bình thường** (không vượt max_iterations):
`supervisor(0)` → `researcher` → `supervisor(1)` → `analyst` → `supervisor(2)` → `writer` → `critic` → END

**Tổng số LLM calls**: 4 (Researcher + Analyst + Writer + Critic)  
**Tổng số iterations**: 3 (mỗi lần Supervisor chạy tăng 1)

---

## Guardrails

| Guardrail | Giá trị | Implementation | Lý do |
|-----------|---------|---------------|-------|
| **Max iterations** | 6 (env: `MAX_ITERATIONS`) | `Supervisor.run()`: if `iteration >= max` → route `"done"` | Ngăn vòng lặp vô hạn nếu state bị corrupted |
| **LLM timeout** | 30s | `OpenAI(timeout=30)` | Tránh treo vô thời hạn khi model chậm |
| **LLM retry** | 3 lần, exponential backoff 1–30s | `@retry(retry_if_exception_type(RateLimitError, ...))` từ `tenacity` | Rate limit và network errors thoáng qua |
| **Search fallback** | `[]` | `SearchClient.search()` wraps all in try/except → return `[]` | Tavily API có thể fail; Researcher vẫn viết notes dù không có source |
| **Input validation** | `query` min 5 chars, `max_sources` 1–20 | Pydantic `ResearchQuery` field constraints | Reject empty/malformed queries ở CLI boundary |
| **Done fallback** | Route `"done"` khi `final_answer is not None` | Supervisor check | Tránh Writer chạy lần 2 nếu max_iterations chưa đến |

---

## Benchmark Plan

### Queries

| # | Query | Mục đích test |
|---|-------|---------------|
| Q1 | "Research GraphRAG state-of-the-art and write a 500-word summary" | Research task với nhiều nguồn |
| Q2 | "Compare single-agent and multi-agent workflows for customer support" | Comparative analysis |
| Q3 | "Summarize production guardrails for LLM agents" | Technical summarization |

### Metrics

| Metric | Cách đo | Baseline | Multi-Agent |
|--------|---------|:--------:|:-----------:|
| Latency (s) | `perf_counter()` wall-clock | 9.85 | 35.54 |
| Cost (USD) | Σ token costs của tất cả agent_results | 0.000424 | 0.002005 |
| Quality (0–10) | `min(words/500×5, 5) + min(citations×0.5, 5)` | 4.8 | 8.3 |
| Citation coverage | `re.findall(r'\[\d+\]', final_answer)` / sources | 0/0 | 40/15 |
| Failure rate | `len(state.errors)` / total runs | 0% | 0% |

### Expected outcomes (trước khi chạy)

- Multi-agent nên có quality cao hơn nhờ real sources
- Baseline nhanh hơn nhưng không có citations
- Failure rate = 0% nếu guardrails hoạt động đúng
- Cost multi-agent ~4–5× cao hơn baseline (đúng: 4.7×)
