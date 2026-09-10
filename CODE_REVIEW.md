# 代码审查报告（Code Review）

> 审查对象：星辰知识库 RAG 问答系统（`E:\ai\ai job\rag`，当前 HEAD）
> 范围：`rag/` 核心包、`app.py`、`scripts/`、`tests/`，约 30 个文件 / 5372 行
> 方式：逐文件精读 + 会话内实测证据交叉验证；所有行号对应当前 HEAD

| 项目信息 | 内容 |
|---|---|
| 技术栈 | Python 3.14 · Chroma(HNSW) · sentence-transformers/LangChain-HuggingFace · BM25(jieba) · OpenAI SDK · Streamlit |
| 规模 | 30 文件 / 约 5400 行（核心包 ~2300，脚本 ~1900，测试 ~800，前端 ~700） |
| 状态 | 核心功能完成、有评测体系（243 题）与单元测试（36 个），未容器化 |
| 关注点 | 安全性 · 生产就绪度 · 并发正确性 |

---

## 【总评】

**整体代码质量：7.5 / 10。** 最大的优点是分层干净、核心链路与 UI 彻底解耦，且"评测-基准-审计"三件套齐备到罕见；最需要改进的是**并发与故障路径上的几个真实缺陷**——SQLite 异常未被捕获会直接炸掉问答主流程、上传文件名未消毒存在路径穿越、LLM 客户端无超时控制，这三个都是一行到十行就能修掉的问题。

---

## 【严重问题】（必须修复）

### 🔴 1. `metrics.record()` 只捕获 OSError，SQLite 异常会炸穿 chat 主流程

- **位置**：`rag/metrics.py:31`（`except OSError: pass`）
- **风险**：🔴 严重。`sqlite3.OperationalError`（最典型：多线程/多进程并发写导致的 `database is locked`）继承自 `sqlite3.Error` 而**不是** `OSError`，因此不会被捕获，异常会沿 `pipeline.chat()` 一路抛到用户界面。评测脚本用 6 线程并发时已具备触发条件。
- **修复**：
```python
    except sqlite3.Error as e: # 指标失败绝不影响主流程
        logging.warning("metrics 落库失败: %s", e)
```
- 同类问题：`_check_alerts()`→`aggregate()` 里的 DB 读取同在此 try 内，一并覆盖。

### 🔴 2. 网页上传的文件名未消毒，存在路径穿越（Path Traversal）

- **位置**：`app.py:273-274`（`p = up_dir / f.name; p.write_bytes(f.getvalue())`）
- **风险**：🔴 严重。`f.name` 来自客户端，`..\..\x.bat`、`/c/windows/...` 类名称会使 `Path` 拼接逃出 `data/uploads/`，任意字节写到可写目录。虽然 Streamlit 的 `type=` 白名单限制了扩展名，但**不限制文件名内容**。
- **修复**：
```python
        for f in uploaded:
            safe_name = Path(f.name).name # 去掉任何目录成分
            if safe_name in ("", ".", "..") or Path(safe_name).suffix.lower() not in SUPPORTED_EXTS:
                st.warning(f"跳过不合规文件：{f.name}")
                continue
            p = up_dir / safe_name
            p.write_bytes(f.getvalue())
```

### 🔴 3. OpenAI 客户端没有超时与重试上限

- **位置**：`rag/llm.py:26`（`OpenAI(api_key=..., base_url=...)`）
- **风险**：🔴 严重。SDK 默认超时 600s——端点挂起时单次问答可阻塞 10 分钟；Streamlit/CLI 无任何兜底。
- **修复**：
```python
        _clients[ck] = OpenAI(api_key=api_key, base_url=base_url,
                              timeout=float(os.getenv("LLM_TIMEOUT", "60")),
                              max_retries=1)
```

### 🟡 4. `_last_usage` 模块级全局变量存在并发竞态与归因错误

- **位置**：`rag/llm.py:15,58-59`；读取方 `rag/pipeline.py:205-210`
- **风险**：🟡 中等。并发调用（评测 6 线程、未来多用户）时后写覆盖先写，Token 归因到错误请求；引用重生成会触发第二次调用，`_last_usage` 只剩最后一次，prompt tokens 低估。
- **修复**：`chat()` 把 usage 随返回值带出（或 thread-local）：
```python
def chat(messages, model=..., ...) -> str:
    ...
    _LAST_USAGE.local = resp.usage # threading.local()
# 或更彻底：chat 返回 (text, usage)，调用方按需取用
```

## 🟡 5. BM25 路径存在 N+1 查询

- **位置**：`rag/retriever.py` `keyword_search()`——对每个命中单独 `collection.get(ids=[cid])`，一次混合检索最多 10 次串行 Chroma 查询。
- **风险**：🟡 中等。BM25 命中越多越慢，实测占关键词路耗时的大头。
- **修复**：一次批量取回：
```python
        pairs = self.bm25.search(question, k)
        got = self.collection.get(ids=[cid for cid, _ in pairs],
                                  include=["documents", "metadatas"])
        by_id = dict(zip(got["ids"], zip(got["documents"], got["metadatas"])))
        hits = [Hit(chunk_id=cid, text=by_id[cid][0], **meta) for cid, _ in pairs if cid in by_id]
```
（更优：构建 BM25 时把元数据缓存进 pickle，查询期零回查。）

### 🟡 6. `st.session_state` 在 `@st.cache_resource` 函数内赋值，仅对首个会话生效

- **位置**：`app.py:123`（`get_retriever` 内写 `st.session_state["kb_collection"]`）
- **风险**：🟡 中等。`cache_resource` 全进程只执行一次，第二个浏览器会话拿不到该键，上传时回落到默认集合名——与界面实际使用的集合可能不一致。
- **修复**：集合名的选择移到模块级函数（`active_collection()`），`get_retriever` 与上传入口都调用它，不经过 session_state。

### 🟡 7. 超长输入静默截断，无提示

- **位置**：`rag/security.py:34`（`sanitize_input` 截到 500 字符）
- **风险**：🟡 中等。用户粘贴长文被截断后问题语义不完整，且无任何反馈，容易被当成"系统答非所问"。
- **修复**：`chat()` 返回结构中带 `truncated: True`，前端/CLI 显式提示"输入超过 500 字已截断"。

---

## 【改进建议】（按维度分组）

**Bug 与健壮性**
- `rag/security.py:95` 日志写入只捕 `OSError`——建议同样放宽为 `Exception` 并记 warning（磁盘满、权限问题）。优先级：中
- `rag/bm25.py` 索引 pickle 含 `BM25Okapi` 对象，跨版本升级不兼容时 `load()` 静默返回 None → 混合检索静默降级为纯向量。建议失败时打显式日志并触发重建。优先级：中
- `run_eval.py` 的 `_judge_one` 并发下共享 `rag.llm._last_usage`——修复问题 4 后自然消除。优先级：中

**性能与资源**
- 嵌入层双重归一化（`_LangChainAdapter.encode` 与 `embed_texts` 各归一化一次）——幂等但冗余，留一处即可。优先级：低
- `metrics.record()` 每次新建 SQLite 连接——高频场景建议复用连接或改批量缓冲。优先级：低
- BM25 `get_scores` 为 O(N) 全库打分——≤10 万块无碍（实测 P50 65-84ms），更大规模配合分片/迁移方案。优先级：低
- 查询缓存只覆盖 Streamlit（`st.cache_data`），CLI 与脚本无缓存——如需共享可加进程内 LRU。优先级：低

**架构与设计**
- 分层总体清晰（parsers → chunking → retriever → llm → pipeline → UI），换组件成本已被验证（嵌入层切 LangChain、多厂商路由）。继续保持"脚本薄、核心厚"。
- `scripts/*` 多处重复 `sys.path.insert` 样板——可提供一个 `scripts/_bootstrap.py` 或把包安装为 editable。优先级：低
- `security._append_log` 被 `metrics.py` 以私有名导入（`from .security import _append_log`）——提为公共函数 `append_jsonl()`。优先级：低
- 生产部署形态：Streamlit 是单机有状态应用（本地 chroma_db + sqlite + 日志），不支持水平扩展——文档已注明，若要服务化建议把 `pipeline.chat` 套 FastAPI（核心包已具备条件）。优先级：中

**代码质量**
- `tests/test_routing.py:44` `test_alias_takes_last_at` 是无断言的死测试（裸表达式）——删除或改为真实断言 `resolve_model("a@b")[0] == "a"`。优先级：中
- 个别函数偏长（`app.py` 主问答流程 ~120 行、`pipeline.chat` ~80 行）——建议抽出"检索上下文组装"与"来源信息构建"两个纯函数，便于测试。优先级：中
- 类型注解覆盖良好，个别 `dataclass` 字段（`Hit.rerank_score`）的语义值得在 docstring 标注"None=未重排"。

**可测试性**
- 核心逻辑（切分/BM25/统计/安全/改写）均可离线单测 ✓；`llm.chat` 依赖外部端点但通过 `resolve_model` + 注入 client 已可 mock——建议补一个用 `respx`/假 client 的 `chat()` 重试与超时测试。优先级：中
- 缺失测试：`parse_pdf` 对真实 PDF 的回归（可用 `make_samples` 生成的 PDF 固定为 fixture）、`metrics.aggregate` 的分位数正确性。优先级：中

**生产就绪度**
- 缺 Dockerfile/docker-compose 与健康检查端点——Streamlit 自带 `/_stcore/health` 可直接用，但需在文档/部署配置中写明。优先级：高（若要部署）
- 优雅关闭：无长连接与后台线程常驻（除 Streamlit 自身），风险低。优先级：低
- 审计日志记录完整 prompt 与用户问题——含潜在隐私数据，需在 README 声明保留策略与清理任务（当前仅靠 `logs/` 不入 git）。优先级：中

---

## 【亮点】（值得保持）

1. **核心链路与 UI 完全解耦**：`rag/` 包不含任何 Streamlit 依赖，网页/CLI/评测脚本三个入口共用同一 `pipeline`，换前端零成本。
2. **幂等与增量设计**：chunk_id 由文档内容哈希派生、入库先删后插、MD5 manifest 增量跳过——重复运行不产生脏数据（实测千篇 1.25s）。
3. **引用可信度闭环**：编号由检索结果反向解析 + 范围校验 + 伪造自动重生成 + 审计记录，而不是单纯信任模型输出。
4. **失败降级路径完整**：LLM 不可用→规则改写；Reranker 加载失败→自动跳过；单题评测失败→不中断整体；配额类 4xx 不做无谓重试。
5. **可复现性**：金标"同源构造+校验"、每题明细全量落 JSON、探针/评测/基准脚本均可一键重跑——本次审查中的所有结论都能被第三方复现。
6. **测试意识**：36 个离线单元测试覆盖切分/表格/安全/统计/路由，且明确不含外部依赖。

## 【优先行动清单 Top 5】（影响面 × 修复成本）

| # | 事项 | 成本 | 影响 |
|---|---|---|---|
| 1 | `metrics.py:31` `except OSError` → `except sqlite3.Error`（并发写炸主流程） | 1 行 | 线上稳定性 |
| 2 | 上传文件名消毒（路径穿越） | 5 行 | 安全 |
| 3 | OpenAI 客户端加 `timeout=60, max_retries=1` | 1 行 | 可用性 |
| 4 | `keyword_search` 批量取回消除 N+1 | 10 行 | 检索延迟 |
| 5 | `_last_usage` 去全局化（返回值/线程本地）+ 删除死测试 | 15 行 | 并发正确性 / 测试质量 |

> 以上 1-3 建议立即修复（合计 <10 行改动）；4-5 可并入下个迭代。修复后建议跑一遍 `pytest` + `run_eval --retrieval-only` 做回归确认。
