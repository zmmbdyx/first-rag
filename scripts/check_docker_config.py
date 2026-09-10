# -*- coding: utf-8 -*-
"""静态校验 Docker 配置：compose 语法 + 引用文件存在性 + Dockerfile 关键指令。

说明：本机未安装 Docker，无法真正 docker build；此脚本做的是可做的静态检查，
用于尽早发现"引用了不存在的文件""端口/卷写错""缺少关键指令"这类错误。
"""
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent

ok = True


def check(name, cond, detail=""):
    global ok
    if not cond:
        ok = False
    print(("  ✅ " if cond else "  ❌ ") + name + (f"  ({detail})" if detail else ""))


print("== 1. docker-compose.yml 语法 ==")
try:
    import yaml
    comp = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    check("YAML 解析成功", True)
    check("包含 redis/api/ui 三个服务",
          set(comp.get("services", {})) == {"redis", "api", "ui"},
          str(list(comp.get("services", {}))))
    check("声明了 hf_models 卷", "hf_models" in comp.get("volumes", {}))
    check("api 依赖 redis 健康检查",
          "redis" in (comp["services"]["api"].get("depends_on") or {}))
    check("redis 有健康检查", "healthcheck" in comp["services"]["redis"])
    # 端口不与宿主机常见冲突
    ports = [p for s in comp["services"].values() for p in (s.get("ports") or [])]
    check("端口已声明", len(ports) >= 3, str(ports))
except ImportError:
    print("  ⚠️  未安装 pyyaml，跳过 compose 校验")
except Exception as e:
    check(f"YAML 解析失败: {e}", False)

print("== 2. compose 引用的挂载路径 ==")
for p in ("chroma_db", "logs", "data/uploads"):
    exists = (ROOT / p).exists() or (ROOT / p).parent.exists()
    check(f"{p} 的父目录存在（compose 会自动创建）", exists, str(ROOT / p))
check(".env 模板存在（compose 的 env_file 指向 .env）", (ROOT / ".env.example").exists())
check(".env 被 gitignore 排除（不提交密钥）",
      ".env" in (ROOT / ".gitignore").read_text(encoding="utf-8"))

print("== 3. Dockerfile 关键指令 ==")
df = (ROOT / "Dockerfile").read_text(encoding="utf-8")
check("基于 python:3.11-slim", "python:3.11-slim" in df)
check("先 COPY requirements 再 COPY 代码（利用层缓存）",
      df.index("COPY requirements.txt") < df.index("COPY . ."))
check("安装系统依赖", "apt-get install" in df)
check("非 root 运行", "USER appuser" in df)
check("暴露 8000", "EXPOSE 8000" in df)
check("配置 HEALTHCHECK", "HEALTHCHECK" in df)
check("启动 FastAPI 服务", "uvicorn" in df and "api_server:app" in df)
check("未硬编码任何密钥", not re.search(r"sk-[A-Za-z0-9]{10,}", df))

print("== 4. Dockerfile COPY 的文件真实存在 ==")
for f in ("requirements.txt", "api_server.py", "app.py", "rag/pipeline.py"):
    check(f"{f} 存在", (ROOT / f).exists())

print("== 5. .dockerignore 排除敏感/冗余内容 ==")
di = ""
if (ROOT / ".dockerignore").exists():
    di = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    check(".dockerignore 存在", True)
    for pat in (".env", "chroma_db", "logs", "__pycache__", ".tools"):
        check(f"排除 {pat}", pat in di)
else:
    check(".dockerignore 存在", False)

print("== 6. requirements 覆盖运行所需依赖 ==")
req = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
for dep in ("fastapi", "uvicorn", "redis", "python-multipart", "streamlit",
            "chromadb", "pymupdf", "python-docx", "sentence-transformers"):
    check(f"声明 {dep}", dep in req)

print("\n" + "=" * 46)
print("静态校验通过 ✅" if ok else "存在失败项 ❌")
sys.exit(0 if ok else 1)
