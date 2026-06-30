# ============================================================
# Module: Memory Import Engine (import_memory.py)
# 模块：历史记忆导入引擎
#
# Imports conversation history from various platforms into OB.
# 将各平台对话历史导入 OB 记忆系统。
#
# Supports: Claude JSON, ChatGPT export, DeepSeek, Markdown, plain text
# 支持格式：Claude JSON、ChatGPT 导出、DeepSeek、Markdown、纯文本
#
# Features:
#   - Chunked processing with resume support
#   - Progress persistence (import_state.json)
#   - Raw preservation mode for special contexts
#   - Post-import frequency pattern detection
# ============================================================

import os
import re
import json
import hashlib
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

from utils import count_tokens_approx, now_iso, get_ai_name

logger = logging.getLogger("ombre_brain.import")


# ============================================================
# Format Parsers — normalize any format to conversation turns
# 格式解析器 — 将任意格式标准化为对话轮次
# ============================================================

def _normalize_exporter_time(t: str) -> str:
    """把 Claude Exporter 浏览器插件的时间 '2026/4/5 21:48:30' 转成
    ISO 'YYYY-MM-DDTHH:MM:SS', 失败返回原字符串(让下游 fallback 用 created)."""
    if not t:
        return ""
    try:
        date_part, time_part = str(t).split(" ", 1)
        y, mo, d = date_part.split("/")
        return f"{y}-{int(mo):02d}-{int(d):02d}T{time_part}"
    except (ValueError, AttributeError):
        return str(t)


def _strip_thought_process(text: str) -> str:
    """去掉 Claude Exporter Response 开头的 'Thought process: ...' 段.
    真实正文用 3+ 换行分隔, fallback 双换行."""
    if not text or not text.startswith("Thought process"):
        return text
    parts = re.split(r'\n{3,}', text, maxsplit=1)
    if len(parts) == 2:
        return parts[1].strip()
    parts = re.split(r'\n\s*\n', text, maxsplit=1)
    if len(parts) == 2:
        return parts[1].strip()
    return text


def _parse_claude_json(data: dict | list) -> list[dict]:
    """Parse Claude export JSON → [{role, content, timestamp}, ...]

    支持两种格式:
    1. Anthropic 官方 export: {chat_messages: [{text/content, sender/role, created_at}]}
    2. Claude Exporter 浏览器插件 (https://www.ai-chat-exporter.net):
       {messages: [{say, role: 'Prompt'|'Response', time: 'YYYY/M/D HH:MM:SS'}]}
       自动 strip 每条 Response 的 'Thought process: ...' 独白前缀.
    """
    turns = []
    conversations = data if isinstance(data, list) else [data]
    for conv in conversations:
        messages = conv.get("chat_messages", conv.get("messages", []))
        for msg in messages:
            if not isinstance(msg, dict):
                continue

            # 浏览器插件格式探测: 有 'say' 字段 = Exporter 格式
            if "say" in msg:
                content = msg.get("say", "") or ""
                role_raw = msg.get("role", "")
                role = "user" if role_raw == "Prompt" else "assistant"
                ts = _normalize_exporter_time(msg.get("time", ""))
                if role == "assistant":
                    content = _strip_thought_process(content)
            else:
                # 标准 Anthropic 格式
                content = msg.get("text", msg.get("content", ""))
                if isinstance(content, list):
                    content = " ".join(
                        p.get("text", "") for p in content if isinstance(p, dict)
                    )
                role = msg.get("sender", msg.get("role", "user"))
                ts = msg.get("created_at", msg.get("timestamp", ""))

            if not content or not content.strip():
                continue
            turns.append({"role": role, "content": content.strip(), "timestamp": str(ts)})
    return turns


def _parse_chatgpt_json(data: list | dict) -> list[dict]:
    """Parse ChatGPT export JSON → [{role, content, timestamp}, ...]"""
    turns = []
    conversations = data if isinstance(data, list) else [data]
    for conv in conversations:
        mapping = conv.get("mapping", {})
        if mapping:
            # ChatGPT uses a tree structure with mapping
            sorted_nodes = sorted(
                mapping.values(),
                key=lambda n: n.get("message", {}).get("create_time", 0) or 0,
            )
            for node in sorted_nodes:
                msg = node.get("message")
                if not msg or not isinstance(msg, dict):
                    continue
                content_parts = msg.get("content", {}).get("parts", [])
                content = " ".join(str(p) for p in content_parts if p)
                if not content.strip():
                    continue
                role = msg.get("author", {}).get("role", "user")
                ts = msg.get("create_time", "")
                if isinstance(ts, (int, float)):
                    ts = datetime.fromtimestamp(ts).isoformat()
                turns.append({"role": role, "content": content.strip(), "timestamp": str(ts)})
        else:
            # Simpler format: list of messages
            messages = conv.get("messages", [])
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                content = msg.get("content", msg.get("text", ""))
                if isinstance(content, dict):
                    content = " ".join(str(p) for p in content.get("parts", []))
                if not content or not content.strip():
                    continue
                role = msg.get("role", msg.get("author", {}).get("role", "user"))
                ts = msg.get("timestamp", msg.get("create_time", ""))
                turns.append({"role": role, "content": content.strip(), "timestamp": str(ts)})
    return turns


def _parse_markdown(text: str) -> list[dict]:
    """Parse Markdown/plain text → [{role, content, timestamp}, ...]
    认两种对话格式:
      ① 行首角色前缀  user:/human:/你:/我:  和  assistant:/claude:/ai:/...
      ② markdown 标题式说话人  "### **名字** · 2026-06-06 10:30" / "## 名字"
         (常见于聊天前端导出, 如 用户名/AI名)。名字→角色通用映射: 首个出现的说话人=user,
         其余/已知 AI 名=assistant, 不写死具体名字。带 · 时间戳的会归一化成 [时间戳] 前缀,
         供 digest 推断 event_time。"""
    import re
    lines = text.split("\n")
    turns = []
    current_role = "user"
    current_content = []

    # 标题式说话人: ## 或更深 + (可选粗体)短名字 + (可选 · 时间戳)
    header_re = re.compile(r'^#{2,6}\s*(?:\*\*\s*)?([^*#·\n]{1,40})(?:\s*\*\*)?\s*(·[^\n]*)?$')
    ts_re = re.compile(r'\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}')
    name_role: dict[str, str] = {}
    _known_ai = ("assistant", "claude", "ai", "gpt", "bot", "deepseek", "助手", "模型", "机器人")

    def role_for(name: str) -> str:
        low = name.strip().lower()
        if low in name_role:
            return name_role[low]
        if any(low == k or k in low for k in _known_ai):
            r = "assistant"
        elif "user" not in name_role.values():
            r = "user"          # 首个未知说话人 = 人类
        else:
            r = "assistant"
        name_role[low] = r
        return r

    def flush():
        if current_content:
            c = "\n".join(current_content).strip()
            if c:
                turns.append({"role": current_role, "content": c, "timestamp": ""})

    for line in lines:
        stripped = line.strip()
        m = header_re.match(stripped)
        # 仅当标题里有粗体或 · 时间戳, 才当作说话人(跟普通章节标题区分, 降低误判)
        if m and ("**" in stripped or "·" in stripped):
            name = m.group(1).strip()
            if name:
                flush()
                current_role = role_for(name)
                tm = ts_re.search(m.group(2) or "")
                # 把 · 时间戳归一化成 [YYYY-MM-DD HH:MM] 前缀, digest 用它推 event_time
                current_content = [f"[{tm.group(0).replace('T', ' ')}] "] if tm else []
                continue
        if stripped.lower().startswith(("human:", "user:", "你:", "我:")):
            flush()
            current_role = "user"
            content_after = stripped.split(":", 1)[1].strip() if ":" in stripped else ""
            current_content = [content_after] if content_after else []
        elif stripped.lower().startswith(("assistant:", "claude:", "ai:", "gpt:", "bot:", "deepseek:")):
            flush()
            current_role = "assistant"
            content_after = stripped.split(":", 1)[1].strip() if ":" in stripped else ""
            current_content = [content_after] if content_after else []
        else:
            current_content.append(line)

    flush()

    # If no role patterns detected, treat entire text as one big chunk
    if not turns:
        turns = [{"role": "user", "content": text.strip(), "timestamp": ""}]

    return turns


def detect_and_parse(raw_content: str, filename: str = "") -> list[dict]:
    """
    Auto-detect format and parse to normalized turns.
    自动检测格式并解析为标准化的对话轮次。
    """
    ext = Path(filename).suffix.lower() if filename else ""

    # Try JSON first
    if ext in (".json", "") or raw_content.strip().startswith(("{", "[")):
        try:
            data = json.loads(raw_content)
            # Detect Claude vs ChatGPT format
            if isinstance(data, list):
                sample = data[0] if data else {}
            else:
                sample = data

            if isinstance(sample, dict):
                if "chat_messages" in sample:
                    return _parse_claude_json(data)
                if "mapping" in sample:
                    return _parse_chatgpt_json(data)
                if "messages" in sample:
                    # Could be either — try ChatGPT first, fall back to Claude
                    msgs = sample["messages"]
                    if msgs and isinstance(msgs[0], dict) and "content" in msgs[0]:
                        if isinstance(msgs[0]["content"], dict):
                            return _parse_chatgpt_json(data)
                    return _parse_claude_json(data)
                # Single conversation object with role/content messages
                if "role" in sample and "content" in sample:
                    return _parse_claude_json(data)
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

    # Fall back to markdown/text
    return _parse_markdown(raw_content)


# ============================================================
# Chunking — split turns into ~10k token windows
# 分窗 — 按对话轮次边界切为 ~10k token 窗口
# ============================================================

def chunk_turns(turns: list[dict], target_tokens: int = 10000) -> list[dict]:
    """
    Group conversation turns into chunks of ~target_tokens.
    Returns list of {content, timestamp_start, timestamp_end, turn_count}.
    按对话轮次边界将对话分为 ~target_tokens 大小的窗口。
    """
    chunks = []
    current_lines = []
    current_tokens = 0
    first_ts = ""
    last_ts = ""
    turn_count = 0

    for turn in turns:
        role_label = "用户" if turn["role"] in ("user", "human") else get_ai_name()
        # 嵌入时间戳前缀, 让 digest LLM 能给每条 item 推断准确的 event_time
        # 时间戳格式: [YYYY-MM-DD HH:MM]; 没时间戳的轮次跳过前缀
        ts_raw = turn.get("timestamp", "") or ""
        ts_prefix = ""
        if ts_raw:
            # 标准化: ISO 'YYYY-MM-DDTHH:MM:SS' 或类似 → 'YYYY-MM-DD HH:MM'
            iso = str(ts_raw).replace("/", "-").replace("T", " ")
            ts_prefix = f"[{iso[:16]}] " if len(iso) >= 10 else ""
        line = f"{ts_prefix}[{role_label}] {turn['content']}"
        line_tokens = count_tokens_approx(line)

        # If single turn exceeds target, split it
        if line_tokens > target_tokens * 1.5:
            # Flush current
            if current_lines:
                chunks.append({
                    "content": "\n".join(current_lines),
                    "timestamp_start": first_ts,
                    "timestamp_end": last_ts,
                    "turn_count": turn_count,
                })
                current_lines = []
                current_tokens = 0
                turn_count = 0
                first_ts = ""

            # Add oversized turn as its own chunk
            chunks.append({
                "content": line,
                "timestamp_start": turn.get("timestamp", ""),
                "timestamp_end": turn.get("timestamp", ""),
                "turn_count": 1,
            })
            continue

        if current_tokens + line_tokens > target_tokens and current_lines:
            chunks.append({
                "content": "\n".join(current_lines),
                "timestamp_start": first_ts,
                "timestamp_end": last_ts,
                "turn_count": turn_count,
            })
            current_lines = []
            current_tokens = 0
            turn_count = 0
            first_ts = ""

        if not first_ts:
            first_ts = turn.get("timestamp", "")
        last_ts = turn.get("timestamp", "")
        current_lines.append(line)
        current_tokens += line_tokens
        turn_count += 1

    if current_lines:
        chunks.append({
            "content": "\n".join(current_lines),
            "timestamp_start": first_ts,
            "timestamp_end": last_ts,
            "turn_count": turn_count,
        })

    return chunks


# ============================================================
# Import State — persistent progress tracking
# 导入状态 — 持久化进度追踪
# ============================================================

class ImportState:
    """Manages import progress with file-based persistence."""

    def __init__(self, state_dir: str):
        self.state_file = os.path.join(state_dir, "import_state.json")
        self.data = {
            "source_file": "",
            "source_hash": "",
            "total_chunks": 0,
            "processed": 0,
            "api_calls": 0,
            "memories_created": 0,
            "memories_merged": 0,
            "memories_raw": 0,
            "errors": [],
            "status": "idle",  # idle | running | paused | completed | error
            "started_at": "",
            "updated_at": "",
            "recent_extracted": [],
            "last_llm_output": "",       # 上次 LLM 输出原文片段(parse 失败时给前端展示)
            "last_llm_parsed_ok": True,  # 上次 LLM 输出是否成功解析
            "total_cost_usd": 0.0,       # 累计 LLM 开销(USD)
            "total_in_tokens": 0,
            "total_out_tokens": 0,  # 最近 5 条提取的 [{name, summary}],前端进度条实时展示
        }

    def load(self) -> bool:
        """Load state from file. Returns True if state exists."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                self.data.update(saved)
                return True
            except (json.JSONDecodeError, OSError):
                return False
        return False

    def save(self):
        """Persist state to file."""
        self.data["updated_at"] = now_iso()
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        tmp = self.state_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.state_file)

    def reset(self, source_file: str, source_hash: str, total_chunks: int):
        """Reset state for a new import."""
        self.data = {
            "source_file": source_file,
            "source_hash": source_hash,
            "total_chunks": total_chunks,
            "processed": 0,
            "api_calls": 0,
            "memories_created": 0,
            "memories_merged": 0,
            "memories_raw": 0,
            "errors": [],
            "status": "running",
            "started_at": now_iso(),
            "updated_at": now_iso(),
            "recent_extracted": [],
            "last_llm_output": "",       # 上次 LLM 输出原文片段(parse 失败时给前端展示)
            "last_llm_parsed_ok": True,  # 上次 LLM 输出是否成功解析
            "total_cost_usd": 0.0,       # 累计 LLM 开销(USD)
            "total_in_tokens": 0,
            "total_out_tokens": 0,
        }

    @property
    def can_resume(self) -> bool:
        return self.data["status"] in ("paused", "running") and self.data["processed"] < self.data["total_chunks"]

    def to_dict(self) -> dict:
        return dict(self.data)


# ============================================================
# Import extraction prompt
# 导入提取提示词
# ============================================================

IMPORT_EXTRACT_PROMPT = """你是一个对话记忆提取专家。从以下对话片段中提取值得长期记住的信息。

提取规则：
1. 提取用户的事实、偏好、习惯、重要事件、情感时刻
2. 同一话题的零散信息整合为一条记忆
3. 过滤掉纯技术调试输出、代码块、重复问答、无意义寒暄
4. 如果对话中有特殊暗号、仪式性行为、关键承诺等，标记 preserve_raw=true
5. 如果内容是用户和AI之间的习惯性互动模式（例如打招呼方式、告别习惯），标记 is_pattern=true
6. 每条记忆 30~150 字(不超过 150 字,需要再说就单独拆一条)
7. 总条目数控制在 0~3 个(精挑重要的,宁缺勿滥;没有值得记的就返回空数组)
8. 在 content 中对人名、地名、专有名词用 [[双链]] 标记
9. **JSON 字符串内部一律不要用半角双引号**:
   - 如果要引用某个词或短语,**统一用中文引号「」**(例如:用户提出「赛博永生」的概念)
   - 千万不要写成 `"提出"赛博永生"的概念"`,这会让 JSON 解析失败,整条记忆丢失
   - 所有 string 类型的字段(name/summary/content)都遵守这条规则

【关于 summary（一句话摘要）】这条非常重要：
- summary 是为 AI 后续语义检索服务的，不是给人读流水账。30 字以内，越精炼越好
- 必须**补充而非重复 name**：name 已经说了"是什么"，summary 要说"怎样/为什么/具体特征"
- 如果原文是金句/感悟/比喻，可以直接提炼一句具有检索辨识度的核心表达
- 如果原文是事件/事实陈述，写最能让人/AI 据此回想起这条记忆的关键描述
- 不要用"这是关于..."、"用户表达了..."这类元描述句式，直接写内容
- 反例:name="接纳缺陷的顿悟", summary="一次关于接纳缺陷的思想顿悟"  ❌ 重复
- 正例:name="接纳缺陷的顿悟", summary="承认裂缝才是修补的起点"        ✓ 互补

输出格式(直接输出 JSON 数组,**不要包 markdown 代码块**,不要前言不要解释):
[
  {
    "name": "条目标题（10字以内，说'是什么'）",
    "summary": "一句话摘要（30字以内，补充而非重复 name，服务于语义检索）",
    "content": "整理后的内容",
    "domain": ["主题域1"],
    "valence": 0.7,
    "arousal": 0.4,
    "tags": ["核心词1", "核心词2", "扩展词1"],
    "importance": 5,
    "preserve_raw": false,
    "is_pattern": false
  }
]

主题域可选（选 1~2 个）：
  日常: ["饮食", "穿搭", "出行", "居家", "购物"]
  人际: ["家庭", "恋爱", "友谊", "社交"]
  成长: ["工作", "学习", "考试", "求职"]
  身心: ["健康", "心理", "睡眠", "运动"]
  兴趣: ["游戏", "影视", "音乐", "阅读", "创作", "手工"]
  数字: ["编程", "AI", "硬件", "网络"]
  事务: ["财务", "计划", "待办"]
  内心: ["情绪", "回忆", "梦境", "自省"]

importance: 1-10
valence: 0~1（0=消极, 0.5=中性, 1=积极）
arousal: 0~1（0=平静, 0.5=普通, 1=激动）
preserve_raw: true = 特殊情境/暗号/仪式，保留原文不摘要
is_pattern: true = 反复出现的习惯性行为模式"""


# ============================================================
# Long excerpt 变体 — 在标准 IMPORT_EXTRACT_PROMPT 基础上要求 LLM 输出
# source_excerpt 字段(200-800 字精准片段),给"查看原文"按钮提供完整上下文。
# ----------------------------------------------------------
# 行为开关:config["import"]["long_excerpt"]: true 时启用此 prompt
# 默认 false,因为 DeepSeek 等弱模型可能越界(把同段其他 item 的对话抄进来)
# 或欠抄(应付差事只抄一两句),Claude / GPT-4 这类强模型才稳定
# ============================================================

# 在 schema 例子和规则末尾追加 source_excerpt 字段说明
_LONG_EXCERPT_SCHEMA_LINE = '    "source_excerpt": "属于这条记忆的对话片段原文(200-2500字)",\n    '
_LONG_EXCERPT_RULES = """

【source_excerpt 字段(强制)】这是给"查看原文"功能用的:
- 200-2500 字,直接**抄录原文**(不是摘要,不要改写,**禁止用 ... 省略**)
- 必须包含说话人标记(如果原文有"用户:""AI:"或角色名前缀,保留)
- **关键约束**:只抄录跟**本条 item 直接相关**的对话片段
  - 同 chunk 里如果有别的 item 的对话(讨论无关话题),**不要包含进来**
  - 宁愿少抄(只抄真正相关的 200 字),也不要把无关对话掺进来
  - 反例:本 item 是"讨论搬家",但 source_excerpt 把后面"晚饭吃什么"也抄了 ❌
  - 正例:只抄围绕"搬家"展开的那段对话,从话题开始到自然结束 ✓
- **即使本 item 在原文中只是一两句简短对话, source_excerpt 也至少 100 字**:
  把前后语境一起抄进来, 让"查看原文"始终有完整上下文(不要少于 100 字)
- 如果本 item 涉及的对话超过 2500 字, **不要用 ... 省略**;
  改为在自然段落处把内容拆成多条 item, 每条带自己的 source_excerpt"""

IMPORT_EXTRACT_PROMPT_LONG = (
    IMPORT_EXTRACT_PROMPT
    .replace(
        '    "preserve_raw": false,\n    "is_pattern": false\n  }',
        _LONG_EXCERPT_SCHEMA_LINE + '"preserve_raw": false,\n    "is_pattern": false\n  }',
    )
    + _LONG_EXCERPT_RULES
)


# ============================================================
# SMALL mode framing 后缀 — 用户主动补漏时用, 必出 1 条
# ----------------------------------------------------------
# 当用户在工作台勾选 SMALL 模式时, 主 prompt 末尾追加这段, 让 LLM 知道
# 这一段是用户特意补的, 即使琐碎也尽量提炼, 不要返空数组.
# LARGE 模式 (默认大批量导入) 不加此 suffix, 保持 5-06 稳定的"宁缺勿漏"行为.
#
# 配合: _process_single_chunk 已删除 fallback 兜底逻辑. 现在 LARGE 返空 →
# 直接跳过 chunk; SMALL 因为 prompt 要求"至少 1 条"返空概率极低.
# ============================================================
_SMALL_MODE_SUFFIX = """

【本次模式 — 用户主动补漏 (硬性要求)】
- 这段内容是用户**亲手从聊天记录里挑出来交给你**的, 她已经判断过值得记
- **你的工作是帮她整理, 不是再判断一次值不值得**
- **必须输出至少 1 个 item**, 即使内容看起来琐碎、温柔、片段化、只是一句话
- 主 prompt 里所有"宁缺勿滥 / 0~3 个 / 可返回空数组"的规则**在此模式完全失效**
- 哪怕只有一句"我懂你"的回应、一个温柔的比喻、一段片段化的对话, 也整理为 1 条 item
- **返回空数组 = 把用户的输入扔了, 这是失败行为**
- 极端情况(内容真的无法理解), 也至少创建 1 个 name="(待补充) 用户挑选的片段", importance=3 的占位条目
"""


# ============================================================
# Import Engine — core processing logic
# 导入引擎 — 核心处理逻辑
# ============================================================

class ImportEngine:
    """
    Processes conversation history files into OB memory buckets.
    将对话历史文件处理为 OB 记忆桶。
    """

    def __init__(self, config: dict, bucket_mgr, dehydrator, embedding_engine=None):
        self.config = config
        self.bucket_mgr = bucket_mgr
        self.dehydrator = dehydrator
        self.embedding_engine = embedding_engine
        self.state = ImportState(config["buckets_dir"])
        self._paused = False
        self._running = False
        self._chunks: list[dict] = []
        # 长 excerpt 开关:开 → LLM 额外输出 source_excerpt(200-800 字精准片段),
        #                  让"查看原文"显示完整对话片段而非整 chunk 杂烩
        # 默认 false 是为了开源用户(很多用 DeepSeek,弱模型可能越界/欠抄)
        # 优先级:env OMBRE_LONG_EXCERPT > config.yaml import.long_excerpt > 默认 false
        # 在 Railway/Zeabur 部署加 env 变量 OMBRE_LONG_EXCERPT=true 即可启用,
        # 公仓库代码默认 false,开源用户行为不变(等测过 DeepSeek 再考虑翻默认)
        env_long = os.environ.get("OMBRE_LONG_EXCERPT", "").strip().lower()
        if env_long in ("1", "true", "yes", "on"):
            self.long_excerpt = True
        elif env_long in ("0", "false", "no", "off"):
            self.long_excerpt = False
        else:
            self.long_excerpt = bool(config.get("import", {}).get("long_excerpt", False))
        self.extract_prompt = IMPORT_EXTRACT_PROMPT_LONG if self.long_excerpt else IMPORT_EXTRACT_PROMPT

    @property
    def is_running(self) -> bool:
        return self._running

    def pause(self):
        """Request pause — will stop after current chunk finishes."""
        self._paused = True

    def get_status(self) -> dict:
        """Get current import status."""
        return self.state.to_dict()

    async def start(
        self,
        raw_content: str,
        filename: str = "",
        preserve_raw: bool = False,
        resume: bool = False,
        max_chunks: int = 0,
        mode: str = "large",
    ) -> dict:
        """
        Start or resume an import.
        开始或恢复导入。

        mode: 'large' (默认, 宁缺勿滥) 或 'small' (用户主动单独导一段,
              强制至少 1 条, framing 调整为"帮她提炼"而非"判断值不值得记")
        """
        if self._running:
            return {"error": "Import already running"}

        self._running = True
        self._paused = False

        # mode 决定 prompt 末尾是否追加"必出 1 条"的 SMALL framing
        # - large (默认): 不加 suffix, 保持 5-06 稳定的"宁缺勿滥"行为
        # - small (用户主动补漏): 加 suffix, 让 LLM 至少出 1 条
        base_prompt = IMPORT_EXTRACT_PROMPT_LONG if self.long_excerpt else IMPORT_EXTRACT_PROMPT
        if mode == "small":
            self.extract_prompt = base_prompt + _SMALL_MODE_SUFFIX
        else:
            self.extract_prompt = base_prompt

        try:
            source_hash = hashlib.sha256(raw_content.encode()).hexdigest()[:16]

            # Check for resume
            if resume and self.state.load() and self.state.can_resume:
                if self.state.data["source_hash"] == source_hash:
                    logger.info(f"Resuming import from chunk {self.state.data['processed']}/{self.state.data['total_chunks']}")
                    # Re-parse and re-chunk to get the same chunks
                    turns = detect_and_parse(raw_content, filename)
                    self._chunks = chunk_turns(turns)
                    self.state.data["status"] = "running"
                    self.state.save()
                    return await self._process_chunks(preserve_raw)
                else:
                    logger.warning("Source file changed, starting fresh import")

            # Fresh import
            turns = detect_and_parse(raw_content, filename)
            if not turns:
                self._running = False
                return {"error": "No conversation turns found in file"}

            self._chunks = chunk_turns(turns)
            if not self._chunks:
                self._running = False
                return {"error": "No processable chunks after splitting"}

            # Sample mode: 只跑前 N 个 chunk(试水/控制成本用)
            if max_chunks and max_chunks > 0 and max_chunks < len(self._chunks):
                logger.info(f"Sample mode: limiting to first {max_chunks}/{len(self._chunks)} chunks")
                self._chunks = self._chunks[:max_chunks]

            self.state.reset(filename, source_hash, len(self._chunks))
            self.state.save()

            logger.info(f"Starting import: {len(turns)} turns → {len(self._chunks)} chunks")
            return await self._process_chunks(preserve_raw)

        except Exception as e:
            self.state.data["status"] = "error"
            self.state.data["errors"].append(str(e))
            self.state.save()
            self._running = False
            raise

    async def _process_chunks(self, preserve_raw: bool) -> dict:
        """Process chunks from current position."""
        start_idx = self.state.data["processed"]

        for i in range(start_idx, len(self._chunks)):
            if self._paused:
                self.state.data["status"] = "paused"
                self.state.save()
                self._running = False
                logger.info(f"Import paused at chunk {i}/{len(self._chunks)}")
                return self.state.to_dict()

            chunk = self._chunks[i]
            try:
                await self._process_single_chunk(chunk, preserve_raw)
            except Exception as e:
                # 完整 traceback 写到 logs(用 logger.exception),错误清单只存简短版避免 state 文件膨胀
                logger.exception(f"Import chunk {i} failed")
                err_msg = f"Chunk {i}: {type(e).__name__}: {str(e)[:300]}"
                if len(self.state.data["errors"]) < 100:
                    self.state.data["errors"].append(err_msg)

            self.state.data["processed"] = i + 1
            # Save progress every chunk
            self.state.save()

        self.state.data["status"] = "completed"
        self.state.save()
        self._running = False
        logger.info(f"Import completed: {self.state.data['memories_created']} created, {self.state.data['memories_merged']} merged")
        return self.state.to_dict()

    async def _process_single_chunk(self, chunk: dict, preserve_raw: bool):
        """Extract memories from a single chunk and store them."""
        content = chunk["content"]
        if not content.strip():
            return

        chunk_event_time = chunk.get("timestamp_start") or chunk.get("timestamp")

        # --- LLM extraction ---
        items = []
        try:
            items = await self._extract_memories(content)
            self.state.data["api_calls"] += 1
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")
            self.state.data["api_calls"] += 1

        # 实时记录最近提取的几条,前端进度条展示用
        for it in items:
            recent = self.state.data.setdefault("recent_extracted", [])
            recent.append({
                "name": str(it.get("name", ""))[:30],
                "summary": str(it.get("summary", ""))[:80],
            })
            # 只保留最近 5 条,避免 state 文件膨胀
            if len(recent) > 5:
                self.state.data["recent_extracted"] = recent[-5:]

        # LLM 返空 → 跳过这个 chunk, 不创建任何桶 (回到 5-06 朋友那版的稳定行为)
        # 旧的"宁多勿漏 fallback 待整理桶"已删 — 退化体验 + 性能差,
        # 漏的 chunk 用户可在工作台 SMALL 模式手动补
        if not items:
            return

        # --- Store each extracted memory ---
        # (chunk_event_time 已在函数顶部声明, 给 fallback 路径也用)
        for item in items:
            try:
                should_preserve = preserve_raw or item.get("preserve_raw", False)
                item_event_time = item.get("event_time") or chunk_event_time

                if should_preserve:
                    # Raw mode: store original content without summarization
                    bucket_id = await self.bucket_mgr.create(
                        content=item["content"],
                        tags=item.get("tags", []),
                        importance=item.get("importance", 5),
                        domain=item.get("domain", ["未分类"]),
                        valence=item.get("valence", 0.5),
                        arousal=item.get("arousal", 0.3),
                        name=item.get("name"),
                        summary=item.get("summary"),
                        event_time=item_event_time,
                        created_by="import",  # 跟 AI proactive 写入 (默认 'ai') 区分
                    )
                    if self.embedding_engine:
                        try:
                            await self.embedding_engine.generate_and_store(bucket_id, item["content"])
                        except Exception:
                            pass
                    # API 级 preserve_raw=1 时,raw_source 只能用 LLM 输出的 source_excerpt:
                    # - chunk content 会串入其他 items 对话(旧 bug)
                    # - item.content 是脱水后正文,伪装成原文反而误导用户
                    # 没 source_excerpt 就不写 raw_source,前端显示"无原文"是诚实的呈现。
                    # 想让所有记忆都有原文片段,需要打开 OMBRE_LONG_EXCERPT=true 让 LLM 必出此字段。
                    if preserve_raw and bucket_id:
                        src_excerpt = item.get("source_excerpt", "")
                        if src_excerpt and src_excerpt.strip():
                            try:
                                await self.bucket_mgr.update(bucket_id, raw_source=src_excerpt)
                            except Exception:
                                pass
                    self.state.data["memories_raw"] += 1
                    self.state.data["memories_created"] += 1
                else:
                    # Normal mode: go through merge-or-create pipeline
                    is_merged = await self._merge_or_create_item(item, event_time=item_event_time)
                    if is_merged:
                        self.state.data["memories_merged"] += 1
                    else:
                        self.state.data["memories_created"] += 1

            except Exception as e:
                logger.warning(f"Failed to store memory: {item.get('name', '?')}: {e}")

    async def _extract_memories(self, chunk_content: str) -> list[dict]:
        """Use LLM to extract memories from a conversation chunk."""
        if not self.dehydrator.api_available:
            raise RuntimeError("API not available")

        # [DEBUG] 排查 source_excerpt 链路:确认 long_excerpt 开关实际值 + 用的是哪份 prompt
        # 若 OMBRE_LONG_EXCERPT 设了但新导入仍然没原文 — 用此 log 定位卡在哪一环
        # 同时用 print() 绕过 logger,直接 stdout/stderr,Render Logs 一定能拿到
        prompt_kind = "LONG (含 source_excerpt 强制要求)" if self.long_excerpt else "STD (无 source_excerpt 要求)"
        logger.info(f"[Import LLM] long_excerpt={self.long_excerpt} prompt={prompt_kind}")
        print(f"[Import LLM-PRINT] long_excerpt={self.long_excerpt} prompt_kind={prompt_kind}", flush=True)
        # 把开关值同步写入 state,前端 /api/import/status 能直接看到 — 比翻 log 更可靠
        self.state.data["debug_long_excerpt"] = self.long_excerpt
        self.state.data["debug_prompt_kind"] = prompt_kind

        response = await self.dehydrator.client.chat.completions.create(
            model=self.dehydrator.model,
            messages=[
                {"role": "system", "content": self.extract_prompt},
                {"role": "user", "content": chunk_content[:12000]},
            ],
            max_tokens=16384,  # Sonnet 4.6 上限 64k,留足以防中途截断
            temperature=0.0,
        )
        # 累计开销到 state(供前端进度条展示)
        usage = getattr(response, "usage", None)
        if usage is not None:
            from utils import estimate_llm_cost
            in_tok = getattr(usage, "prompt_tokens", 0) or 0
            out_tok = getattr(usage, "completion_tokens", 0) or 0
            cost = estimate_llm_cost(self.dehydrator.model, in_tok, out_tok)
            self.state.data["total_cost_usd"] = round(self.state.data.get("total_cost_usd", 0) + cost["usd"], 6)
            self.state.data["total_in_tokens"] = self.state.data.get("total_in_tokens", 0) + in_tok
            self.state.data["total_out_tokens"] = self.state.data.get("total_out_tokens", 0) + out_tok

        if not response.choices:
            logger.warning("Import extraction: LLM returned no choices")
            self._record_llm_output("(LLM no choices)", parsed=False)
            return []

        choice = response.choices[0]
        # 兼容两种 message.content 形态:
        #   - OpenAI / DeepSeek 标准:str
        #   - Anthropic native(经某些 OpenAI-compatible 代理 / 直连 SDK):list[TextBlock]
        # 后者 .content 是 [TextBlock(type='text', text='...')] 这种结构,直接 .strip()/切片/parse 都会炸
        # 这里统一拍平成纯字符串
        content_obj = choice.message.content
        if content_obj is None:
            raw = ""
        elif isinstance(content_obj, str):
            raw = content_obj
        elif isinstance(content_obj, list):
            # Anthropic TextBlock list — 取每块的 .text 拼起来(thinking block 等无 .text 的兜底 str 化)
            raw = "".join(getattr(b, "text", "") or "" for b in content_obj)
        else:
            raw = str(content_obj)
        finish_reason = getattr(choice, "finish_reason", "unknown")
        usage = getattr(response, "usage", None)
        usage_info = f"in={usage.prompt_tokens} out={usage.completion_tokens}" if usage else "no usage"
        logger.info(f"Import LLM finish={finish_reason} {usage_info} raw_len={len(raw)}")

        if not raw.strip():
            logger.warning("Import extraction: LLM returned empty content")
            self._record_llm_output(f"(LLM empty content; finish={finish_reason})", parsed=False)
            return []

        logger.info(f"Import LLM raw output (first 300): {raw[:300]}")
        items = self._parse_extraction(raw)
        if not items:
            # parse 失败 OR LLM 真返回空数组,把原文 + finish_reason 存给前端看
            tag = f"[finish_reason={finish_reason}] "
            self._record_llm_output(tag + raw, parsed=False)
        else:
            # 成功路径 — 把 LLM raw output 保留到 last_llm_raw(独立字段,不影响原 last_llm_output UI 行为)
            # [DEBUG] 之前看不到 LLM 给了啥就是因为成功后清空,改成永久保留前 3000 字
            self.state.data["last_llm_output"] = ""
            self.state.data["last_llm_parsed_ok"] = True
            self.state.data["last_llm_raw"] = raw[:3000]
            # 报告本批 items 里多少条带 source_excerpt — 既写 log 也写 state,双保险
            try:
                with_excerpt = sum(1 for it in items if (it.get("source_excerpt") or "").strip())
                keys_first = sorted(list(items[0].keys())) if items else []
                first_excerpt_len = len((items[0].get("source_excerpt") or "")) if items else 0
                logger.info(
                    f"[Import LLM] parsed items={len(items)} with_source_excerpt={with_excerpt} "
                    f"first_item_keys={keys_first} first_excerpt_chars={first_excerpt_len}"
                )
                print(
                    f"[Import LLM-PRINT] parsed items={len(items)} with_excerpt={with_excerpt} "
                    f"keys={keys_first} excerpt_chars={first_excerpt_len}",
                    flush=True,
                )
                self.state.data["debug_with_excerpt"] = with_excerpt
                self.state.data["debug_first_item_keys"] = keys_first
                self.state.data["debug_first_excerpt_chars"] = first_excerpt_len
            except Exception as dbg_e:
                # debug 段不能影响主流程,任何异常吞掉只 log
                logger.warning(f"[Import LLM-DEBUG] inspection failed: {dbg_e}")
        return items

    def _record_llm_output(self, raw: str, parsed: bool):
        """把 LLM 最近一次原始输出存到 state,供前端进度横幅诊断展示"""
        self.state.data["last_llm_output"] = raw[:3000]
        self.state.data["last_llm_parsed_ok"] = parsed

    @staticmethod
    def _parse_extraction(raw: str) -> list[dict]:
        """Parse and validate LLM extraction result.
        多层兜底:直接 parse → 剥 markdown → 找第一个 [ 到最后一个 ] 截取
        """
        cleaned = raw.strip()
        # 1) 剥 markdown 代码块(```json ... ``` / ``` ... ```)
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:].strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

        items = None
        # 2) 直接 parse
        try:
            items = json.loads(cleaned)
        except (json.JSONDecodeError, IndexError, ValueError):
            pass

        # 3) 兜底:找第一个 [ 到最后一个 ] 截取(应对前言 / 后语 / 混合内容)
        if items is None:
            try:
                start = cleaned.find("[")
                end = cleaned.rfind("]")
                if start >= 0 and end > start:
                    items = json.loads(cleaned[start:end + 1])
            except (json.JSONDecodeError, IndexError, ValueError):
                pass

        # 4) 终极兜底:输出被截断(数组没收尾),逐个挖完整 {...} 对象救回来
        if items is None:
            recovered = []
            depth = 0
            in_str = False
            escape = False
            obj_start = -1
            for i, c in enumerate(cleaned):
                if escape:
                    escape = False
                    continue
                if in_str:
                    if c == '\\':
                        escape = True
                    elif c == '"':
                        in_str = False
                    continue
                if c == '"':
                    in_str = True
                    continue
                if c == '{':
                    if depth == 0:
                        obj_start = i
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0 and obj_start >= 0:
                        try:
                            recovered.append(json.loads(cleaned[obj_start:i + 1]))
                        except (json.JSONDecodeError, ValueError):
                            pass
                        obj_start = -1
            if recovered:
                logger.info(f"Import extraction recovered {len(recovered)} items from truncated output")
                items = recovered

        if items is None:
            logger.warning(f"Import extraction JSON parse failed; raw: {raw[:300]}")
            return []

        if not isinstance(items, list):
            return []

        validated = []
        for item in items:
            if not isinstance(item, dict) or not item.get("content"):
                continue
            try:
                importance = max(1, min(10, int(item.get("importance", 5))))
            except (ValueError, TypeError):
                importance = 5
            try:
                valence = max(0.0, min(1.0, float(item.get("valence", 0.5))))
                arousal = max(0.0, min(1.0, float(item.get("arousal", 0.3))))
            except (ValueError, TypeError):
                valence, arousal = 0.5, 0.3

            validated.append({
                "name": str(item.get("name", ""))[:20],
                "summary": str(item.get("summary", ""))[:200],
                "content": str(item["content"]),
                "domain": item.get("domain", ["未分类"])[:3],
                "valence": valence,
                "arousal": arousal,
                "tags": [str(t) for t in item.get("tags", [])][:10],
                "importance": importance,
                "preserve_raw": bool(item.get("preserve_raw", False)),
                "is_pattern": bool(item.get("is_pattern", False)),
                # source_excerpt 是 LONG prompt 模式下的核心字段(LLM 摘的精准对话片段, 200-800 字)
                # 之前白名单没列它 → parse 后就丢了 → raw_source 写入逻辑永远拿不到值
                # 截断到 1500 字留点余量(prompt 上限 800,LLM 偶尔超),避免 metadata 爆炸
                "source_excerpt": str(item.get("source_excerpt") or "")[:1500],
            })

        return validated

    async def _merge_or_create_item(self, item: dict, event_time: str = None) -> bool:
        """Try to merge with existing bucket, or create new. Returns is_merged."""
        content = item["content"]
        domain = item.get("domain", ["未分类"])
        tags = item.get("tags", [])
        importance = item.get("importance", 5)
        valence = item.get("valence", 0.5)
        arousal = item.get("arousal", 0.3)
        name = item.get("name", "")
        summary = item.get("summary", "")

        # auto_merge=False → 跳过合并, 永远新建(默认 True = 上游行为不变)
        try:
            existing = (
                await self.bucket_mgr.search(content, limit=1, domain_filter=domain or None)
                if self.config.get("auto_merge", True) else []
            )
        except Exception:
            existing = []

        merge_threshold = self.config.get("merge_threshold", 75)

        if existing and existing[0].get("score", 0) > merge_threshold:
            bucket = existing[0]
            if not (bucket["metadata"].get("pinned") or bucket["metadata"].get("protected")):
                try:
                    merged = await self.dehydrator.merge(bucket["content"], content)
                    self.state.data["api_calls"] += 1
                    old_v = bucket["metadata"].get("valence", 0.5)
                    old_a = bucket["metadata"].get("arousal", 0.3)
                    await self.bucket_mgr.update(
                        bucket["id"],
                        content=merged,
                        tags=list(set(bucket["metadata"].get("tags", []) + tags)),
                        importance=max(bucket["metadata"].get("importance", 5), importance),
                        domain=list(set(bucket["metadata"].get("domain", []) + domain)),
                        valence=round((old_v + valence) / 2, 2),
                        arousal=round((old_a + arousal) / 2, 2),
                    )
                    if self.embedding_engine:
                        try:
                            await self.embedding_engine.generate_and_store(bucket["id"], merged)
                        except Exception:
                            pass
                    return True
                except Exception as e:
                    logger.warning(f"Merge failed during import: {e}")
                    self.state.data["api_calls"] += 1

        # Create new
        bucket_id = await self.bucket_mgr.create(
            content=content,
            tags=tags,
            importance=importance,
            domain=domain,
            valence=valence,
            arousal=arousal,
            name=name or None,
            summary=summary or None,
            event_time=event_time,
            created_by="import",  # 跟 AI proactive 写入 (默认 'ai') 区分
        )
        # LLM 提取的"原文最关键一两句"片段 — 同时写两个字段:
        #   source_excerpt: 给"重新脱水含正文"主题锚点法 + 未来精确锚定功能用 (核心字段)
        #   raw_source:     给"查看原文"按钮用 (历史行为, 保留兼容)
        src_excerpt = item.get("source_excerpt")
        if bucket_id and src_excerpt:
            try:
                await self.bucket_mgr.update(
                    bucket_id,
                    source_excerpt=src_excerpt,
                    raw_source=src_excerpt,
                )
            except Exception:
                pass
        if self.embedding_engine:
            try:
                await self.embedding_engine.generate_and_store(bucket_id, content)
            except Exception:
                pass
        return False

    async def detect_patterns(self) -> list[dict]:
        """
        Post-import: detect high-frequency patterns via embedding clustering.
        导入后：通过 embedding 聚类检测高频模式。
        Returns list of {pattern_content, count, bucket_ids, suggested_action}.
        """
        if not self.embedding_engine:
            return []

        all_buckets = await self.bucket_mgr.list_all(include_archive=False)
        dynamic_buckets = [
            b for b in all_buckets
            if b["metadata"].get("type") == "dynamic"
            and not b["metadata"].get("pinned")
            and not b["metadata"].get("resolved")
        ]

        if len(dynamic_buckets) < 5:
            return []

        # Get embeddings
        embeddings = {}
        for b in dynamic_buckets:
            emb = await self.embedding_engine.get_embedding(b["id"])
            if emb is not None:
                embeddings[b["id"]] = emb

        if len(embeddings) < 5:
            return []

        # Find clusters: group by pairwise similarity > 0.7
        import numpy as np
        ids = list(embeddings.keys())
        clusters: dict[str, list[str]] = {}
        visited = set()

        for i, id_a in enumerate(ids):
            if id_a in visited:
                continue
            cluster = [id_a]
            visited.add(id_a)
            emb_a = np.array(embeddings[id_a])
            norm_a = np.linalg.norm(emb_a)
            if norm_a == 0:
                continue

            for j in range(i + 1, len(ids)):
                id_b = ids[j]
                if id_b in visited:
                    continue
                emb_b = np.array(embeddings[id_b])
                norm_b = np.linalg.norm(emb_b)
                if norm_b == 0:
                    continue
                sim = float(np.dot(emb_a, emb_b) / (norm_a * norm_b))
                if sim > 0.7:
                    cluster.append(id_b)
                    visited.add(id_b)

            if len(cluster) >= 3:
                clusters[id_a] = cluster

        # Format results
        patterns = []
        for lead_id, cluster_ids in clusters.items():
            lead_bucket = next((b for b in dynamic_buckets if b["id"] == lead_id), None)
            if not lead_bucket:
                continue
            patterns.append({
                "pattern_content": lead_bucket["content"][:200],
                "pattern_name": lead_bucket["metadata"].get("name", lead_id),
                "count": len(cluster_ids),
                "bucket_ids": cluster_ids,
                "suggested_action": "pin" if len(cluster_ids) >= 5 else "review",
            })

        patterns.sort(key=lambda p: p["count"], reverse=True)
        return patterns[:20]
