# ============================================================
# Module: MCP Server Entry Point (server.py)
# 模块：MCP 服务器主入口
#
# Starts the Ombre Brain MCP service and registers memory
# operation tools for Claude to call.
# 启动 Ombre Brain MCP 服务，注册记忆操作工具供 Claude 调用。
#
# Core responsibilities:
# 核心职责：
#   - Initialize config, bucket manager, dehydrator, decay engine
#     初始化配置、记忆桶管理器、脱水器、衰减引擎
#   - Expose 5 MCP tools:
#     暴露 5 个 MCP 工具：
#       breath — Surface unresolved memories or search by keyword
#                浮现未解决记忆 或 按关键词检索
#       hold   — Store a single memory
#                存储单条记忆
#       grow   — Diary digest, auto-split into multiple buckets
#                日记归档，自动拆分多桶
#       trace  — Modify metadata / resolved / delete
#                修改元数据 / resolved 标记 / 删除
#       pulse  — System status + bucket listing
#                系统状态 + 所有桶列表
#
# Startup:
# 启动方式：
#   Local:  python server.py
#   Remote: OMBRE_TRANSPORT=streamable-http python server.py
#   Docker: docker-compose up
# ============================================================

import os
import sys
import time
import random
import logging
import asyncio
import httpx


# --- Ensure same-directory modules can be imported ---
# --- 确保同目录下的模块能被正确导入 ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP

from bucket_manager import BucketManager
from dehydrator import Dehydrator
from decay_engine import DecayEngine
from embedding_engine import EmbeddingEngine
from import_memory import ImportEngine
from utils import load_config, setup_logging, strip_wikilinks, count_tokens_approx, is_internalized, is_protected, is_highlighted, get_ai_name

# --- Load config & init logging / 加载配置 & 初始化日志 ---
config = load_config()
setup_logging(config.get("log_level", "INFO"))
logger = logging.getLogger("ombre_brain")

# --- Initialize core components / 初始化核心组件 ---
bucket_mgr = BucketManager(config)                  # Bucket manager / 记忆桶管理器
dehydrator = Dehydrator(config)                      # Dehydrator / 脱水器
decay_engine = DecayEngine(config, bucket_mgr)       # Decay engine / 衰减引擎
embedding_engine = EmbeddingEngine(config)            # Embedding engine / 向量化引擎
import_engine = ImportEngine(config, bucket_mgr, dehydrator, embedding_engine)  # Import engine / 导入引擎

# --- /api/buckets in-memory cache / 内存级缓存 ---
# 每个视图(cells/network/console/mobile)启动都自己拉一遍 /api/buckets,
# 切视图 = 重复拉 = 重复 IO 200+ 个 .md frontmatter.load. 用户切视图卡几分钟.
# 加 15s TTL 内存缓存: 切视图秒回, 写操作主动 invalidate 避免看到旧数据.
# MCP tool (hold/grow 等) 写桶不调 invalidate, 15s 后自然失效, 可容忍.
_BUCKETS_CACHE = {"ts": 0.0, "payload": None}
_BUCKETS_CACHE_TTL = 15.0  # 秒

def _invalidate_buckets_cache():
    _BUCKETS_CACHE["ts"] = 0.0
    _BUCKETS_CACHE["payload"] = None


# --- Create MCP server instance / 创建 MCP 服务器实例 ---
# host="0.0.0.0" so Docker container's SSE is externally reachable
# stdio mode ignores host (no network)
mcp = FastMCP(
    "Ombre Brain",
    host="0.0.0.0",
    port=8000,
)


# =============================================================
# /health endpoint: lightweight keepalive
# 轻量保活接口
# For Cloudflare Tunnel or reverse proxy to ping, preventing idle timeout
# 供 Cloudflare Tunnel 或反代定期 ping，防止空闲超时断连
# =============================================================
@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    from starlette.responses import JSONResponse
    try:
        stats = await bucket_mgr.get_stats()
        return JSONResponse({
            "status": "ok",
            "buckets": stats["permanent_count"] + stats["dynamic_count"],
            "decay_engine": "running" if decay_engine.is_running else "stopped",
        })
    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


# =============================================================
# /breath-hook endpoint: Dedicated hook for SessionStart
# 会话启动专用挂载点
# =============================================================
@mcp.custom_route("/breath-hook", methods=["GET"])
async def breath_hook(request):
    from starlette.responses import PlainTextResponse
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
        # 核心准则段:highlight=True 的桶始终浮现(已内化的隐藏)
        pinned = [b for b in all_buckets
                  if is_highlighted(b["metadata"])
                  and not is_internalized(b["metadata"])]
        # 永久参考段:protected-only (跟 breath() 主路径同步, 跟用户"钉决=应该在视野"直觉对齐)
        protected_only_hook = [b for b in all_buckets
                               if is_protected(b["metadata"])
                               and not is_highlighted(b["metadata"])
                               and not is_internalized(b["metadata"])]
        # 普通浮现池:排除已经进核心准则区的 highlighted 桶,排除 permanent/feel 类型
        unresolved = [b for b in all_buckets
                      if not b["metadata"].get("resolved", False)
                      and b["metadata"].get("type") not in ("permanent", "feel")
                      and not is_highlighted(b["metadata"])
                      and not is_internalized(b["metadata"])]
        scored = sorted(unresolved, key=lambda b: decay_engine.calculate_score(b["metadata"]), reverse=True)

        parts = []
        token_budget = 10000
        for b in pinned:
            summary = await dehydrator.dehydrate(strip_wikilinks(b["content"]), {k: v for k, v in b["metadata"].items() if k != "tags"})
            parts.append(f"📌 [核心准则] {summary}")
            token_budget -= count_tokens_approx(summary)
        for b in protected_only_hook:
            if token_budget <= 0:
                break
            summary = await dehydrator.dehydrate(strip_wikilinks(b["content"]), {k: v for k, v in b["metadata"].items() if k != "tags"})
            parts.append(f"❖ [永久参考] {summary}")
            token_budget -= count_tokens_approx(summary)

        # Diversity: top-1 fixed + shuffle rest from top-20
        candidates = list(scored)
        if len(candidates) > 1:
            top1 = [candidates[0]]
            pool = candidates[1:min(20, len(candidates))]
            random.shuffle(pool)
            candidates = top1 + pool + candidates[min(20, len(candidates)):]
        # Hard cap: max 20 surfacing buckets in hook
        candidates = candidates[:20]

        surfaced_ids = []
        for b in candidates:
            if token_budget <= 0:
                break
            summary = await dehydrator.dehydrate(strip_wikilinks(b["content"]), {k: v for k, v in b["metadata"].items() if k != "tags"})
            summary_tokens = count_tokens_approx(summary)
            if summary_tokens > token_budget:
                break
            parts.append(summary)
            surfaced_ids.append(b["id"])
            token_budget -= summary_tokens

        # 记"被自动浮现"命中 (SessionStart hook 也是"被想起"的一条路径, 跟 breath() 主路径一致)
        if surfaced_ids:
            bucket_mgr.record_surfacing(surfaced_ids)

        # 浮现追溯 (kind='surface') — 同 breath() 主路径
        try:
            _surf_trace = []
            for _b in pinned:
                _m = _b["metadata"]
                _surf_trace.append({"id": _b["id"], "name": _m.get("name") or _b["id"],
                                    "type": _m.get("type", "dynamic"), "score": decay_engine.calculate_score(_m),
                                    "highlight": True, "protected": is_protected(_m)})
            for _b in protected_only_hook:
                _m = _b["metadata"]
                _surf_trace.append({"id": _b["id"], "name": _m.get("name") or _b["id"],
                                    "type": _m.get("type", "permanent"), "score": decay_engine.calculate_score(_m),
                                    "highlight": False, "protected": True})
            _surf_set = set(surfaced_ids)
            for _b in candidates:
                if _b["id"] not in _surf_set:
                    continue
                _m = _b["metadata"]
                _surf_trace.append({"id": _b["id"], "name": _m.get("name") or _b["id"],
                                    "type": _m.get("type", "dynamic"), "score": decay_engine.calculate_score(_m),
                                    "highlight": False, "protected": False})
            if _surf_trace:
                bucket_mgr.record_surface_trace(_surf_trace)
        except Exception:
            pass

        if not parts:
            return PlainTextResponse("")
        return PlainTextResponse("[Ombre Brain - 记忆浮现]\n" + "\n---\n".join(parts))
    except Exception as e:
        logger.warning(f"Breath hook failed: {e}")
        return PlainTextResponse("")


# =============================================================
# /dream-hook endpoint: Dedicated hook for Dreaming
# Dreaming 专用挂载点
# =============================================================
@mcp.custom_route("/dream-hook", methods=["GET"])
async def dream_hook(request):
    from starlette.responses import PlainTextResponse
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
        candidates = [
            b for b in all_buckets
            if b["metadata"].get("type") not in ("permanent", "feel")
            and not is_highlighted(b["metadata"])
            and not is_internalized(b["metadata"])
        ]
        candidates.sort(key=lambda b: b["metadata"].get("created", ""), reverse=True)
        recent = candidates[:10]

        if not recent:
            return PlainTextResponse("")

        parts = []
        for b in recent:
            meta = b["metadata"]
            resolved_tag = "[已解决]" if meta.get("resolved", False) else "[未解决]"
            parts.append(
                f"{meta.get('name', b['id'])} {resolved_tag} "
                f"V{meta.get('valence', 0.5):.1f}/A{meta.get('arousal', 0.3):.1f}\n"
                f"{strip_wikilinks(b['content'][:200])}"
            )

        return PlainTextResponse("[Ombre Brain - Dreaming]\n" + "\n---\n".join(parts))
    except Exception as e:
        logger.warning(f"Dream hook failed: {e}")
        return PlainTextResponse("")


# =============================================================
# Internal helper: merge-or-create
# 内部辅助：检查是否可合并，可以则合并，否则新建
# Shared by hold and grow to avoid duplicate logic
# hold 和 grow 共用，避免重复逻辑
# =============================================================
async def _merge_or_create(
    content: str,
    tags: list,
    importance: int,
    domain: list,
    valence: float,
    arousal: float,
    name: str = "",
    event_time: str = None,
) -> tuple[str, bool]:
    """
    Check if a similar bucket exists for merging; merge if so, create if not.
    Returns (bucket_id_or_name, is_merged).
    检查是否有相似桶可合并，有则合并，无则新建。
    返回 (桶ID或名称, 是否合并)。
    """
    # auto_merge=False → 跳过相似桶合并, 永远新建(默认 True = 上游行为不变)。
    # 合并不稳: 打分被调高(precise/boosts)后会误合并不相干记忆 → 个人实例可关掉改手动去重。
    try:
        existing = (
            await bucket_mgr.search(content, limit=1, domain_filter=domain or None)
            if config.get("auto_merge", True) else []
        )
    except Exception as e:
        logger.warning(f"Search for merge failed, creating new / 合并搜索失败，新建: {e}")
        existing = []

    if existing and existing[0].get("score", 0) > config.get("merge_threshold", 75):
        bucket = existing[0]
        # --- Never merge into pinned/protected buckets ---
        # --- 不合并到钉选/保护桶 ---
        if not (bucket["metadata"].get("pinned") or bucket["metadata"].get("protected")):
            try:
                merged = await dehydrator.merge(bucket["content"], content)
                old_v = bucket["metadata"].get("valence", 0.5)
                old_a = bucket["metadata"].get("arousal", 0.3)
                merged_valence = round((old_v + valence) / 2, 2)
                merged_arousal = round((old_a + arousal) / 2, 2)
                update_kwargs = dict(
                    content=merged,
                    tags=list(set(bucket["metadata"].get("tags", []) + tags)),
                    importance=max(bucket["metadata"].get("importance", 5), importance),
                    domain=list(set(bucket["metadata"].get("domain", []) + domain)),
                    valence=merged_valence,
                    arousal=merged_arousal,
                )
                # 合并时若调用方明确传了 event_time,跟随更新(用更近的事件时间);
                # 没传就保持旧 event_time 不动
                if event_time is not None:
                    update_kwargs["event_time"] = event_time
                await bucket_mgr.update(bucket["id"], **update_kwargs)
                # --- Update embedding after merge ---
                try:
                    await embedding_engine.generate_and_store(bucket["id"], merged)
                except Exception:
                    pass
                return bucket["metadata"].get("name", bucket["id"]), True
            except Exception as e:
                logger.warning(f"Merge failed, creating new / 合并失败，新建: {e}")

    bucket_id = await bucket_mgr.create(
        content=content,
        tags=tags,
        importance=importance,
        domain=domain,
        valence=valence,
        arousal=arousal,
        name=name or None,
        event_time=event_time,
    )
    # --- Generate embedding for new bucket ---
    try:
        await embedding_engine.generate_and_store(bucket_id, content)
    except Exception:
        pass
    return bucket_id, False


# =============================================================
# Tool 1: breath — Breathe
# 工具 1：breath — 呼吸
#
# No args: surface highest-weight unresolved memories (active push)
# 无参数：浮现权重最高的未解决记忆
# With args: search by keyword + emotion coordinates
# 有参数：按关键词+情感坐标检索记忆
# =============================================================
@mcp.tool()
async def breath(
    query: str = "",
    max_tokens: int = 10000,
    domain: str = "",
    valence: float = -1,
    arousal: float = -1,
    max_results: int = 20,
) -> str:
    """检索/浮现记忆。不传query或传空=自动浮现,有query=关键词检索。max_tokens控制返回总token上限(默认10000)。domain逗号分隔,valence/arousal 0~1(-1忽略)。max_results控制返回数量上限(默认20,最大50)。"""
    await decay_engine.ensure_started()
    max_results = min(max_results, 50)
    max_tokens = min(max_tokens, 20000)

    # --- No args or empty query: surfacing mode (weight pool active push) ---
    # --- 无参数或空query：浮现模式（权重池主动推送）---
    if not query or not query.strip():
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
        except Exception as e:
            logger.error(f"Failed to list buckets for surfacing / 浮现列桶失败: {e}")
            return "记忆系统暂时无法访问。"

        # --- Highlighted buckets: always surface as core principles ---
        # --- 置顶桶(highlight=True):作为核心准则始终浮现(已内化的隐藏) ---
        pinned_buckets = [
            b for b in all_buckets
            if is_highlighted(b["metadata"])
            and not is_internalized(b["metadata"])
        ]
        pinned_results = []
        for b in pinned_buckets:
            try:
                clean_meta = {k: v for k, v in b["metadata"].items() if k != "tags"}
                summary = await dehydrator.dehydrate(strip_wikilinks(b["content"]), clean_meta)
                pinned_results.append(f"📌 [核心准则] [bucket_id:{b['id']}] {summary}")
            except Exception as e:
                logger.warning(f"Failed to dehydrate pinned bucket / 钉选桶脱水失败: {e}")
                continue

        # --- Protected-only buckets: surface as 永久参考 ---
        # --- 钉决但未高亮 (protected=True, highlight=False):
        #     之前 unresolved 排除 type=permanent, pinned_buckets 又要求 highlight,
        #     protected-only 桶两边都漏 → 跟用户"钉决=应该一直在视野里"的直觉冲突.
        #     单开一个区段, 让钉决的桶也参与浮现. ---
        protected_only = [
            b for b in all_buckets
            if is_protected(b["metadata"])
            and not is_highlighted(b["metadata"])
            and not is_internalized(b["metadata"])
        ]
        protected_results = []
        for b in protected_only:
            try:
                clean_meta = {k: v for k, v in b["metadata"].items() if k != "tags"}
                summary = await dehydrator.dehydrate(strip_wikilinks(b["content"]), clean_meta)
                protected_results.append(f"❖ [永久参考] [bucket_id:{b['id']}] {summary}")
            except Exception as e:
                logger.warning(f"Failed to dehydrate protected bucket / 钉决桶脱水失败: {e}")
                continue

        # --- Unresolved buckets: surface top N by weight ---
        # --- 未解决桶：按权重浮现前 N 条 ---
        unresolved = [
            b for b in all_buckets
            if not b["metadata"].get("resolved", False)
            and b["metadata"].get("type") not in ("permanent", "feel")
            and not is_highlighted(b["metadata"])
            and not is_internalized(b["metadata"])
        ]

        logger.info(
            f"Breath surfacing: {len(all_buckets)} total, "
            f"{len(pinned_buckets)} highlighted, "
            f"{len(protected_only)} protected-only, "
            f"{len(unresolved)} unresolved"
        )

        scored = sorted(
            unresolved,
            key=lambda b: decay_engine.calculate_score(b["metadata"]),
            reverse=True,
        )

        if scored:
            top_scores = [(b["metadata"].get("name", b["id"]), decay_engine.calculate_score(b["metadata"])) for b in scored[:5]]
            logger.info(f"Top unresolved scores: {top_scores}")

        # --- Token-budgeted surfacing with diversity + hard cap ---
        # --- 按 token 预算浮现，带多样性 + 硬上限 ---
        # Top-1 always surfaces; rest sampled from top-20 for diversity
        token_budget = max_tokens
        for r in pinned_results:
            token_budget -= count_tokens_approx(r)
        for r in protected_results:
            token_budget -= count_tokens_approx(r)

        # --- 冷启动检测：从未被访问过(activation_count==0)且重要度>=8的桶，优先插最前(最多2个) ---
        # 让你新存的重要记忆有机会先被想起一次，而不是没露过脸就沉底（对齐上游 B-04）。
        cold_start = [
            b for b in unresolved
            if int(b["metadata"].get("activation_count", 0)) == 0
            and int(b["metadata"].get("importance", 0)) >= 8
        ][:2]
        cold_start_ids = {b["id"] for b in cold_start}
        scored_with_cold = cold_start + [b for b in scored if b["id"] not in cold_start_ids]

        candidates = list(scored_with_cold)
        if len(candidates) > 1:
            # 冷启动桶固定在最前；其余从 top-20 里随机打乱保多样性
            n_cold = len(cold_start)
            non_cold = candidates[n_cold:]
            if len(non_cold) > 1:
                top1 = [non_cold[0]]
                pool = non_cold[1:min(20, len(non_cold))]
                random.shuffle(pool)
                non_cold = top1 + pool + non_cold[min(20, len(non_cold)):]
            candidates = cold_start + non_cold
        # Hard cap: never surface more than max_results buckets
        candidates = candidates[:max_results]

        dynamic_results = []
        surfaced_ids = []
        for b in candidates:
            if token_budget <= 0:
                break
            try:
                clean_meta = {k: v for k, v in b["metadata"].items() if k != "tags"}
                summary = await dehydrator.dehydrate(strip_wikilinks(b["content"]), clean_meta)
                summary_tokens = count_tokens_approx(summary)
                if summary_tokens > token_budget:
                    break
                # NOTE: no touch() here — surfacing should NOT reset decay timer
                score = decay_engine.calculate_score(b["metadata"])
                dynamic_results.append(f"[权重:{score:.2f}] [bucket_id:{b['id']}] {summary}")
                surfaced_ids.append(b["id"])
                token_budget -= summary_tokens
            except Exception as e:
                logger.warning(f"Failed to dehydrate surfaced bucket / 浮现脱水失败: {e}")
                continue

        # 记"被自动浮现"命中: 跟 search 关键词命中分开计(surface_count), 让观测统计的
        # "被想起 = 被检索 + 被浮现"完整。只记动态权重池浮现的桶; 钉选/永久参考每次都在、计了无意义。
        if surfaced_ids:
            bucket_mgr.record_surfacing(surfaced_ids)

        # 浮现追溯 (kind='surface'): 把这次 breath 浮现的高亮/钉决/动态记进最近追溯,
        # 让观测台「最近浮现 · 检索」能看到"这次 breath 浮现了什么"(跟关键词检索区分)。
        try:
            _surf_trace = []
            for _b in pinned_buckets:
                _m = _b["metadata"]
                _surf_trace.append({"id": _b["id"], "name": _m.get("name") or _b["id"],
                                    "type": _m.get("type", "dynamic"), "score": decay_engine.calculate_score(_m),
                                    "highlight": True, "protected": is_protected(_m)})
            for _b in protected_only:
                _m = _b["metadata"]
                _surf_trace.append({"id": _b["id"], "name": _m.get("name") or _b["id"],
                                    "type": _m.get("type", "permanent"), "score": decay_engine.calculate_score(_m),
                                    "highlight": False, "protected": True})
            _surf_set = set(surfaced_ids)
            for _b in candidates:
                if _b["id"] not in _surf_set:
                    continue
                _m = _b["metadata"]
                _surf_trace.append({"id": _b["id"], "name": _m.get("name") or _b["id"],
                                    "type": _m.get("type", "dynamic"), "score": decay_engine.calculate_score(_m),
                                    "highlight": False, "protected": False})
            if _surf_trace:
                bucket_mgr.record_surface_trace(_surf_trace)
        except Exception:
            pass

        if not pinned_results and not protected_results and not dynamic_results:
            return "权重池平静，没有需要处理的记忆。"

        parts = []
        if pinned_results:
            parts.append("=== 核心准则 ===\n" + "\n---\n".join(pinned_results))
        if protected_results:
            parts.append("=== 永久参考 ===\n" + "\n---\n".join(protected_results))
        if dynamic_results:
            parts.append("=== 浮现记忆 ===\n" + "\n---\n".join(dynamic_results))
        return "\n\n".join(parts)

    # --- Feel retrieval: domain="feel" is a special channel ---
    # --- Feel 检索：domain="feel" 是独立入口 ---
    if domain.strip().lower() == "feel":
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
            feels = [b for b in all_buckets if b["metadata"].get("type") == "feel"]
            feels.sort(key=lambda b: b["metadata"].get("created", ""), reverse=True)
            if not feels:
                return "没有留下过 feel。"
            results = []
            for f in feels:
                created = f["metadata"].get("created", "")
                entry = f"[{created}] [bucket_id:{f['id']}]\n{strip_wikilinks(f['content'])}"
                results.append(entry)
                if count_tokens_approx("\n---\n".join(results)) > max_tokens:
                    break
            return "=== 你留下的 feel ===\n" + "\n---\n".join(results)
        except Exception as e:
            logger.error(f"Feel retrieval failed: {e}")
            return "读取 feel 失败。"

    # --- With args: search mode (keyword + vector dual channel) ---
    # --- 有参数：检索模式（关键词 + 向量双通道）---
    domain_filter = [d.strip() for d in domain.split(",") if d.strip()] or None
    q_valence = valence if 0 <= valence <= 1 else None
    q_arousal = arousal if 0 <= arousal <= 1 else None

    try:
        matches = await bucket_mgr.search(
            query,
            limit=max(max_results, 20),
            domain_filter=domain_filter,
            query_valence=q_valence,
            query_arousal=q_arousal,
        )
    except Exception as e:
        logger.error(f"Search failed / 检索失败: {e}")
        return "检索过程出错，请稍后重试。"

    # --- Exclude internalized/noise/feel from search results ---
    # --- 搜索模式排除 已内化桶、噪声桶、feel 桶 ---
    # 注:protected 但非 highlighted 的桶可被搜出(防衰减不影响搜索)
    # noise = resolved + importance=1, 用户软删除标记 → 默认从检索排除
    # feel = 第一人称感受, 设计上只能通过 domain="feel" 独立通道读取, 不参与普通搜索/浮现
    #
    # 历史 bug 修复(2026-05): highlighted (钉选) 桶原本也被搜索排除, 设计意图是
    # "它们在浮现模式核心准则区已可见, 避免搜索结果重复". 但实际场景下:
    #   - 用户精准搜索 "完整指南" / "互动指南" / "关系文档" 这类高重要度桶时, 几乎都是钉选状态
    #   - 全被排除 → 用户感受是"我的桶搜不到"
    # 搜索和浮现是两个独立 mode (返回字符串完全不同), 不会真"重复"; 搜索应当尊重用户 query 意图.
    def _is_noise(meta):
        return bool(meta.get("resolved", False) and meta.get("importance", 5) == 1)
    matches = [b for b in matches
               if not (is_internalized(b["metadata"])
                       or _is_noise(b["metadata"])
                       or b["metadata"].get("type") == "feel")]

    # --- Vector similarity channel: find semantically related buckets ---
    # --- 向量相似度通道：找到语义相关的桶 ---
    matched_ids = {b["id"] for b in matches}
    try:
        vector_results = await embedding_engine.search_similar(query, top_k=max(max_results, 20))
        for bucket_id, sim_score in vector_results:
            if bucket_id not in matched_ids and sim_score > 0.5:
                bucket = await bucket_mgr.get(bucket_id)
                # 注: 排除钉选/永久参考桶 (protected 或 highlight) — 语义相似正是用户不想要的
                #   "模糊命中占记忆位"。它们已在开窗核心准则/永久参考区读取; title 强命中的情况
                #   已由上面关键词通道 (search()) 收进 matches, 这里只补"keyword 没命中但语义相近"
                #   的桶, 它们不该靠这条旁路混入。
                if bucket and not (is_internalized(bucket["metadata"])
                                   or is_highlighted(bucket["metadata"])
                                   or is_protected(bucket["metadata"])
                                   or _is_noise(bucket["metadata"])
                                   or bucket["metadata"].get("type") == "feel"):
                    bucket["score"] = round(sim_score * 100, 2)
                    bucket["vector_match"] = True
                    matches.append(bucket)
                    matched_ids.add(bucket_id)
    except Exception as e:
        logger.warning(f"Vector search failed, using keyword only / 向量搜索失败: {e}")

    results = []
    token_used = 0
    for bucket in matches:
        if token_used >= max_tokens:
            break
        try:
            clean_meta = {k: v for k, v in bucket["metadata"].items() if k != "tags"}
            # --- Memory reconstruction: shift displayed valence by current mood ---
            # --- 记忆重构：根据当前情绪微调展示层 valence（±0.1）---
            if q_valence is not None and "valence" in clean_meta:
                original_v = float(clean_meta.get("valence", 0.5))
                shift = (q_valence - 0.5) * 0.2  # ±0.1 max shift
                clean_meta["valence"] = max(0.0, min(1.0, original_v + shift))
            summary = await dehydrator.dehydrate(strip_wikilinks(bucket["content"]), clean_meta)
            summary_tokens = count_tokens_approx(summary)
            if token_used + summary_tokens > max_tokens:
                break
            await bucket_mgr.touch(bucket["id"])
            if bucket.get("vector_match"):
                summary = f"[语义关联] [bucket_id:{bucket['id']}] {summary}"
            else:
                summary = f"[bucket_id:{bucket['id']}] {summary}"
            results.append(summary)
            token_used += summary_tokens
        except Exception as e:
            logger.warning(f"Failed to dehydrate search result / 检索结果脱水失败: {e}")
            continue

    # --- Random surfacing: when search returns < 3, 40% chance to float old memories ---
    # --- 随机浮现：检索结果不足 3 条时，40% 概率从低权重旧桶里漂上来 ---
    if len(matches) < 3 and random.random() < 0.4:
        try:
            all_buckets = await bucket_mgr.list_all(include_archive=False)
            matched_ids = {b["id"] for b in matches}
            low_weight = [
                b for b in all_buckets
                if b["id"] not in matched_ids
                and decay_engine.calculate_score(b["metadata"]) < 2.0
                and not is_internalized(b["metadata"])
                and b["metadata"].get("type") != "feel"
            ]
            if low_weight:
                drifted = random.sample(low_weight, min(random.randint(1, 3), len(low_weight)))
                drift_results = []
                for b in drifted:
                    clean_meta = {k: v for k, v in b["metadata"].items() if k != "tags"}
                    summary = await dehydrator.dehydrate(strip_wikilinks(b["content"]), clean_meta)
                    drift_results.append(f"[surface_type: random]\n{summary}")
                results.append("--- 忽然想起来 ---\n" + "\n---\n".join(drift_results))
        except Exception as e:
            logger.warning(f"Random surfacing failed / 随机浮现失败: {e}")

    if not results:
        return "未找到相关记忆。"

    return "\n---\n".join(results)


# =============================================================
# Tool 2: hold — Hold on to this
# 工具 2：hold — 握住，留下来
# =============================================================
@mcp.tool()
async def hold(
    content: str,
    tags: str = "",
    importance: int = 5,
    pinned: bool = False,
    feel: bool = False,
    source_bucket: str = "",    valence: float = -1,
    arousal: float = -1,
    event_time: str = "",
) -> str:
    """存储单条记忆——对话里出现值得跨对话记住的事实/事件/约定就主动调用(别等用户开口要)。自动打标+合并。tags逗号分隔,importance 1-10。pinned=True创建永久钉选桶。feel=True存储你的第一人称感受(不参与普通浮现)。source_bucket=被你内化的记忆桶ID(feel模式下,标记源记忆为已内化,从此不再浮现)。event_time=事件实际发生时间(YYYY-MM-DD 或 ISO 时间戳),不传默认就是现在。当用户提到的事件不是发生在现在时(如"上周末""昨晚""三月那次"),应当传 event_time 而非默认。"""
    await decay_engine.ensure_started()

    # --- Input validation / 输入校验 ---
    if not content or not content.strip():
        return "内容为空，无法存储。"

    importance = max(1, min(10, importance))
    extra_tags = [t.strip() for t in tags.split(",") if t.strip()]

    # --- Feel mode: store as feel type, minimal metadata ---
    # --- Feel 模式：存为 feel 类型，最少元数据 ---
    if feel:
        # Feel valence/arousal = model's own perspective
        feel_valence = valence if 0 <= valence <= 1 else 0.5
        feel_arousal = arousal if 0 <= arousal <= 1 else 0.3
        bucket_id = await bucket_mgr.create(
            content=content,
            tags=[],
            importance=5,
            domain=[],
            valence=feel_valence,
            arousal=feel_arousal,
            name=None,
            bucket_type="feel",
        )
        try:
            await embedding_engine.generate_and_store(bucket_id, content)
        except Exception:
            pass
        # --- Mark source memory as internalized + store model's valence perspective ---
        # --- 标记源记忆为已内化 + 存储模型视角的 valence ---
        if source_bucket and source_bucket.strip():
            try:
                update_kwargs = {"internalized": True}
                if 0 <= valence <= 1:
                    update_kwargs["model_valence"] = feel_valence
                await bucket_mgr.update(source_bucket.strip(), **update_kwargs)
            except Exception as e:
                logger.warning(f"Failed to mark source as internalized / 标记已内化失败: {e}")
        return f"🫧feel→{bucket_id}"

    # --- Step 1: auto-tagging / 自动打标 ---
    try:
        analysis = await dehydrator.analyze(content)
    except Exception as e:
        logger.warning(f"Auto-tagging failed, using defaults / 自动打标失败: {e}")
        analysis = {
            "domain": ["未分类"], "valence": 0.5, "arousal": 0.3,
            "tags": [], "suggested_name": "",
        }

    domain = analysis["domain"]
    valence = analysis["valence"]
    arousal = analysis["arousal"]
    auto_tags = analysis["tags"]
    suggested_name = analysis.get("suggested_name", "")

    all_tags = list(dict.fromkeys(auto_tags + extra_tags))

    # --- Pinned buckets bypass merge and are created directly in permanent dir ---
    # --- 钉选桶跳过合并，直接新建到 permanent 目录 ---
    if pinned:
        bucket_id = await bucket_mgr.create(
            content=content,
            tags=all_tags,
            importance=10,
            domain=domain,
            valence=valence,
            arousal=arousal,
            name=suggested_name or None,
            bucket_type="permanent",
            pinned=True,
            event_time=event_time,
        )
        try:
            await embedding_engine.generate_and_store(bucket_id, content)
        except Exception:
            pass
        return f"📌钉选→{bucket_id} {','.join(domain)}"

    # --- Step 2: merge or create / 合并或新建 ---
    result_name, is_merged = await _merge_or_create(
        content=content,
        tags=all_tags,
        importance=importance,
        domain=domain,
        valence=valence,
        arousal=arousal,
        name=suggested_name,
        event_time=event_time or None,
    )

    action = "合并→" if is_merged else "新建→"
    return f"{action}{result_name} {','.join(domain)}"


# =============================================================
# Tool 3: grow — Grow, fragments become memories
# 工具 3：grow — 生长，一天的碎片长成记忆
# =============================================================
@mcp.tool()
async def grow(content: str, event_time: str = "") -> str:
    """日记归档,自动拆分为多桶。短内容(<30字)走快速路径。event_time=这篇日记记录的事件发生时间(YYYY-MM-DD 或 ISO),不传默认就是现在。整篇日记拆出的所有桶会共享这个 event_time(因为本来就是"那天发生的事")。"""
    await decay_engine.ensure_started()

    if not content or not content.strip():
        return "内容为空，无法整理。"

    # --- Short content fast path: skip digest, use hold logic directly ---
    # --- 短内容快速路径：跳过 digest 拆分，直接走 hold 逻辑省一次 API ---
    # For very short inputs (like "1"), calling digest is wasteful:
    # it sends the full DIGEST_PROMPT (~800 tokens) to DeepSeek for nothing.
    # Instead, run analyze + create directly.
    if len(content.strip()) < 30:
        logger.info(f"grow short-content fast path: {len(content.strip())} chars")
        try:
            analysis = await dehydrator.analyze(content)
        except Exception as e:
            logger.warning(f"Fast-path analyze failed / 快速路径打标失败: {e}")
            analysis = {
                "domain": ["未分类"], "valence": 0.5, "arousal": 0.3,
                "tags": [], "suggested_name": "",
            }
        result_name, is_merged = await _merge_or_create(
            content=content.strip(),
            tags=analysis.get("tags", []),
            importance=analysis.get("importance", 5) if isinstance(analysis.get("importance"), int) else 5,
            domain=analysis.get("domain", ["未分类"]),
            valence=analysis.get("valence", 0.5),
            arousal=analysis.get("arousal", 0.3),
            name=analysis.get("suggested_name", ""),
            event_time=event_time or None,
        )
        action = "合并" if is_merged else "新建"
        return f"{action} → {result_name} | {','.join(analysis.get('domain', []))} V{analysis.get('valence', 0.5):.1f}/A{analysis.get('arousal', 0.3):.1f}"

    # --- Step 1: let API split and organize / 让 API 拆分整理 ---
    try:
        items = await dehydrator.digest(content)
    except Exception as e:
        logger.error(f"Diary digest failed / 日记整理失败, 退化为整段存一条: {e}")
        items = None

    # --- 兜底: 拆分失败/为空也【绝不丢内容】---
    # grow 原本失败直接 return 错误, 那段内容就彻底消失("以为存了其实没存")。
    # 改为: 整段当一条记忆存下来, 至少不丢; 用户回看时还能再 redehydrate/拆。
    if not items:
        logger.warning("grow digest 为空, 整段存为单条记忆 (兜底防丢)")
        try:
            analysis = await dehydrator.analyze(content)
        except Exception:
            analysis = {"domain": ["未分类"], "valence": 0.5, "arousal": 0.3,
                        "tags": [], "suggested_name": ""}
        result_name, _is_merged = await _merge_or_create(
            content=content.strip(),
            tags=analysis.get("tags", []),
            importance=analysis.get("importance", 5) if isinstance(analysis.get("importance"), int) else 5,
            domain=analysis.get("domain", ["未分类"]),
            valence=analysis.get("valence", 0.5),
            arousal=analysis.get("arousal", 0.3),
            name=analysis.get("suggested_name", ""),
            event_time=event_time or None,
        )
        return f"⚠ 自动拆分失败,已【整段存为一条】记忆(未拆分,内容没丢)→ {result_name}"

    results = []
    created = 0
    merged = 0

    # --- Step 2: merge or create each item (with per-item error handling) ---
    # --- 逐条合并或新建（单条失败不影响其他）---
    for item in items:
        try:
            result_name, is_merged = await _merge_or_create(
                content=item["content"],
                tags=item.get("tags", []),
                importance=item.get("importance", 5),
                domain=item.get("domain", ["未分类"]),
                valence=item.get("valence", 0.5),
                arousal=item.get("arousal", 0.3),
                name=item.get("name", ""),
                event_time=event_time or None,
            )

            if is_merged:
                results.append(f"📎{result_name}")
                merged += 1
            else:
                results.append(f"📝{item.get('name', result_name)}")
                created += 1
        except Exception as e:
            logger.warning(
                f"Failed to process diary item / 日记条目处理失败: "
                f"{item.get('name', '?')}: {e}"
            )
            results.append(f"⚠️{item.get('name', '?')}")

    return f"{len(items)}条|新{created}合{merged}\n" + "\n".join(results)


# =============================================================
# Tool 4: trace — Trace, redraw the outline of a memory
# 工具 4：trace — 描摹，重新勾勒记忆的轮廓
# Also handles deletion (delete=True)
# 同时承接删除功能
# =============================================================
@mcp.tool()
async def trace(
    bucket_id: str,
    name: str = "",
    domain: str = "",
    valence: float = -1,
    arousal: float = -1,
    importance: int = -1,
    tags: str = "",
    resolved: int = -1,
    protected: int = -1,
    highlight: int = -1,
    pinned: int = -1,  # 老字段名,等价 protected=1 + highlight=1
    internalized: int = -1,
    digested: int = -1,  # 老字段名,兼容历史调用方,语义同 internalized
    event_time: str = "",
    content: str = "",
    delete: bool = False,
) -> str:
    """修改记忆元数据或内容。resolved=1归档(移入归档区→不再浮现、也不再被检索;可在 dashboard 归档区查看/恢复)/0取消归档标记,protected=1防衰减/0取消,highlight=1浮现优先/0取消,internalized=1隐藏(留在原地但不浮现/不检索)/0取消,event_time=纠正事件实际发生时间(YYYY-MM-DD 或 ISO,空字符串=清除该字段),content=替换桶正文,delete=True删除。只传需改的,-1或空=不改。pinned 是 protected+highlight 的旧组合别名;digested 是 internalized 旧名,仍可用。"""

    if not bucket_id or not bucket_id.strip():
        return "请提供有效的 bucket_id。"

    # --- 闸门:用户手写的桶,AI 没权限改/删/归档 ---
    # --- created_by="user" 是 dashboard 新建桶时打的标记,代表"这是用户手写的事实",
    #     你只能引用,不能改写 / 删除 / 归档。需要修改请告诉用户去 dashboard 改 ---
    bucket = await bucket_mgr.get(bucket_id)
    if not bucket:
        return f"未找到记忆桶: {bucket_id}"
    if bucket.get("metadata", {}).get("created_by") == "user":
        return (
            f"记忆桶 {bucket_id} 是用户手动写入的,你没有权限修改/删除/归档,"
            f"只能引用。如有需要,告诉用户去 dashboard 调整。"
        )

    # --- Delete mode / 删除模式 ---
    if delete:
        success = await bucket_mgr.delete(bucket_id)
        if success:
            embedding_engine.delete_embedding(bucket_id)
        return f"已遗忘记忆桶: {bucket_id}" if success else f"未找到记忆桶: {bucket_id}"

    # --- Collect only fields actually passed / 只收集用户实际传入的字段 ---
    updates = {}
    if name:
        updates["name"] = name
    if domain:
        updates["domain"] = [d.strip() for d in domain.split(",") if d.strip()]
    if 0 <= valence <= 1:
        updates["valence"] = valence
    if 0 <= arousal <= 1:
        updates["arousal"] = arousal
    if 1 <= importance <= 10:
        updates["importance"] = importance
    if tags:
        updates["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
    if resolved in (0, 1):
        updates["resolved"] = bool(resolved)
    # protected/highlight 是新字段,pinned 是老组合别名(=两个都设)
    if protected in (0, 1):
        updates["protected"] = bool(protected)
        if protected == 1:
            updates["importance"] = 10  # protected → lock importance
    if highlight in (0, 1):
        updates["highlight"] = bool(highlight)
    if pinned in (0, 1) and protected == -1 and highlight == -1:
        updates["protected"] = bool(pinned)
        updates["highlight"] = bool(pinned)
        if pinned == 1:
            updates["importance"] = 10
    # internalized 优先,digested 是兼容老调用方的别名
    if internalized in (0, 1):
        updates["internalized"] = bool(internalized)
    elif digested in (0, 1):
        updates["internalized"] = bool(digested)
    # event_time 透传到 bucket_mgr.update,内部会做格式校验
    # 空字符串语义=清掉该字段,非空但非法的会被规范化函数返回 None 然后清掉
    if event_time != "":
        updates["event_time"] = event_time
    if content:
        updates["content"] = content

    if not updates:
        return "没有任何字段需要修改。"

    success = await bucket_mgr.update(bucket_id, **updates)
    if not success:
        return f"修改失败: {bucket_id}"

    # Re-generate embedding if content changed
    if "content" in updates:
        try:
            await embedding_engine.generate_and_store(bucket_id, updates["content"])
        except Exception:
            pass

    changed = ", ".join(f"{k}={v}" for k, v in updates.items() if k != "content")
    if "content" in updates:
        changed += (", content=已替换" if changed else "content=已替换")
    # Explicit hint about resolved state change semantics
    # 特别提示 resolved 状态变化的语义
    if "resolved" in updates:
        if updates["resolved"]:
            changed += " → 已归档，不再参与浮现/检索（可在 dashboard 归档区查看或恢复）"
        else:
            changed += " → 已取消归档标记，将重新参与浮现排序"
    if "internalized" in updates:
        if updates["internalized"]:
            changed += " → 已内化，保留但不再浮现"
        else:
            changed += " → 已取消内化，重新参与浮现"
    return f"已修改记忆桶 {bucket_id}: {changed}"


# =============================================================
# Tool 5: pulse — Heartbeat, system status + memory listing
# 工具 5：pulse — 脉搏，系统状态 + 记忆列表
# =============================================================
@mcp.tool()
async def pulse(include_archive: bool = False) -> str:
    """查看系统状态 + 记忆桶列表。仅当用户主动问起"你都记得什么 / 记忆系统状态"时才调用,别在普通对话里随手调。include_archive=True 含归档。"""
    try:
        stats = await bucket_mgr.get_stats()
    except Exception as e:
        return f"获取系统状态失败: {e}"

    status = (
        f"=== Ombre Brain 记忆系统 ===\n"
        f"固化记忆桶: {stats['permanent_count']} 个\n"
        f"动态记忆桶: {stats['dynamic_count']} 个\n"
        f"归档记忆桶: {stats['archive_count']} 个\n"
        f"总存储大小: {stats['total_size_kb']:.1f} KB\n"
        f"衰减引擎: {'运行中' if decay_engine.is_running else '已停止'}\n"
    )

    # --- List all bucket summaries / 列出所有桶摘要 ---
    try:
        buckets = await bucket_mgr.list_all(include_archive=include_archive)
    except Exception as e:
        return status + f"\n列出记忆桶失败: {e}"

    if not buckets:
        return status + "\n记忆库为空。"

    lines = []
    for b in buckets:
        meta = b.get("metadata", {})
        if meta.get("pinned") or meta.get("protected"):
            icon = "📌"
        elif meta.get("type") == "permanent":
            icon = "📦"
        elif meta.get("type") == "feel":
            icon = "🫧"
        elif meta.get("type") == "archived":
            icon = "🗄️"
        elif meta.get("resolved", False):
            icon = "✅"
        else:
            icon = "💭"
        try:
            score = decay_engine.calculate_score(meta)
        except Exception:
            score = 0.0
        domains = ",".join(meta.get("domain", []))
        val = meta.get("valence", 0.5)
        aro = meta.get("arousal", 0.3)
        resolved_tag = " [已解决]" if meta.get("resolved", False) else ""
        lines.append(
            f"{icon} [{meta.get('name', b['id'])}]{resolved_tag} "
            f"bucket_id:{b['id']} "
            f"主题:{domains} "
            f"情感:V{val:.1f}/A{aro:.1f} "
            f"重要:{meta.get('importance', '?')} "
            f"权重:{score:.2f} "
            f"标签:{','.join(meta.get('tags', []))}"
        )

    return status + "\n=== 记忆列表 ===\n" + "\n".join(lines)


# =============================================================
# Tool 6: dream — Dreaming, digest recent memories
# 工具 6：dream — 做梦，消化最近的记忆
#
# Reads recent surface-level buckets (≤10), returns them for
# Claude to introspect under prompt guidance.
# 读取最近新增的表层桶（≤10个），返回给 Claude 在提示词引导下自主思考。
# Claude then decides: resolve some, write feels, or do nothing.
# =============================================================
@mcp.tool()
async def dream() -> str:
    """做梦——读取最近新增的记忆桶,供你自省。读完后可以trace(resolved=1)放下,或hold(feel=True)写感受。"""
    await decay_engine.ensure_started()

    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
    except Exception as e:
        logger.error(f"Dream failed to list buckets: {e}")
        return "记忆系统暂时无法访问。"

    # --- Filter: recent surface-level dynamic buckets (not permanent/pinned/feel/noise) ---
    # noise = resolved + importance=1, 用户软删除标记, 不应该被 AI dream 翻出来
    candidates = [
        b for b in all_buckets
        if b["metadata"].get("type") not in ("permanent", "feel")
        and not b["metadata"].get("pinned", False)
        and not b["metadata"].get("protected", False)
        and not (b["metadata"].get("resolved", False) and b["metadata"].get("importance", 5) == 1)
    ]

    # --- Sort by creation time desc, take top 10 ---
    candidates.sort(key=lambda b: b["metadata"].get("created", ""), reverse=True)
    recent = candidates[:10]

    if not recent:
        return "没有需要消化的新记忆。"

    parts = []
    for b in recent:
        meta = b["metadata"]
        resolved_tag = " [已解决]" if meta.get("resolved", False) else " [未解决]"
        domains = ",".join(meta.get("domain", []))
        val = meta.get("valence", 0.5)
        aro = meta.get("arousal", 0.3)
        created = meta.get("created", "")
        parts.append(
            f"[{meta.get('name', b['id'])}]{resolved_tag} "
            f"主题:{domains} V{val:.1f}/A{aro:.1f} "
            f"创建:{created}\n"
            f"ID: {b['id']}\n"
            f"{strip_wikilinks(b['content'][:500])}"
        )

    header = (
        "=== Dreaming ===\n"
        "以下是你最近的记忆。用第一人称想：\n"
        "- 这些东西里有什么在你这里留下了重量？\n"
        "- 有什么还没想清楚？\n"
        "- 有什么可以放下了？\n"
        "想完之后：值得放下的用 trace(bucket_id, resolved=1)；\n"
        "有沉淀的用 hold(content=\"...\", feel=True, source_bucket=\"bucket_id\", valence=你的感受) 写下来。\n"
        "valence 是你对这段记忆的感受，不是事件本身的情绪。\n"
        "没有沉淀就不写，不强迫产出。\n"
    )

    # --- Connection hint: find most similar pair via embeddings ---
    connection_hint = ""
    if embedding_engine and embedding_engine.enabled and len(recent) >= 2:
        try:
            best_pair = None
            best_sim = 0.0
            ids = [b["id"] for b in recent]
            names = {b["id"]: b["metadata"].get("name", b["id"]) for b in recent}
            embeddings = {}
            for bid in ids:
                emb = await embedding_engine.get_embedding(bid)
                if emb is not None:
                    embeddings[bid] = emb
            for i, id_a in enumerate(ids):
                for id_b in ids[i+1:]:
                    if id_a in embeddings and id_b in embeddings:
                        sim = embedding_engine._cosine_similarity(embeddings[id_a], embeddings[id_b])
                        if sim > best_sim:
                            best_sim = sim
                            best_pair = (id_a, id_b)
            if best_pair and best_sim > 0.5:
                connection_hint = (
                    f"\n💭 [{names[best_pair[0]]}] 和 [{names[best_pair[1]]}] "
                    f"似乎有关联 (相似度:{best_sim:.2f})——不替你下结论，你自己想。\n"
                )
        except Exception as e:
            logger.warning(f"Dream connection hint failed: {e}")

    # --- Feel crystallization hint: detect repeated feel themes ---
    crystal_hint = ""
    if embedding_engine and embedding_engine.enabled:
        try:
            feels = [b for b in all_buckets if b["metadata"].get("type") == "feel"]
            if len(feels) >= 3:
                feel_embeddings = {}
                for f in feels:
                    emb = await embedding_engine.get_embedding(f["id"])
                    if emb is not None:
                        feel_embeddings[f["id"]] = emb
                # Find clusters: feels with similarity > 0.7 to at least 2 others
                for fid, femb in feel_embeddings.items():
                    similar_feels = []
                    for oid, oemb in feel_embeddings.items():
                        if oid != fid:
                            sim = embedding_engine._cosine_similarity(femb, oemb)
                            if sim > 0.7:
                                similar_feels.append(oid)
                    if len(similar_feels) >= 2:
                        feel_bucket = next((f for f in feels if f["id"] == fid), None)
                        if feel_bucket and not feel_bucket["metadata"].get("pinned"):
                            content_preview = strip_wikilinks(feel_bucket["content"][:80])
                            crystal_hint = (
                                f"\n🔮 你已经写过 {len(similar_feels)+1} 条相似的 feel "
                                f"（围绕「{content_preview}…」）。"
                                f"如果这已经是确信而不只是感受了，"
                                f"你可以用 hold(content=\"...\", pinned=True) 升级它。"
                                f"不急，你自己决定。\n"
                            )
                            break
        except Exception as e:
            logger.warning(f"Dream crystallization hint failed: {e}")

    return header + "\n---\n".join(parts) + connection_hint + crystal_hint


# =============================================================
# Runtime config endpoints — 前端 config 页面切 API 用
# 持久化:{buckets_dir}/runtime_config.json
# 加载链:runtime_config.json > env vars > config.yaml > 默认
# 安全:API key 只在 GET 时返回 mask 形式;dehydrator 实例热重载
# =============================================================
import json as _json_cfg

def _runtime_config_path():
    return os.path.join(config.get("buckets_dir", "./buckets"), "runtime_config.json")

def _read_runtime_config():
    p = _runtime_config_path()
    if not os.path.exists(p):
        return {"active": None, "profiles": {}}
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = _json_cfg.load(f)
        if not isinstance(d, dict):
            return {"active": None, "profiles": {}}
        d.setdefault("active", None)
        d.setdefault("profiles", {})
        return d
    except Exception as e:
        logger.warning(f"runtime_config.json read fail: {e}")
        return {"active": None, "profiles": {}}

def _write_runtime_config(rc: dict):
    p = _runtime_config_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        _json_cfg.dump(rc, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)

def _mask_key(k: str) -> str:
    if not k or len(k) < 12:
        return "***" if k else ""
    return k[:6] + "..." + k[-4:]

def _reload_dehydrator_from_runtime():
    """读 runtime_config + 应用 active profile 到 dehydrator 实例。"""
    rc = _read_runtime_config()
    active_id = rc.get("active")
    profiles = rc.get("profiles", {})
    fresh_cfg = {"dehydration": dict(config.get("dehydration", {})), "buckets_dir": config.get("buckets_dir")}
    if active_id and active_id in profiles:
        p = profiles[active_id]
        if p.get("api_key"): fresh_cfg["dehydration"]["api_key"] = p["api_key"]
        if p.get("base_url"): fresh_cfg["dehydration"]["base_url"] = p["base_url"]
        if p.get("model"): fresh_cfg["dehydration"]["model"] = p["model"]
    dehydrator.reload(fresh_cfg)

def _reload_decay_from_runtime():
    """读 runtime_config['decay'] → decay_engine.apply_runtime_overrides。
    启动 + 每次 POST /api/decay-config 后调一次, 立刻生效到下次 score 计算。"""
    rc = _read_runtime_config()
    decay_overrides = rc.get("decay") or {}
    if isinstance(decay_overrides, dict):
        decay_engine.apply_runtime_overrides(decay_overrides)


def _reload_prompts_from_runtime():
    """读 runtime_config['prompts'] → dehydrator.set_prompts。
    启动 + 每次 POST /api/prompts-config 后调一次。
    runtime_config['prompts'] 形如 {key: prompt_str}, 缺 key/空串都视为用默认。"""
    import dehydrator as _dh
    rc = _read_runtime_config()
    prompt_overrides = rc.get("prompts") or {}
    if isinstance(prompt_overrides, dict):
        _dh.set_prompts(prompt_overrides)


def _reload_scoring_from_runtime():
    """读 runtime_config['scoring'] → bucket_mgr.apply_runtime_scoring_overrides。
    启动 + 每次 POST /api/scoring-config 后调一次, 立刻生效到下次 search()。"""
    rc = _read_runtime_config()
    scoring_overrides = rc.get("scoring") or {}
    if isinstance(scoring_overrides, dict):
        bucket_mgr.apply_runtime_scoring_overrides(scoring_overrides)


@mcp.custom_route("/api/decay-config", methods=["GET"])
async def api_decay_config_get(request):
    """读当前 decay 各参数当前值 + 出厂默认 + 范围/标签 schema (前端 slider 用)。"""
    from starlette.responses import JSONResponse
    # 字段元信息(前端 slider 显示用); 范围 / step / 中文名 / 解释
    schema = [
        {"key": "feel_score",            "label": "feel 基础权重",     "min": 0,    "max": 100,  "step": 1,    "hint": "0=跟随 importance, 锁定值则 feel 桶恒定 score"},
        {"key": "protected_score",       "label": "protected 上限",     "min": 50,   "max": 200,  "step": 1,    "hint": "钉决/永久桶的 score(原硬编 999)"},
        {"key": "highlight_boost_pct",   "label": "highlight 加成 %",   "min": 0,    "max": 50,   "step": 1,    "hint": "标重要的桶 score 上浮 X%"},
        {"key": "surface_threshold",     "label": "浮现阈值",            "min": 1,    "max": 80,   "step": 1,    "hint": "score 高于此值标记为活跃(UI 提示)"},
        {"key": "archive_threshold",     "label": "归档阈值",            "min": 0.01, "max": 2.0,  "step": 0.01, "hint": "score 低于此值自动归档"},
        {"key": "decay_lambda",          "label": "衰减速率 λ",          "min": 0.01, "max": 0.30, "step": 0.01, "hint": "时间衰减斜率(大=衰得快)"},
        {"key": "arousal_boost",         "label": "arousal 加成",        "min": 0.0,  "max": 2.0,  "step": 0.1,  "hint": "高 arousal 提升 emotion_weight 系数"},
        {"key": "emotion_base",          "label": "情感基线",            "min": 0.5,  "max": 2.0,  "step": 0.1,  "hint": "情感权重最低值(arousal=0 时)"},
        {"key": "resolved_factor",       "label": "噪声衰减系数",        "min": 0.01, "max": 0.50, "step": 0.01, "hint": "标 resolved/noise 时 score × 此系数"},
        {"key": "internalized_resolved_factor", "label": "已内化+噪声系数", "min": 0.01, "max": 0.50, "step": 0.01, "hint": "双标时更激进的衰减系数"},
    ]
    return JSONResponse({
        "current": decay_engine.current_overrides(),
        "defaults": dict(DecayEngine.DEFAULTS),
        "schema": schema,
    })


@mcp.custom_route("/api/decay-config", methods=["POST"])
async def api_decay_config_post(request):
    """更新 decay 参数。body 是部分或全部 key→value 字典, 只更新传入的字段;
    存到 runtime_config.json['decay'], 立刻 reload decay_engine。"""
    from starlette.responses import JSONResponse
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "body 必须是对象"}, status_code=400)
    rc = _read_runtime_config()
    cur_decay = rc.get("decay") or {}
    if not isinstance(cur_decay, dict):
        cur_decay = {}
    # 只接受白名单字段, 强制 float
    for k in DecayEngine.DEFAULTS.keys():
        if k in body:
            try:
                cur_decay[k] = float(body[k])
            except (TypeError, ValueError):
                pass
    rc["decay"] = cur_decay
    _write_runtime_config(rc)
    _reload_decay_from_runtime()
    return JSONResponse({
        "ok": True,
        "current": decay_engine.current_overrides(),
    })


@mcp.custom_route("/api/decay-config/reset", methods=["POST"])
async def api_decay_config_reset(request):
    """全部恢复出厂默认。清掉 runtime_config['decay'] 后 reload。"""
    from starlette.responses import JSONResponse
    rc = _read_runtime_config()
    rc["decay"] = {}
    _write_runtime_config(rc)
    _reload_decay_from_runtime()
    return JSONResponse({
        "ok": True,
        "current": decay_engine.current_overrides(),
    })


# =============================================================
# Scoring config — 检索打分微调(title 命中加分 / 关键词优先排序 / dryrun 日志)
# 设计: 默认全 0/False → 跟原 search 行为 100% 一致(开源/上游零变化);
#       用户 runtime 设值才生效。pattern 完全 mirror /api/decay-config。
# =============================================================

_SCORING_SCHEMA = [
    {
        "key": "content_weight",
        "label": "正文检索权重",
        "type": "float",
        "min": 0, "max": 10, "step": 0.5,
        "hint": "正文字段在检索里的权重(默认 1.0 对齐上游)。调高(如 3.0)让'正文里写过的内容'也能被搜到('我写过却搜不到'的解药); 太高可能让正文相似但主题不同的桶误命中",
    },
    {
        "key": "title_hit_bonus",
        "label": "title 命中加分",
        "type": "float",
        "min": 0, "max": 100, "step": 1,
        "hint": "query 在桶名命中时给 final score 加此分(0=关), 推荐 +15~+30; 调高让 title 命中桶顶上去",
    },
    {
        "key": "keyword_first_sort",
        "label": "关键词命中优先排序",
        "type": "bool",
        "hint": "开启时 title 命中的桶整体排到非 title 命中之前(bonus 不够压时的兜底)",
    },
    {
        "key": "dryrun_log",
        "label": "dryrun 日志",
        "type": "bool",
        "hint": "每次 search 打印 top-10 详细到服务日志, 调权重时打开看效果",
    },
    {
        "key": "precise_match_mode",
        "label": "严格关键词模式",
        "type": "bool",
        "hint": "开启后检索走严格 token 命中(query 按词切, 每个词在桶各字段做精确命中), 砍掉情感/时间/重要度/温度偏置。解决「长 query 模糊匹配错乱」和「高 valence 桶没关键词也排前」。默认关=模糊匹配(上游兼容)",
    },
    {
        "key": "warmth_boost",
        "label": "温度偏置",
        "type": "float",
        "min": 0, "max": 5, "step": 0.5,
        "hint": "给高 valence(>0.5 的「温暖」桶)检索时额外加分, 跟 query 带不带情感无关——让温暖的记忆天然更易浮现。默认 0=关(零行为变化); 调到 2.0 ≈ 给 valence=0.9 桶加约 1/5 个主题命中分",
    },
]


@mcp.custom_route("/api/scoring-config", methods=["GET"])
async def api_scoring_config_get(request):
    """读当前 scoring 参数 + 出厂默认 + schema。"""
    from starlette.responses import JSONResponse
    return JSONResponse({
        "current": bucket_mgr.current_scoring_overrides(),
        "defaults": dict(bucket_mgr.SCORING_OVERRIDE_DEFAULTS),
        "schema": _SCORING_SCHEMA,
    })


@mcp.custom_route("/api/scoring-config", methods=["POST"])
async def api_scoring_config_post(request):
    """更新 scoring 参数。body 部分或全部 key→value, 存到 runtime_config['scoring'] 并立刻生效。"""
    from starlette.responses import JSONResponse
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "body 必须是对象"}, status_code=400)
    rc = _read_runtime_config()
    cur_scoring = rc.get("scoring") or {}
    if not isinstance(cur_scoring, dict):
        cur_scoring = {}
    # 白名单 + 类型强制
    for k, default in bucket_mgr.SCORING_OVERRIDE_DEFAULTS.items():
        if k not in body:
            continue
        v = body[k]
        if isinstance(default, bool):
            cur_scoring[k] = bool(v)
        else:
            try:
                cur_scoring[k] = float(v)
            except (TypeError, ValueError):
                pass
    rc["scoring"] = cur_scoring
    _write_runtime_config(rc)
    _reload_scoring_from_runtime()
    return JSONResponse({
        "ok": True,
        "current": bucket_mgr.current_scoring_overrides(),
    })


@mcp.custom_route("/api/scoring-config/reset", methods=["POST"])
async def api_scoring_config_reset(request):
    """全部恢复出厂默认 = 关掉所有 scoring 微调。清掉 runtime_config['scoring'] 后 reload。"""
    from starlette.responses import JSONResponse
    rc = _read_runtime_config()
    rc["scoring"] = {}
    _write_runtime_config(rc)
    _reload_scoring_from_runtime()
    return JSONResponse({
        "ok": True,
        "current": bucket_mgr.current_scoring_overrides(),
    })


# =============================================================
# Hit stats — 命中频次统计 (反向反馈"哪些桶被高频检索 / 哪些从未")
# v1: in-memory only, 重启清零 (UI 已说明)。未来可加 flush 到 hit_stats.json。
# =============================================================

@mcp.custom_route("/api/hit-stats", methods=["GET"])
async def api_hit_stats(request):
    """Return {total_searches, total_buckets, hit_buckets, zero_buckets, order,
    items: [{id, name, count, last_hit, last_query, gated, missing}, ...]}.
    Query params:
      limit        默认 50 (max 2000)
      include_zero =1 把从未命中的桶也并进来 (count 0) → 冷记忆视图
      order        desc(默认 高频在前) / asc(冷门在前)
      exclude_gated=1 排除钉选/永久参考/feel/已内化桶 (它们 ×0 是预期)"""
    from starlette.responses import JSONResponse
    qp = request.query_params
    try:
        limit = int(qp.get("limit", "50"))
    except ValueError:
        limit = 50

    def _truthy(v):
        return str(v).lower() in ("1", "true", "yes", "on")

    try:
        data = await bucket_mgr.get_hit_stats(
            limit=limit,
            include_zero=_truthy(qp.get("include_zero", "")),
            order=qp.get("order", "desc"),
            exclude_gated=_truthy(qp.get("exclude_gated", "")),
        )
        return JSONResponse(data)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/hit-stats/reset", methods=["POST"])
async def api_hit_stats_reset(request):
    """清零命中统计 — 实验型: 想看"清零后再用一段时间, 哪些桶又会被命中"。"""
    from starlette.responses import JSONResponse
    bucket_mgr.reset_hit_stats()
    return JSONResponse({"ok": True})


@mcp.custom_route("/api/recent-searches", methods=["GET"])
async def api_recent_searches(request):
    """Return list of recent search traces (newest first), capped 20.
    Query param: limit (default 10). 给前端"我这次发消息浮现了什么"看, 直击 dryrun_log 太难看的问题。"""
    from starlette.responses import JSONResponse
    try:
        limit = int(request.query_params.get("limit", "10"))
    except ValueError:
        limit = 10
    try:
        items = bucket_mgr.get_recent_searches(limit=limit)
        return JSONResponse({"items": items})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# =============================================================
# Prompts config — 让前端配置页直接编辑系统 prompt, 不动代码
# 数据流: GET → 当前生效 + 出厂默认 + schema(标签/说明)
#         POST {key: prompt_str | ""} → 写 runtime_config['prompts'] + 立刻生效
#         POST /reset → 全部回出厂; POST /reset {key} → 单个回出厂
# =============================================================

@mcp.custom_route("/api/prompts-config", methods=["GET"])
async def api_prompts_config_get(request):
    """返回 {defaults, current, overridden, schema} — 前端编辑界面初始化用"""
    from starlette.responses import JSONResponse
    import dehydrator as _dh
    return JSONResponse(_dh.get_prompts_state())


@mcp.custom_route("/api/prompts-config", methods=["POST"])
async def api_prompts_config_post(request):
    """更新 prompt 覆盖。body 是 {key: prompt_str} 部分或全部 key.
       - prompt_str 非空 → 写入覆盖
       - prompt_str 为 "" / null → 撤销该 key 的覆盖, 回到默认
       写到 runtime_config['prompts'], 立刻 reload 到 dehydrator。"""
    from starlette.responses import JSONResponse
    import dehydrator as _dh
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "body 必须是对象"}, status_code=400)
    rc = _read_runtime_config()
    cur_prompts = rc.get("prompts") or {}
    if not isinstance(cur_prompts, dict):
        cur_prompts = {}
    # 只接受白名单 key
    valid_keys = set(_dh._DEFAULT_PROMPTS.keys())
    for k, v in body.items():
        if k not in valid_keys:
            continue
        if v is None or (isinstance(v, str) and not v.strip()):
            cur_prompts.pop(k, None)  # 撤销
        elif isinstance(v, str):
            # 限个长度 (32K 字符, 远超任何合理 prompt)
            cur_prompts[k] = v[:32000]
    rc["prompts"] = cur_prompts
    _write_runtime_config(rc)
    _reload_prompts_from_runtime()
    return JSONResponse(_dh.get_prompts_state())


@mcp.custom_route("/api/prompts-config/align-upstream", methods=["POST"])
async def api_prompts_config_align_upstream(request):
    """一键把所有"上游有的"prompt 切到上游版本(作为运行时覆盖, 不改代码)。
    本项目独创的 prompt (redehydrate / regen_content) 上游没有, 跳过不动。
    持久化到 runtime_config['prompts']。"""
    from starlette.responses import JSONResponse
    import dehydrator as _dh
    result = _dh.align_to_upstream()
    # 持久化当前 active 到 runtime_config
    rc = _read_runtime_config()
    rc["prompts"] = dict(_dh._ACTIVE_PROMPTS)
    _write_runtime_config(rc)
    state = _dh.get_prompts_state()
    return JSONResponse({
        "ok": True,
        "aligned": result["aligned"],
        "skipped": result["skipped"],
        "state": state,
    })


@mcp.custom_route("/api/prompts-config/reset", methods=["POST"])
async def api_prompts_config_reset(request):
    """复位 — 默认全部恢复 (清掉 runtime_config['prompts'])。
       可选 body {"key": "dehydrate"} → 只复位单个 key."""
    from starlette.responses import JSONResponse
    import dehydrator as _dh
    target_key = None
    try:
        body = await request.json()
        if isinstance(body, dict):
            target_key = body.get("key")
    except Exception:
        pass
    rc = _read_runtime_config()
    cur_prompts = rc.get("prompts") or {}
    if not isinstance(cur_prompts, dict):
        cur_prompts = {}
    if target_key:
        if target_key in _dh._DEFAULT_PROMPTS:
            cur_prompts.pop(target_key, None)
    else:
        cur_prompts = {}
    rc["prompts"] = cur_prompts
    _write_runtime_config(rc)
    _reload_prompts_from_runtime()
    return JSONResponse(_dh.get_prompts_state())


@mcp.custom_route("/api/config/api", methods=["GET"])
async def api_config_get(request):
    """读取所有 API profile + 当前激活 + 当前生效配置。api_key 走 mask。"""
    from starlette.responses import JSONResponse
    rc = _read_runtime_config()
    profiles_out = []
    for pid, p in rc.get("profiles", {}).items():
        profiles_out.append({
            "id": pid,
            "name": p.get("name", pid),
            "model": p.get("model", ""),
            "base_url": p.get("base_url", ""),
            "api_key_mask": _mask_key(p.get("api_key", "")),
            "has_key": bool(p.get("api_key")),
        })
    return JSONResponse({
        "active": rc.get("active"),
        "profiles": profiles_out,
        # 当前 dehydrator 实例上真正生效的(如果 runtime 没设过,反映 env/yaml)
        "current_effective": {
            "model": dehydrator.model,
            "base_url": dehydrator.base_url,
            "api_key_mask": _mask_key(dehydrator.api_key),
            "api_available": dehydrator.api_available,
        },
    })


@mcp.custom_route("/api/config/api/profile", methods=["POST"])
async def api_config_profile_upsert(request):
    """新增或更新 profile。body: {id?, name, model, base_url, api_key?}
    id 没传就生成新的;api_key 留空 → 保留旧值(不覆盖)。"""
    from starlette.responses import JSONResponse
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    pid = body.get("id") or ""
    name = (body.get("name") or "").strip()
    model = (body.get("model") or "").strip()
    base_url = (body.get("base_url") or "").strip()
    api_key = body.get("api_key", "")  # 可能为空表示"不改"
    if not name or not model or not base_url:
        return JSONResponse({"error": "name / model / base_url 必填"}, status_code=400)
    rc = _read_runtime_config()
    profiles = rc.setdefault("profiles", {})
    if not pid:
        # 自增 id (id_001, id_002, ...)
        n = 1
        while f"p{n:03d}" in profiles:
            n += 1
        pid = f"p{n:03d}"
    existing = profiles.get(pid, {})
    profiles[pid] = {
        "name": name,
        "model": model,
        "base_url": base_url,
        "api_key": api_key if api_key else existing.get("api_key", ""),
    }
    _write_runtime_config(rc)
    return JSONResponse({"ok": True, "id": pid})


@mcp.custom_route("/api/config/api/profile/{pid}/delete", methods=["POST"])
async def api_config_profile_delete(request):
    from starlette.responses import JSONResponse
    pid = request.path_params["pid"]
    rc = _read_runtime_config()
    if pid not in rc.get("profiles", {}):
        return JSONResponse({"error": "profile 不存在"}, status_code=404)
    del rc["profiles"][pid]
    if rc.get("active") == pid:
        rc["active"] = None  # 删的是当前激活的 → 退回 env/yaml
        _write_runtime_config(rc)
        _reload_dehydrator_from_runtime()
    else:
        _write_runtime_config(rc)
    return JSONResponse({"ok": True})


@mcp.custom_route("/api/config/api/active", methods=["POST"])
async def api_config_set_active(request):
    """切换激活 profile。body: {id} (传 null/空 → 清空,回退到 env/yaml)。"""
    from starlette.responses import JSONResponse
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    pid = body.get("id") or None
    rc = _read_runtime_config()
    if pid and pid not in rc.get("profiles", {}):
        return JSONResponse({"error": "profile 不存在"}, status_code=404)
    rc["active"] = pid
    _write_runtime_config(rc)
    _reload_dehydrator_from_runtime()
    return JSONResponse({
        "ok": True,
        "active": pid,
        "current_effective": {
            "model": dehydrator.model,
            "base_url": dehydrator.base_url,
            "api_key_mask": _mask_key(dehydrator.api_key),
        },
    })


@mcp.custom_route("/api/config/strategy", methods=["GET"])
async def api_config_strategy_get(request):
    """读取当前生效的策略参数(merge_threshold / max_recall / auto_merge)。"""
    from starlette.responses import JSONResponse
    return JSONResponse({
        "merge_threshold": int(config.get("merge_threshold", 75)),
        "max_recall": int(config.get("matching", {}).get("max_results", 5)),
        "auto_merge": bool(config.get("auto_merge", True)),
    })


@mcp.custom_route("/api/config/strategy", methods=["POST"])
async def api_config_strategy_set(request):
    """更新策略参数。body: { merge_threshold?: 0~100, max_recall?: 1~50 }
    持久化到 runtime_config.json + 内存中 config dict 即时刷新(下次调用就生效)。"""
    from starlette.responses import JSONResponse
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    rc = _read_runtime_config()
    strategy = rc.setdefault("strategy", {})
    if "merge_threshold" in body:
        try:
            v = max(0, min(100, int(body["merge_threshold"])))
            strategy["merge_threshold"] = v
            config["merge_threshold"] = v
        except (ValueError, TypeError):
            return JSONResponse({"error": "merge_threshold 必须是 0-100 整数"}, status_code=400)
    if "auto_merge" in body:
        v = bool(body["auto_merge"])
        strategy["auto_merge"] = v
        config["auto_merge"] = v
    if "max_recall" in body:
        try:
            v = max(1, min(50, int(body["max_recall"])))
            strategy["max_recall"] = v
            config.setdefault("matching", {})["max_results"] = v
            # bucket_mgr 启动时把 max_results 缓存到了实例字段,顺手刷新
            try: bucket_mgr.max_results = v
            except Exception: pass
        except (ValueError, TypeError):
            return JSONResponse({"error": "max_recall 必须是 1-50 整数"}, status_code=400)
    _write_runtime_config(rc)
    return JSONResponse({
        "ok": True,
        "merge_threshold": int(config.get("merge_threshold", 75)),
        "max_recall": int(config.get("matching", {}).get("max_results", 5)),
    })


@mcp.custom_route("/api/config/api/test", methods=["POST"])
async def api_config_test(request):
    """测试一组配置能否连通。body: {model, base_url, api_key} 直接测;
    或 {id} 用已存的 profile 测。返回 {ok, latency_ms, error?, sample?}"""
    from starlette.responses import JSONResponse
    import time
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    pid = body.get("id")
    if pid:
        rc = _read_runtime_config()
        p = rc.get("profiles", {}).get(pid)
        if not p:
            return JSONResponse({"error": "profile 不存在"}, status_code=404)
        model = p["model"]; base_url = p["base_url"]; api_key = p.get("api_key", "")
    else:
        model = (body.get("model") or "").strip()
        base_url = (body.get("base_url") or "").strip()
        api_key = body.get("api_key", "")
    if not model or not base_url or not api_key:
        return JSONResponse({"error": "model / base_url / api_key 都需要"}, status_code=400)
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=20.0)
        t0 = time.time()
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=8,
            temperature=0.0,
        )
        latency = int((time.time() - t0) * 1000)
        sample = ""
        if resp.choices:
            sample = (resp.choices[0].message.content or "")[:60]
        return JSONResponse({"ok": True, "latency_ms": latency, "sample": sample})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)[:300]}, status_code=200)


# =============================================================
# Dashboard API endpoints (for lightweight Web UI)
# 仪表板 API（轻量 Web UI 用）
# =============================================================
@mcp.custom_route("/api/buckets", methods=["GET"])
async def api_buckets(request):
    """List all buckets with metadata (no content for efficiency).

    走 15s 内存缓存 — 多视图同时拉/用户切视图重复请求会瞬间命中,
    避免重复 IO 200+ frontmatter.load. 写操作 endpoint 会主动 invalidate.
    """
    from starlette.responses import JSONResponse
    now = time.monotonic()
    if _BUCKETS_CACHE["payload"] is not None and (now - _BUCKETS_CACHE["ts"]) < _BUCKETS_CACHE_TTL:
        return JSONResponse(_BUCKETS_CACHE["payload"])
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=True)
        result = []
        for b in all_buckets:
            meta = b.get("metadata", {})
            result.append({
                "id": b["id"],
                "name": meta.get("name", b["id"]),
                "type": meta.get("type", "dynamic"),
                "domain": meta.get("domain", []),
                "tags": meta.get("tags", []),
                "valence": meta.get("valence", 0.5),
                "arousal": meta.get("arousal", 0.3),
                "model_valence": meta.get("model_valence"),
                "importance": meta.get("importance", 5),
                "resolved": meta.get("resolved", False),
                "protected": is_protected(meta),
                "highlight": is_highlighted(meta),
                "pinned": is_protected(meta) or is_highlighted(meta),  # 兼容旧前端,等价老语义
                "internalized": is_internalized(meta),
                "digested": is_internalized(meta),  # 兼容旧前端
                "event_time": meta.get("event_time", ""),
                "created_by": meta.get("created_by", "ai"),  # 默认 ai,dashboard 手动新建会标 user
                "created": meta.get("created", ""),
                "last_active": meta.get("last_active", ""),
                "activation_count": meta.get("activation_count", 1),
                "score": decay_engine.calculate_score(meta),
                "summary": meta.get("summary", ""),  # 用户编辑过的摘要(v2 modal),空则前端回退用 content_preview
                "content_preview": strip_wikilinks(b.get("content", ""))[:200],
            })
        result.sort(key=lambda x: x["score"], reverse=True)
        _BUCKETS_CACHE["payload"] = result
        _BUCKETS_CACHE["ts"] = time.monotonic()
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/bucket/{bucket_id}", methods=["GET"])
async def api_bucket_detail(request):
    """Get full bucket content by ID."""
    from starlette.responses import JSONResponse
    bucket_id = request.path_params["bucket_id"]
    bucket = await bucket_mgr.get(bucket_id)
    if not bucket:
        return JSONResponse({"error": "not found"}, status_code=404)
    meta = bucket.get("metadata", {})
    return JSONResponse({
        "id": bucket["id"],
        "metadata": meta,
        "content": strip_wikilinks(bucket.get("content", "")),
        "score": decay_engine.calculate_score(meta),
    })


# =============================================================
# Bucket edit endpoints (slice 6 — dashboard 用户直接编辑入口)
# 这些端点把 trace 工具 + bucket_mgr.archive/unarchive/create 暴露给前端
# Dashboard 编辑模态框 / 归档按钮 / 新建桶按钮都通过这几个端点写
# =============================================================

@mcp.custom_route("/api/bucket/{bucket_id}/update", methods=["POST"])
async def api_bucket_update(request):
    """更新一条桶的元数据/内容。body 同 trace 工具的字段子集。"""
    from starlette.responses import JSONResponse
    bucket_id = request.path_params["bucket_id"]
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    bucket = await bucket_mgr.get(bucket_id)
    if not bucket:
        return JSONResponse({"error": "not found"}, status_code=404)

    # 透传到 bucket_mgr.update — 它内部已处理 protected/highlight 拆分、
    # internalized/digested 兼容、event_time 校验、pinned 别名等
    allowed = {
        "name", "domain", "tags", "valence", "arousal", "importance",
        "resolved", "protected", "highlight", "pinned",
        "internalized", "digested", "event_time", "content", "model_valence",
        "type",  # 支持 feel ↔ dynamic 切换(导入工作台 feel 开关)
        "summary",  # 用户可编辑的摘要(v2 modal),为空时回退到 content_preview
        "raw_source",  # 用户可手动补全/修订的原文片段(详情 modal 原文浮层编辑入口)
        "created_by",  # 来源分类 user/ai/import (历史 ai 桶可手动改成 import 等)
    }
    updates = {k: v for k, v in body.items() if k in allowed}
    if not updates:
        return JSONResponse({"error": "no valid fields", "allowed": sorted(allowed)}, status_code=400)

    # tags / domain 接受字符串(逗号分隔)或数组
    if isinstance(updates.get("tags"), str):
        updates["tags"] = [t.strip() for t in updates["tags"].split(",") if t.strip()]
    if isinstance(updates.get("domain"), str):
        updates["domain"] = [d.strip() for d in updates["domain"].split(",") if d.strip()]

    success = await bucket_mgr.update(bucket_id, **updates)
    if not success:
        return JSONResponse({"error": "update failed"}, status_code=500)
    _invalidate_buckets_cache()

    # content 改了顺手刷 embedding
    if "content" in updates and embedding_engine and embedding_engine.enabled:
        try:
            await embedding_engine.generate_and_store(bucket_id, updates["content"])
        except Exception:
            pass

    fresh = await bucket_mgr.get(bucket_id)
    return JSONResponse({
        "ok": True,
        "id": bucket_id,
        "metadata": fresh.get("metadata", {}) if fresh else {},
        "applied": sorted(updates.keys()),
    })


@mcp.custom_route("/api/bucket/{bucket_id}/redehydrate", methods=["POST"])
async def api_bucket_redehydrate(request):
    """对单条记忆重新提炼 — **预览模式, 不写盘**。
    工作台"↻ 重新脱水"按钮调用; 前端拿到预览后, 用户确认/编辑后再调
    /api/bucket/{id}/redehydrate-commit 写回。

    可选 JSON body: {"regenerate_content": bool}
      true → 先用 metadata.raw_source + 主题锚点重写正文, 再用新正文提炼 metadata
             (要求 raw_source 非空, 否则 422)
      false / 缺省 → 仅基于现有 content 提炼 metadata, 不动正文

    返回 {old: {...}, new: {...}, cost, ...} — 老/新双份, 让前端做对比 UI。
    """
    from starlette.responses import JSONResponse
    from utils import estimate_llm_cost
    bucket_id = request.path_params["bucket_id"]
    bucket = await bucket_mgr.get(bucket_id)
    if not bucket:
        return JSONResponse({"error": "not found"}, status_code=404)

    regenerate_content = False
    try:
        body = await request.json()
        if isinstance(body, dict):
            regenerate_content = bool(body.get("regenerate_content", False))
    except Exception:
        pass

    meta = bucket.get("metadata") or {}
    raw_source = meta.get("raw_source", "") or ""
    source_excerpt = meta.get("source_excerpt", "") or ""
    old_content = bucket.get("content", "")
    old_meta_snapshot = {
        "name": meta.get("name", ""),
        "summary": meta.get("summary", ""),
        "domain": meta.get("domain", []),
        "valence": meta.get("valence"),
        "arousal": meta.get("arousal"),
        "tags": [t for t in (meta.get("tags") or []) if not str(t).startswith("__")],
    }

    total_in_tok = 0
    total_out_tok = 0
    model_used = dehydrator.model
    new_content = None  # None = 用户没勾"重写正文"

    # ---- Step 1 (可选): 重写正文 — 主题锚点版 ----
    if regenerate_content:
        if not raw_source.strip():
            return JSONResponse({
                "error": "此条无原文(metadata.raw_source 为空), 无法重新提炼正文",
            }, status_code=422)
        try:
            regen = await dehydrator.regenerate_content_from_source(
                raw_source,
                current_content=old_content,
                source_excerpt=source_excerpt,
                theme_name=old_meta_snapshot["name"],
                theme_summary=old_meta_snapshot["summary"],
                theme_tags=old_meta_snapshot["tags"],
            )
        except RuntimeError as e:
            return JSONResponse({"error": str(e)}, status_code=500)
        new_content = regen.get("content", "")
        total_in_tok += regen.get("_prompt_tokens", 0)
        total_out_tok += regen.get("_completion_tokens", 0)
        model_used = regen.get("_model_used", model_used)
        content_for_meta = new_content
    else:
        content_for_meta = old_content

    if not content_for_meta.strip():
        return JSONResponse({"error": "正文为空,无可提炼内容"}, status_code=400)

    # ---- Step 2: metadata 提炼 ----
    try:
        new_meta = await dehydrator.redehydrate(content_for_meta)
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    raw_output = new_meta.pop("_raw_output", "")
    parse_ok = new_meta.pop("_parse_ok", True)
    total_in_tok += new_meta.pop("_prompt_tokens", 0)
    total_out_tok += new_meta.pop("_completion_tokens", 0)
    model_used = new_meta.pop("_model_used", model_used)
    cost = estimate_llm_cost(model_used, total_in_tok, total_out_tok)

    has_meaningful = bool(new_meta.get("name") or new_meta.get("summary") or new_meta.get("tags"))
    if not parse_ok or not has_meaningful:
        return JSONResponse({
            "error": "LLM 输出无法解析或为空" if not parse_ok else "LLM 没产出有效字段",
            "raw_output": raw_output[:1500],
            "parse_ok": parse_ok,
            "new_meta": new_meta,
        }, status_code=422)

    # ---- Step 3: 返回预览 — 不写盘! ----
    return JSONResponse({
        "ok": True,
        "id": bucket_id,
        "regenerated_content": new_content is not None,
        "old": {
            "content": old_content,
            **old_meta_snapshot,
        },
        "new": {
            "content": new_content if new_content is not None else old_content,
            "name": new_meta.get("name", ""),
            "summary": new_meta.get("summary", ""),
            "domain": new_meta.get("domain", []),
            "valence": new_meta.get("valence"),
            "arousal": new_meta.get("arousal"),
            "tags": new_meta.get("tags", []),
        },
        "raw_output": raw_output[:500],
        "cost": cost,
    })


@mcp.custom_route("/api/bucket/{bucket_id}/redehydrate-commit", methods=["POST"])
async def api_bucket_redehydrate_commit(request):
    """把用户在预览界面确认/编辑后的最终值写回 bucket。
    body 字段(全部可选, 缺哪个就不动哪个):
      {
        "content": str,          # 提供则写正文 + 刷 embedding
        "name": str,
        "summary": str,
        "domain": list,
        "tags": list,            # 注意: 隐藏 status tag(__开头)由后端补回
        "valence": float,
        "arousal": float,
      }
    """
    from starlette.responses import JSONResponse
    bucket_id = request.path_params["bucket_id"]
    bucket = await bucket_mgr.get(bucket_id)
    if not bucket:
        return JSONResponse({"error": "not found"}, status_code=404)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "body must be JSON object"}, status_code=400)

    allowed_keys = {"content", "name", "summary", "domain", "tags", "valence", "arousal"}
    update_kwargs = {}
    for k in allowed_keys:
        if k not in body:
            continue
        v = body[k]
        if v is None:
            continue
        if isinstance(v, str) and not v.strip() and k != "content":
            continue  # 空字符串(content 除外)忽略
        update_kwargs[k] = v

    if not update_kwargs:
        return JSONResponse({"error": "no fields to update"}, status_code=400)

    # 隐藏状态 tag 保留(__import_refined / __import_flagged 等)
    if "tags" in update_kwargs:
        existing_tags = (bucket.get("metadata") or {}).get("tags") or []
        hidden = [t for t in existing_tags if str(t).startswith("__")]
        clean_new = [t for t in update_kwargs["tags"] if not str(t).startswith("__")]
        update_kwargs["tags"] = clean_new + hidden

    new_content = update_kwargs.get("content")
    success = await bucket_mgr.update(bucket_id, **update_kwargs)
    if not success:
        return JSONResponse({"error": "写回失败"}, status_code=500)
    _invalidate_buckets_cache()

    # 正文变了 → 刷 embedding
    if new_content is not None and embedding_engine and embedding_engine.enabled:
        try:
            await embedding_engine.generate_and_store(bucket_id, new_content)
        except Exception:
            pass

    fresh = await bucket_mgr.get(bucket_id)
    return JSONResponse({
        "ok": True,
        "id": bucket_id,
        "applied": sorted(update_kwargs.keys()),
        "metadata": fresh.get("metadata", {}) if fresh else {},
        "content": fresh.get("content", "") if fresh else "",
    })


def _label_mood(v: float, a: float) -> str:
    """把 (valence, arousal) 坐标映射到口语标签, 帮 LLM 理解用户选了什么心情"""
    hi_a = a >= 0.6
    lo_a = a <= 0.4
    pos_v = v >= 0.6
    neg_v = v <= 0.4
    if pos_v and hi_a: return "兴奋/欣快"
    if pos_v and lo_a: return "平和/满足"
    if neg_v and hi_a: return "焦虑/愤怒"
    if neg_v and lo_a: return "低落/沮丧"
    if hi_a:           return "激动"
    if lo_a:           return "平静"
    if pos_v:          return "微微正向"
    if neg_v:          return "微微负向"
    return "中性"


@mcp.custom_route("/api/mood-evoke", methods=["POST"])
async def api_mood_evoke(request):
    """情感唤起: 用户给一个 (valence, arousal) 坐标 → 找最近的 top_n 条记忆 →
    让 LLM 按这个心情口吻把它们串成 1-2 段叙事。返回叙事 + 引用源。"""
    from starlette.responses import JSONResponse
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "请求体不是 JSON"}, status_code=400)
    try:
        v = float(body.get("valence", 0.5))
        a = float(body.get("arousal", 0.3))
    except (TypeError, ValueError):
        return JSONResponse({"error": "valence/arousal 须为数字"}, status_code=400)
    if not (0.0 <= v <= 1.0 and 0.0 <= a <= 1.0):
        return JSONResponse({"error": "valence/arousal 须在 0~1 范围内"}, status_code=400)
    top_n = max(1, min(int(body.get("top_n", 5)), 10))
    # radius: 距离上限(欧氏 + 象限加权后), 默认 0.35 ≈ 正常档
    try:
        radius = float(body.get("radius", 0.35))
    except (TypeError, ValueError):
        radius = 0.35
    radius = max(0.05, min(radius, 1.5))

    if not dehydrator.api_available or dehydrator.client is None:
        return JSONResponse({"error": "LLM 未配置, 无法生成叙事"}, status_code=503)

    all_buckets = await bucket_mgr.list_all(include_archive=False)
    if not all_buckets:
        return JSONResponse({"error": "记忆库为空"}, status_code=404)

    # 象限以 (0.5, 0.5) 为中心切. 0.5 附近视为中性, 不算同/对象限
    import math
    def _qsign(x):
        if x > 0.55: return 1
        if x < 0.45: return -1
        return 0
    user_qv, user_qa = _qsign(v), _qsign(a)

    def _weighted_dist(meta):
        bv = float(meta.get("valence", 0.5))
        ba = float(meta.get("arousal", 0.3))
        d = math.sqrt((bv - v) ** 2 + (ba - a) ** 2)
        # 象限加权:同象限 *0.7, 对角 *1.5, 邻近(只一个轴翻) *1.0
        if user_qv == 0 or user_qa == 0:
            mult = 1.0  # 用户选了中性轴上, 不参与象限调整
        else:
            bqv, bqa = _qsign(bv), _qsign(ba)
            if bqv == user_qv and bqa == user_qa:
                mult = 0.7
            elif bqv == -user_qv and bqa == -user_qa:
                mult = 1.5
            else:
                mult = 1.0
        return d * mult

    scored = []
    for b in all_buckets:
        meta = b.get("metadata", {})
        scored.append((_weighted_dist(meta), b))
    scored.sort(key=lambda x: x[0])

    # radius 过滤 → top_n 截取. 完全没人入选时报告而不是凑数
    in_radius = [(d, b) for d, b in scored if d <= radius]
    relaxed = False
    if len(in_radius) == 0:
        # 一条都不达标时退一步:取最近 2 条让 LLM 至少能写点东西, 但前端会显示"已放宽"
        in_radius = scored[:2]
        relaxed = True
    picks_with_d = in_radius[:top_n]

    lines = []
    sources = []
    for d, b in picks_with_d:
        meta = b.get("metadata", {})
        name = meta.get("name", b["id"])
        body_txt = strip_wikilinks(b.get("content", ""))[:600]
        lines.append(f"《{name}》\n{body_txt}")
        sources.append({
            "id": b["id"],
            "name": name,
            "summary": (meta.get("summary", "") or "")[:200] or body_txt[:120],
            "valence": float(meta.get("valence", 0.5)),
            "arousal": float(meta.get("arousal", 0.3)),
            "event_time": meta.get("event_time", ""),
            "distance": round(d, 3),
        })

    mood_label = _label_mood(v, a)
    sys_prompt = (
        "你是一位记忆叙事者。用户选了一个情感坐标(valence 效价 / arousal 唤醒度), "
        "请按这个心情的口吻把以下几条记忆串成 1-2 段、约 80-200 字的短叙事。"
        "可以是回忆、感慨、或心情侧写, 不要罗列, 不要标号, 不要解释规则, "
        "用第二人称'你'或第一人称'我'都可, 保留原记忆中具体的人/事/词。"
    )
    user_msg = (
        f"心情坐标: valence={v:.2f}, arousal={a:.2f} ({mood_label})\n\n"
        f"素材记忆 ({len(picks_with_d)} 条):\n\n" + "\n\n---\n\n".join(lines)
    )
    try:
        response = await dehydrator.client.chat.completions.create(
            model=dehydrator.model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg[:8000]},
            ],
            max_tokens=600,
            temperature=0.85,
        )
        narrative = (response.choices[0].message.content or "").strip() if response.choices else ""
    except Exception as e:
        return JSONResponse({"error": f"LLM 调用失败: {e}"}, status_code=500)

    return JSONResponse({
        "ok": True,
        "valence": v,
        "arousal": a,
        "radius": radius,
        "mood_label": mood_label,
        "narrative": narrative,
        "sources": sources,
        "relaxed": relaxed,    # True = radius 内一条都没,已退到最近 2 条
    })


def _build_merge_meta(a_meta: dict, b_meta: dict) -> dict:
    """把 A、B 元数据合到 B 的字段集合;tags/domain 并集,importance max,情感平均"""
    a_tags = a_meta.get("tags", []) or []
    b_tags = b_meta.get("tags", []) or []
    a_domain = a_meta.get("domain", []) or []
    b_domain = b_meta.get("domain", []) or []
    a_imp = a_meta.get("importance", 5)
    b_imp = b_meta.get("importance", 5)
    a_v = float(a_meta.get("valence", 0.5))
    a_a = float(a_meta.get("arousal", 0.3))
    b_v = float(b_meta.get("valence", 0.5))
    b_a = float(b_meta.get("arousal", 0.3))
    return {
        "tags": list(dict.fromkeys(b_tags + a_tags)),  # 保留 B 顺序,A 的新 tag 追加
        "domain": list(dict.fromkeys(b_domain + a_domain)),
        "importance": max(a_imp, b_imp),
        "valence": round((a_v + b_v) / 2, 2),
        "arousal": round((a_a + b_a) / 2, 2),
    }


def _check_merge_preconditions(a, b):
    """返回 (ok: bool, error_response_or_none)。校验两个 bucket 能不能合并"""
    from starlette.responses import JSONResponse
    if not a or not b:
        return False, JSONResponse({"error": "bucket not found"}, status_code=404)
    if a["id"] == b["id"]:
        return False, JSONResponse({"error": "不能合并到自己"}, status_code=400)
    b_meta = b.get("metadata", {})
    if b_meta.get("protected") or b_meta.get("pinned"):
        return False, JSONResponse({
            "error": "目标桶已 protected/钉决,拒绝合并(防止改写核心条目)",
            "hint": "如果确实想合并,先去取消目标桶的保护标记",
        }, status_code=409)
    return True, None


@mcp.custom_route("/api/bucket/{bucket_id}/merge-preview", methods=["POST"])
async def api_bucket_merge_preview(request):
    """预算合并结果(不写盘)。query: ?into=<target_b_id>。
    A→B 方向:把 A 合并到 B,删 A 保 B(保留 B 的 last_active/event_time/summary 等积累状态)。
    返回新 content + 元数据合并后的字段,前端展示给用户预览,确认后再调 merge-commit。"""
    from starlette.responses import JSONResponse
    a_id = request.path_params["bucket_id"]
    b_id = request.query_params.get("into", "")
    if not b_id:
        return JSONResponse({"error": "missing ?into=<target_b_id>"}, status_code=400)
    a = await bucket_mgr.get(a_id)
    b = await bucket_mgr.get(b_id)
    ok, err = _check_merge_preconditions(a, b)
    if not ok:
        return err
    a_content = a.get("content", "") or ""
    b_content = b.get("content", "") or ""
    if not a_content.strip() and not b_content.strip():
        return JSONResponse({"error": "两条桶正文都为空,无可合并"}, status_code=400)
    # LLM 跑合并(B 是旧,A 是新,merge 规则里以新为准但去重)
    try:
        merged_content = await dehydrator.merge(b_content, a_content)
    except Exception as e:
        return JSONResponse({"error": f"LLM 合并失败: {e}"}, status_code=500)
    meta_merged = _build_merge_meta(a.get("metadata", {}), b.get("metadata", {}))
    # 估算这次合并的开销
    from utils import estimate_llm_cost
    last_usage = getattr(dehydrator.__class__, "_last_merge_usage", None)
    cost = estimate_llm_cost(
        last_usage["model"] if last_usage else dehydrator.model,
        last_usage["prompt_tokens"] if last_usage else 0,
        last_usage["completion_tokens"] if last_usage else 0,
    ) if last_usage else None
    return JSONResponse({
        "ok": True,
        "preview": True,
        "a": {"id": a_id, "name": a.get("metadata", {}).get("name", a_id)},
        "b": {"id": b_id, "name": b.get("metadata", {}).get("name", b_id)},
        "merged_content": merged_content,
        "a_content": a_content,
        "b_content": b_content,
        **meta_merged,
        "b_summary": b.get("metadata", {}).get("summary", ""),
        "b_event_time": b.get("metadata", {}).get("event_time", ""),
        "b_created": b.get("metadata", {}).get("created", ""),
        "cost": cost,
    })


@mcp.custom_route("/api/bucket/{bucket_id}/merge-commit", methods=["POST"])
async def api_bucket_merge_commit(request):
    """提交合并:把 A 的内容真正合到 B(用前端预览的 content)+ 删除 A。
    body: {"merged_content": "..."}  这里直接拿前端确认过的预览内容,避免再跑一次 LLM
    query: ?into=<target_b_id>"""
    from starlette.responses import JSONResponse
    a_id = request.path_params["bucket_id"]
    b_id = request.query_params.get("into", "")
    if not b_id:
        return JSONResponse({"error": "missing ?into=<target_b_id>"}, status_code=400)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    merged_content = body.get("merged_content", "")
    if not merged_content or not merged_content.strip():
        return JSONResponse({"error": "merged_content 必填"}, status_code=400)
    a = await bucket_mgr.get(a_id)
    b = await bucket_mgr.get(b_id)
    ok, err = _check_merge_preconditions(a, b)
    if not ok:
        return err
    meta_merged = _build_merge_meta(a.get("metadata", {}), b.get("metadata", {}))
    # 1) 更新 B(content + 元数据并集)
    update_ok = await bucket_mgr.update(
        b_id,
        content=merged_content,
        tags=meta_merged["tags"],
        domain=meta_merged["domain"],
        importance=meta_merged["importance"],
        valence=meta_merged["valence"],
        arousal=meta_merged["arousal"],
    )
    if not update_ok:
        return JSONResponse({"error": "更新 B 失败"}, status_code=500)
    # 2) B content 变了,刷 embedding
    if embedding_engine and embedding_engine.enabled:
        try:
            await embedding_engine.generate_and_store(b_id, merged_content)
        except Exception as e:
            logger.warning(f"merge: B embedding refresh failed: {e}")
    # 3) 删 A
    delete_ok = await bucket_mgr.delete(a_id)
    if not delete_ok:
        return JSONResponse({
            "error": "B 已合并但 A 删除失败,需要手动清理",
            "b_id": b_id,
        }, status_code=500)
    # 4) 顺手清掉 A 的 embedding
    if embedding_engine:
        try:
            embedding_engine.delete_embedding(a_id)
        except Exception:
            pass
    _invalidate_buckets_cache()
    fresh = await bucket_mgr.get(b_id)
    return JSONResponse({
        "ok": True,
        "merged_into": b_id,
        "deleted": a_id,
        "metadata": fresh.get("metadata", {}) if fresh else {},
    })


@mcp.custom_route("/api/bucket/{bucket_id}/archive", methods=["POST"])
async def api_bucket_archive(request):
    """手动归档一条桶,移到 archive/,AI 不再调用。"""
    from starlette.responses import JSONResponse
    bucket_id = request.path_params["bucket_id"]
    bucket = await bucket_mgr.get(bucket_id)
    if not bucket:
        return JSONResponse({"error": "not found"}, status_code=404)
    success = await bucket_mgr.archive(bucket_id)
    if not success:
        return JSONResponse({"error": "archive failed"}, status_code=500)
    _invalidate_buckets_cache()
    return JSONResponse({"ok": True, "id": bucket_id, "archived": True})


@mcp.custom_route("/api/bucket/{bucket_id}/delete", methods=["POST"])
async def api_bucket_delete(request):
    """软删除:移到回收站(可在 /v2/trash/ 恢复)。embedding 不动,搜索时
    被 trash type 自然过滤。要真删调 /purge。"""
    from starlette.responses import JSONResponse
    bucket_id = request.path_params["bucket_id"]
    bucket = await bucket_mgr.get(bucket_id)
    if not bucket:
        return JSONResponse({"error": "not found"}, status_code=404)
    success = await bucket_mgr.delete(bucket_id)
    if not success:
        return JSONResponse({"error": "delete failed"}, status_code=500)
    _invalidate_buckets_cache()
    return JSONResponse({"ok": True, "id": bucket_id, "soft_deleted": True})


@mcp.custom_route("/api/bucket/{bucket_id}/restore", methods=["POST"])
async def api_bucket_restore(request):
    """从回收站恢复:把桶从 trash 移回原 type 目录。"""
    from starlette.responses import JSONResponse
    bucket_id = request.path_params["bucket_id"]
    bucket = await bucket_mgr.get(bucket_id)
    if not bucket:
        return JSONResponse({"error": "not found"}, status_code=404)
    success = await bucket_mgr.restore(bucket_id)
    if not success:
        return JSONResponse({"error": "restore failed (可能不在回收站)"}, status_code=400)
    _invalidate_buckets_cache()
    return JSONResponse({"ok": True, "id": bucket_id, "restored": True})


@mcp.custom_route("/api/bucket/{bucket_id}/purge", methods=["POST"])
async def api_bucket_purge(request):
    """永久删除:物理删文件 + 清 embedding。不可撤销。
    通常只从回收站界面调用(经过二次确认)。"""
    from starlette.responses import JSONResponse
    bucket_id = request.path_params["bucket_id"]
    bucket = await bucket_mgr.get(bucket_id)
    if not bucket:
        return JSONResponse({"error": "not found"}, status_code=404)
    success = await bucket_mgr.purge(bucket_id)
    if not success:
        return JSONResponse({"error": "purge failed"}, status_code=500)
    if embedding_engine:
        try:
            embedding_engine.delete_embedding(bucket_id)
        except Exception:
            pass
    _invalidate_buckets_cache()
    return JSONResponse({"ok": True, "id": bucket_id, "purged": True})


@mcp.custom_route("/api/trash/empty", methods=["POST"])
async def api_trash_empty(request):
    """清空回收站:一次请求物理删除所有 trash 桶 + 清对应 embedding。
    避免前端逐条 purge 几百次往返导致的"每次只删一点"。不可撤销。"""
    from starlette.responses import JSONResponse
    try:
        # 先收集 id 用于清 embedding(empty_trash 删文件后就拿不到了)
        ids = []
        try:
            trashed = await bucket_mgr.list_trash()
            ids = [b["id"] for b in trashed]
        except Exception:
            pass
        n = await bucket_mgr.empty_trash()
        if embedding_engine:
            for bid in ids:
                try:
                    embedding_engine.delete_embedding(bid)
                except Exception:
                    pass
        _invalidate_buckets_cache()
        return JSONResponse({"ok": True, "purged": n})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/trash", methods=["GET"])
async def api_trash_list(request):
    """列回收站所有桶。返回 mock 兼容形态(content_preview / summary 等)。"""
    from starlette.responses import JSONResponse
    try:
        trashed = await bucket_mgr.list_trash()
        result = []
        for b in trashed:
            meta = b.get("metadata", {})
            result.append({
                "id": b["id"],
                "name": meta.get("name", b["id"]),
                "type": meta.get("type", "trashed"),
                "original_type": meta.get("original_type", "dynamic"),
                "domain": meta.get("domain", []),
                "tags": meta.get("tags", []),
                "valence": meta.get("valence", 0.5),
                "arousal": meta.get("arousal", 0.3),
                "importance": meta.get("importance", 5),
                "protected": is_protected(meta),
                "highlight": is_highlighted(meta),
                "internalized": is_internalized(meta),
                "event_time": meta.get("event_time", ""),
                "created": meta.get("created", ""),
                "trashed_at": meta.get("trashed_at", ""),
                "summary": meta.get("summary", ""),
                "content_preview": strip_wikilinks(b.get("content", ""))[:200],
                "score": decay_engine.calculate_score(meta),
                "resolved": meta.get("resolved", False),
                "noise": bool(meta.get("resolved", False) and meta.get("importance", 5) == 1),
            })
        return JSONResponse({"trash": result, "count": len(result)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/_diagnose/{bucket_id}", methods=["GET"])
async def api_diagnose_bucket(request):
    """单桶诊断 — 只读, 给定 id 返回它在哪个目录 / 能否加载 / metadata / content 长度 + 诊断标记.
    用途: 排查"id 能搜到但搜不到内容 / tag 搜不到"这类边缘 case. 浏览器直接打开看 JSON 就行.
    """
    from starlette.responses import JSONResponse
    bucket_id = request.path_params["bucket_id"].strip()
    if not bucket_id:
        return JSONResponse({"error": "bucket_id 必填"}, status_code=400)

    dirs = {
        "permanent": bucket_mgr.permanent_dir,
        "dynamic":   bucket_mgr.dynamic_dir,
        "feel":      bucket_mgr.feel_dir,
        "archive":   bucket_mgr.archive_dir,
        "trash":     bucket_mgr.trash_dir,
    }

    hits = []
    for tag, d in dirs.items():
        if not os.path.exists(d):
            continue
        for root, _, files in os.walk(d):
            for fname in files:
                if not fname.endswith(".md"):
                    continue
                name_part = fname[:-3]
                if name_part == bucket_id or name_part.endswith(f"_{bucket_id}"):
                    hits.append({"location": tag, "path": os.path.join(root, fname)})

    if not hits:
        return JSONResponse({
            "bucket_id": bucket_id,
            "found": False,
            "message": "5 个目录里都没找到, 桶可能已被物理删除 (purge) 或 id 不对",
        })

    out = {
        "bucket_id": bucket_id,
        "found": True,
        "hit_count": len(hits),
        "hits": [],
    }
    for h in hits:
        entry = {"location": h["location"], "path": h["path"]}
        try:
            size = os.path.getsize(h["path"])
            entry["file_size_bytes"] = size
        except Exception:
            entry["file_size_bytes"] = None

        try:
            import frontmatter
            post = frontmatter.load(h["path"])
            meta = dict(post.metadata)
            content = post.content or ""
            entry["loaded"] = True
            entry["metadata"] = {
                k: (meta.get(k) if k in meta else None)
                for k in [
                    "id", "name", "type", "domain", "tags",
                    "importance", "valence", "arousal",
                    "resolved", "protected", "highlight", "pinned",
                    "internalized", "digested",
                    "created", "last_active", "event_time",
                    "trashed_at", "archived_at", "created_by",
                ]
            }
            entry["content_length"] = len(content)
            entry["content_preview"] = content.strip()[:300]
            entry["content_empty"] = not content.strip()

            # 诊断标记
            diag = []
            if h["location"] == "trash":
                diag.append("桶在 trash, list_all 默认不含 trash → 关键字搜索看不到 (设计如此)")
            if h["location"] == "archive":
                diag.append("桶在 archive, breath 搜索默认 include_archive=False → 关键字搜索看不到")
            name_v = meta.get("name", "") or ""
            tags_v = meta.get("tags", []) or []
            if not name_v.strip() and not tags_v and not content.strip():
                diag.append("name + tags + content 全空 → fuzz 找不到任何匹配信号")
            elif not name_v.strip() and not tags_v:
                diag.append("name 和 tags 都空, 只能靠 content 模糊匹配")
            if meta.get("resolved") and meta.get("importance", 5) == 1:
                diag.append("resolved+importance=1 → 标了 noise (软删除), 搜索显式排除")
            if meta.get("internalized") or meta.get("digested"):
                diag.append("internalized=True → 搜索显式排除 (设计上已内化的桶不浮现)")
            entry["diagnostics"] = diag
        except Exception as e:
            entry["loaded"] = False
            entry["load_error"] = str(e)
            entry["diagnostics"] = ["frontmatter 解析失败 → 这就是 'id 能见但内容空' 的真凶, 文件可能 YAML 坏掉"]
            try:
                with open(h["path"], "r", encoding="utf-8") as f:
                    first_lines = [next(f, "").rstrip() for _ in range(8)]
                entry["raw_first_lines"] = first_lines
            except Exception as e2:
                entry["raw_read_error"] = str(e2)

        out["hits"].append(entry)

    return JSONResponse(out)


@mcp.custom_route("/api/bucket/{bucket_id}/unarchive", methods=["POST"])
async def api_bucket_unarchive(request):
    """取消归档,从 archive/ 移回 dynamic/,AI 重新可见。"""
    from starlette.responses import JSONResponse
    bucket_id = request.path_params["bucket_id"]
    bucket = await bucket_mgr.get(bucket_id)
    if not bucket:
        return JSONResponse({"error": "not found"}, status_code=404)
    success = await bucket_mgr.unarchive(bucket_id)
    if not success:
        return JSONResponse({"error": "unarchive failed (not in archive?)"}, status_code=400)
    _invalidate_buckets_cache()
    return JSONResponse({"ok": True, "id": bucket_id, "archived": False})


@mcp.custom_route("/api/embeddings/diagnose", methods=["GET"])
async def api_embeddings_diagnose(request):
    """诊断 embedding 状态:配置 + 已生成数 / 总桶数。"""
    from starlette.responses import JSONResponse
    if not embedding_engine:
        return JSONResponse({"error": "embedding_engine 未初始化"}, status_code=500)
    all_buckets = await bucket_mgr.list_all(include_archive=False)
    has_emb = 0
    for b in all_buckets:
        if await embedding_engine.get_embedding(b["id"]) is not None:
            has_emb += 1
    return JSONResponse({
        "enabled": embedding_engine.enabled,
        "model": embedding_engine.model,
        "base_url": embedding_engine.base_url,
        "api_key_masked": (embedding_engine.api_key[:5] + "..." + embedding_engine.api_key[-4:]) if embedding_engine.api_key else "",
        "total_buckets": len(all_buckets),
        "with_embedding": has_emb,
        "missing": len(all_buckets) - has_emb,
    })


@mcp.custom_route("/api/embeddings/backfill", methods=["POST"])
async def api_embeddings_backfill(request):
    """一键给所有缺 embedding 的桶补生成。轻量保护:env OMBRE_ADMIN_TOKEN 鉴权。
    无 token 时拒绝(避免被滥用消耗 API 配额)。"""
    from starlette.responses import JSONResponse
    expected = os.environ.get("OMBRE_ADMIN_TOKEN", "")
    if not expected:
        return JSONResponse({"error": "OMBRE_ADMIN_TOKEN 未配置,拒绝运行"}, status_code=503)
    token = request.query_params.get("token") or request.headers.get("X-Admin-Token", "")
    if token != expected:
        return JSONResponse({"error": "invalid token"}, status_code=401)
    if not embedding_engine or not embedding_engine.enabled:
        return JSONResponse({"error": "embedding 未启用,检查 OMBRE_EMBED_API_KEY"}, status_code=400)

    all_buckets = await bucket_mgr.list_all(include_archive=True)
    missing = []
    for b in all_buckets:
        if await embedding_engine.get_embedding(b["id"]) is None:
            missing.append(b)

    success = 0
    failed = 0
    skipped = 0
    errors = []
    for idx, b in enumerate(missing):
        content = (b.get("content") or "").strip()
        if not content:
            skipped += 1
            continue
        try:
            ok = await embedding_engine.generate_and_store(b["id"], content)
            if ok: success += 1
            else: failed += 1
        except Exception as e:
            failed += 1
            errors.append(f"{b['id'][:12]}: {str(e)[:120]}")
            if len(errors) > 5:
                break
        if (idx + 1) % 20 == 0:
            import asyncio as _aio
            await _aio.sleep(2)
    return JSONResponse({
        "total_missing": len(missing),
        "success": success,
        "failed": failed,
        "skipped_empty": skipped,
        "errors": errors[:5],
    })


@mcp.custom_route("/api/bucket/{bucket_id}/similar", methods=["GET"])
async def api_bucket_similar(request):
    """返回该桶在全库内最相似的 top N(by embedding cosine)。导入工作台用。"""
    from starlette.responses import JSONResponse
    bucket_id = request.path_params["bucket_id"]
    try:
        n = int(request.query_params.get("n", "5"))
    except Exception:
        n = 5
    threshold = float(request.query_params.get("threshold", "0.3"))

    if not embedding_engine or not embedding_engine.enabled:
        return JSONResponse({"similar": [], "note": "embedding 未启用"})

    target_emb = await embedding_engine.get_embedding(bucket_id)
    if target_emb is None:
        return JSONResponse({"similar": [], "note": "目标桶未生成 embedding"})

    all_buckets = await bucket_mgr.list_all(include_archive=False)
    scored = []
    for b in all_buckets:
        bid = b["id"]
        if bid == bucket_id:
            continue
        other_emb = await embedding_engine.get_embedding(bid)
        if other_emb is None:
            continue
        sim = embedding_engine._cosine_similarity(target_emb, other_emb)
        if sim < threshold:
            continue
        meta = b.get("metadata", {})
        scored.append({
            "id": bid,
            "name": meta.get("name", bid),
            "score": round(float(sim), 3),
            "date": (meta.get("event_time") or meta.get("created") or "")[:10],
            "summary": meta.get("summary") or (b.get("content") or "")[:120],
            "type": meta.get("type", "dynamic"),
            "tags": meta.get("tags", []) or [],  # 前端 statusOf 用,判断 pending/refined/flagged
        })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return JSONResponse({"similar": scored[:max(1, n)]})


@mcp.custom_route("/api/bucket/create", methods=["POST"])
async def api_bucket_create(request):
    """用户手动新建一条桶。**不走合并**,所见即所存。带 created_by='user' 标记。"""
    from starlette.responses import JSONResponse
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    content = (body.get("content") or "").strip()
    if not content:
        return JSONResponse({"error": "content is required"}, status_code=400)

    # 接受字符串(逗号分隔)或数组
    tags = body.get("tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    domain = body.get("domain", ["未分类"])
    if isinstance(domain, str):
        domain = [d.strip() for d in domain.split(",") if d.strip()] or ["未分类"]

    importance = int(body.get("importance", 5))
    valence = float(body.get("valence", 0.5))
    arousal = float(body.get("arousal", 0.3))
    name = body.get("name") or None
    event_time = body.get("event_time") or None
    protected = bool(body.get("protected", False))
    highlight = bool(body.get("highlight", False))
    internalized = bool(body.get("internalized", False))
    summary = body.get("summary") or None  # 用户在 WriteDrawer 填的"一句话摘要", 漏接 → 写一条记忆摘要永远存不下来
    # type 字段 — 用户在 WriteDrawer 切 feel 时前端传 type='feel'
    # 默认 'dynamic', 合法值: dynamic / feel / permanent
    bucket_type = body.get("type", "dynamic")
    if bucket_type not in ("dynamic", "feel", "permanent"):
        bucket_type = "dynamic"

    try:
        bucket_id = await bucket_mgr.create(
            content=content,
            tags=tags,
            importance=importance,
            domain=domain,
            valence=valence,
            arousal=arousal,
            name=name,
            protected=protected,
            highlight=highlight,
            event_time=event_time,
            bucket_type=bucket_type,  # feel 切换时这里写入 metadata.type='feel'
            created_by="user",  # dashboard 手动新建标记,跟 AI 写入区分
            summary=summary,
        )
    except Exception as e:
        return JSONResponse({"error": f"create failed: {e}"}, status_code=500)

    # internalized 在创建时不能直接通过 create() 设(create 没这个参数),
    # 用一次 update 顺手设上。失败也不影响桶本身的创建。
    if internalized:
        try:
            await bucket_mgr.update(bucket_id, internalized=True)
        except Exception:
            pass

    if embedding_engine and embedding_engine.enabled:
        try:
            await embedding_engine.generate_and_store(bucket_id, content)
        except Exception:
            pass

    _invalidate_buckets_cache()
    fresh = await bucket_mgr.get(bucket_id)
    return JSONResponse({
        "ok": True,
        "id": bucket_id,
        "metadata": fresh.get("metadata", {}) if fresh else {},
    })


@mcp.custom_route("/api/search", methods=["GET"])
async def api_search(request):
    """
    Search buckets by query — keyword channel(主) + 可选 vector channel(语义猜测).

    Query params:
      q: 搜索词(必填)
      limit: 最多返回多少条 keyword 命中,默认 20
      include_vector: 'true' 时附带向量(语义)结果,默认 'false' (避免污染纯关键词搜索体验)
      exclude_pinned: 'true' 时排除钉选(highlight/protected)记忆,默认 'false'。
                      auto-inject 用 —— 钉选开窗已浮现, 注入再命中=重复噪声; 手动搜索别传(仍要找得到)

    Response shape:
      {
        "query": str,
        "keyword_hits": [  # 走 fuzz partial_ratio,真的"含 query"的桶
          {
            id, name, score, domain, valence, arousal,
            content_preview,
            matched_in: ["title","summary","tag","domain","content"],  # 命中字段,前端高亮 / 标签
            field_scores: {title, summary, domain, tag, content},  # 0-100 原分,debug 用
            summary,  # 摘要原文(给前端命中高亮)
            tags,
          },
          ...
        ],
        "vector_hits": [  # include_vector=true 时才有,语义相近但 query 不在文本里
          { id, name, similarity, content_preview, summary, tags, domain }
        ]
      }
    """
    from starlette.responses import JSONResponse
    query = request.query_params.get("q", "")
    if not query:
        return JSONResponse({"error": "missing q parameter"}, status_code=400)
    try:
        limit = int(request.query_params.get("limit", "20"))
    except ValueError:
        limit = 20
    include_vector = request.query_params.get("include_vector", "false").lower() == "true"
    # 默认排除噪声(resolved+importance=1, 用户软删除态);
    # 调试/查找意图时可 include_noise=true opt-in 拉回
    include_noise = request.query_params.get("include_noise", "false").lower() == "true"
    # 默认排除 feel(私密沉淀, 设计上不参与 AI 检索/自动浮现; 只在 breath domain="feel" 显式入口可取);
    # 确有需要可 include_feel=true opt-in
    include_feel = request.query_params.get("include_feel", "false").lower() == "true"
    # 默认不排钉选(手动搜索仍要能找到钉选记忆); auto-inject 传 exclude_pinned=true 才排。
    # 理由: 钉选(highlight / protected)开窗时已由 breath-hook 必定浮现进上下文,
    #       日常对话自动注入再命中同一批 = 重复噪声。排掉降噪。(调用方见 ombre-inject.js)
    exclude_pinned = request.query_params.get("exclude_pinned", "false").lower() == "true"
    # simulate=true: 即时模拟(浮现观测页用) —— 不记命中统计、不进最近搜索, 纯 dry-run 看"会检索到什么"
    simulate = request.query_params.get("simulate", "false").lower() == "true"

    def _is_noise(meta):
        return bool(meta.get("resolved", False) and meta.get("importance", 5) == 1)

    def _is_feel(meta):
        return meta.get("type") == "feel"

    def _is_pinned(meta):
        return is_highlighted(meta) or is_protected(meta)

    try:
        # === 关键词通道 ===
        matches = await bucket_mgr.search(query, limit=limit, record_stats=not simulate)
        if not include_noise:
            matches = [b for b in matches if not _is_noise(b.get("metadata", {}))]
        if not include_feel:
            matches = [b for b in matches if not _is_feel(b.get("metadata", {}))]
        if exclude_pinned:
            matches = [b for b in matches if not _is_pinned(b.get("metadata", {}))]
        keyword_hits = []
        for b in matches:
            meta = b.get("metadata", {})
            keyword_hits.append({
                "id": b["id"],
                "name": meta.get("name", b["id"]),
                "type": meta.get("type", "dynamic"),  # 暴露桶类型(dynamic/feel/permanent), 供调用方按类型过滤
                "score": b.get("score", 0),
                "domain": meta.get("domain", []),
                "tags": meta.get("tags", []),
                "summary": meta.get("summary", ""),
                "valence": meta.get("valence", 0.5),
                "arousal": meta.get("arousal", 0.3),
                "content_preview": strip_wikilinks(b.get("content", ""))[:200],
                "matched_in": b.get("matched_in", []),
                "field_scores": b.get("field_scores", {}),
            })

        # === 向量通道(可选) — 排除已在 keyword_hits 里的桶,避免重复 ===
        # search_similar 返回 list[tuple[bucket_id, similarity]],不是 dict
        vector_hits = []
        if include_vector and embedding_engine is not None:
            try:
                kw_ids = {h["id"] for h in keyword_hits}
                vec_results = await embedding_engine.search_similar(query, top_k=10)
                for bucket_id, similarity in vec_results:
                    if bucket_id in kw_ids:
                        continue  # 已在关键词命中里,不重复
                    vb = await bucket_mgr.get(bucket_id)
                    if not vb:
                        continue
                    vmeta = vb.get("metadata", {})
                    if not include_noise and _is_noise(vmeta):
                        continue
                    if not include_feel and _is_feel(vmeta):
                        continue
                    if exclude_pinned and _is_pinned(vmeta):
                        continue
                    vector_hits.append({
                        "id": bucket_id,
                        "name": vmeta.get("name", bucket_id),
                        "type": vmeta.get("type", "dynamic"),  # 暴露桶类型, 供调用方按类型过滤
                        "similarity": round(similarity, 3),
                        "domain": vmeta.get("domain", []),
                        "tags": vmeta.get("tags", []),
                        "summary": vmeta.get("summary", ""),
                        "content_preview": strip_wikilinks(vb.get("content", ""))[:200],
                    })
            except Exception as ve:
                logger.warning(f"[/api/search] vector channel failed: {ve}")

        return JSONResponse({
            "query": query,
            "keyword_hits": keyword_hits,
            "vector_hits": vector_hits,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/network", methods=["GET"])
async def api_network(request):
    """Get embedding similarity network for visualization."""
    from starlette.responses import JSONResponse
    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
        nodes = []
        edges = []
        embeddings = {}

        for b in all_buckets:
            meta = b.get("metadata", {})
            bid = b["id"]
            nodes.append({
                "id": bid,
                "name": meta.get("name", bid),
                "type": meta.get("type", "dynamic"),
                "domain": meta.get("domain", []),
                "valence": meta.get("valence", 0.5),
                "arousal": meta.get("arousal", 0.3),
                "score": decay_engine.calculate_score(meta),
                "resolved": meta.get("resolved", False),
                "protected": is_protected(meta),
                "highlight": is_highlighted(meta),
                "pinned": is_protected(meta) or is_highlighted(meta),  # 兼容旧前端
                "internalized": is_internalized(meta),
                "digested": is_internalized(meta),  # 兼容旧前端
            })
            if embedding_engine and embedding_engine.enabled:
                emb = await embedding_engine.get_embedding(bid)
                if emb is not None:
                    embeddings[bid] = emb

        # Build edges from embeddings (similarity > 0.5)
        ids = list(embeddings.keys())
        for i, id_a in enumerate(ids):
            for id_b in ids[i+1:]:
                sim = embedding_engine._cosine_similarity(embeddings[id_a], embeddings[id_b])
                if sim > 0.5:
                    edges.append({"source": id_a, "target": id_b, "similarity": round(sim, 3)})

        return JSONResponse({"nodes": nodes, "edges": edges})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/backup", methods=["POST", "GET"])
async def api_backup(request):
    """夜间自动备份: 把 buckets/ 推到私有 git repo. 用 dulwich (纯 Python git),
    不依赖系统 git 命令.
    需要 env 变量:
      OMBRE_BACKUP_REPO  = https://github.com/<user>/<repo>.git
      OMBRE_BACKUP_TOKEN = github fine-grained PAT (Contents R/W)
      OMBRE_BACKUP_USER  = github 用户名
    可选鉴权:
      OMBRE_ADMIN_TOKEN  = 设了的话,调用方必须带 ?token=... 或 X-Admin-Token: ... header.
                           没设则不鉴权(向后兼容老部署)。强烈建议公网部署设上,
                           否则攻击者可以无限触发备份消耗 GitHub PAT 配额。
    UptimeRobot 每天 ping 一次此 endpoint 即可."""
    from starlette.responses import JSONResponse
    import tempfile, shutil
    from datetime import datetime as _dt

    # 可选 admin token 鉴权: 配置了就强制校验, 没配则放行
    expected = os.environ.get("OMBRE_ADMIN_TOKEN", "").strip()
    if expected:
        provided = request.query_params.get("token") or request.headers.get("X-Admin-Token", "")
        if provided != expected:
            return JSONResponse({"error": "invalid or missing admin token"}, status_code=401)

    repo_url = os.environ.get("OMBRE_BACKUP_REPO", "").strip()
    token    = os.environ.get("OMBRE_BACKUP_TOKEN", "").strip()
    user     = os.environ.get("OMBRE_BACKUP_USER", "ombre-bot").strip()  # 默认 ombre-bot → 文档可标"可选"
    if not (repo_url and token and user):
        return JSONResponse({"error": "OMBRE_BACKUP_REPO/TOKEN/USER 至少一个未设"}, status_code=500)

    buckets_dir = config.get("buckets_dir", "./buckets")
    if not os.path.exists(buckets_dir):
        return JSONResponse({"error": f"buckets_dir 不存在: {buckets_dir}"}, status_code=500)

    if "https://" not in repo_url:
        return JSONResponse({"error": "REPO 必须是 https:// URL"}, status_code=500)
    # 带 token 的 URL 用于 dulwich 的 push (它读 URL 内的 user:pass)
    auth_url = repo_url.replace("https://", f"https://x-access-token:{token}@", 1)

    try:
        from dulwich import porcelain
        from dulwich.errors import NotGitRepository
    except ImportError as e:
        return JSONResponse({"error": f"dulwich 未安装: {e}"}, status_code=500)

    tmp = tempfile.mkdtemp(prefix="ombre-backup-")
    try:
        # 1. 尝试浅克隆; 空仓 / 不存在 / 无权限 都尝试 init 一个空 repo 然后 push 上去
        repo = None
        clone_err = None
        try:
            repo = porcelain.clone(
                auth_url, tmp,
                depth=1,
                checkout=True,
                errstream=open(os.devnull, "wb"),
            )
        except Exception as e:
            clone_err = str(e).replace(token, "***") if token in str(e) else str(e)
            # 不管什么错, 试试当 fresh repo 来推
            shutil.rmtree(tmp, ignore_errors=True)
            tmp = tempfile.mkdtemp(prefix="ombre-backup-")
            repo = porcelain.init(tmp)

        # 2. 拷 buckets 全部内容
        backup_subdir = os.path.join(tmp, "buckets")
        if os.path.exists(backup_subdir):
            shutil.rmtree(backup_subdir)
        shutil.copytree(buckets_dir, backup_subdir)

        # 3. runtime_config.json (放外面, 跟 buckets/ 同级)
        runtime_cfg = os.path.join(buckets_dir, "runtime_config.json")
        if os.path.exists(runtime_cfg):
            shutil.copy2(runtime_cfg, os.path.join(tmp, "runtime_config.json"))

        # 4. add 全部 (dulwich 需要相对路径)
        all_files = []
        for root, dirs, files in os.walk(tmp):
            # 跳过 .git
            if ".git" in dirs:
                dirs.remove(".git")
            for f in files:
                rel = os.path.relpath(os.path.join(root, f), tmp)
                all_files.append(rel)
        if not all_files:
            return JSONResponse({"ok": True, "message": "无文件可备份"})
        porcelain.add(tmp, paths=all_files)

        # 5. commit; bucket 数 = 递归数所有 .md (因为 buckets/ 下面是 dynamic/permanent/feel 等子目录)
        ts = _dt.now().strftime("%Y-%m-%d %H:%M")
        bucket_count = 0
        if os.path.exists(backup_subdir):
            for _root, _dirs, _files in os.walk(backup_subdir):
                bucket_count += sum(1 for f in _files if f.endswith(".md"))
        commit_msg = f"auto-backup {ts} ({bucket_count} buckets)".encode("utf-8")
        author = f"{user} <{user}@ombre-backup>".encode("utf-8")
        try:
            commit_sha = porcelain.commit(
                tmp,
                message=commit_msg,
                author=author,
                committer=author,
            )
        except Exception as e:
            err = str(e).replace(token, "***") if token in str(e) else str(e)
            # commit 失败一般是因为没 staged change → 视为"无变化"
            if "nothing to commit" in err.lower() or "no changes" in err.lower():
                return JSONResponse({"ok": True, "message": "无变化, 跳过 commit"})
            return JSONResponse({"error": f"commit 失败: {err}"}, status_code=500)

        # 6. push 到 origin (如果是 fresh init, 先 add remote)
        try:
            # 尝试直接 push (clone 来的仓库已经有 origin)
            porcelain.push(
                tmp,
                remote_location=auth_url,
                refspecs=b"HEAD:refs/heads/main",
                errstream=open(os.devnull, "wb"),
            )
        except Exception as e:
            err = str(e).replace(token, "***") if token in str(e) else str(e)
            return JSONResponse({
                "error": f"push 失败: {err}",
                "clone_status": clone_err if clone_err else "ok",
            }, status_code=500)

        return JSONResponse({
            "ok": True,
            "timestamp": ts,
            "bucket_count": bucket_count,
            "commit_message": commit_msg.decode("utf-8"),
            "commit_sha": commit_sha.decode("ascii") if isinstance(commit_sha, bytes) else str(commit_sha),
            "clone_status": clone_err if clone_err else "ok",
            "engine": "dulwich",
        })
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        if token:
            tb = tb.replace(token, "***")
        return JSONResponse({
            "error": f"unhandled {type(e).__name__}: {e}",
            "traceback_tail": tb[-1000:],
        }, status_code=500)
    finally:
        try:
            shutil.rmtree(tmp)
        except Exception:
            pass


@mcp.custom_route("/api/cleanup-pseudo-tags", methods=["POST", "GET"])
async def api_cleanup_pseudo_tags(request):
    """一次性清洗 bucket.tags 里 v2 bridge 错误注入的伪 tag.

    背景: v2 前端 bridge 在读端往 item.tags 注入派生 tag (亲手写/AI 写入/导入/已内化/
    保护/重要/feel(柔软)) 用于显示, 但 ItemModal 保存时把这些 tag 一并写回后端,
    每存一次就追加一份, 导致 bucket 文件里 tags 数组堆满冗余字符串.
    bridge 已在写端补了剥离逻辑, 但历史污染需要这个端点一次性清理.

    默认 dry-run; ?commit=true 才真改.
    需要 OMBRE_ADMIN_TOKEN (跟其他 admin 端点一致)."""
    from starlette.responses import JSONResponse

    PSEUDO = {'亲手写', 'AI 写入', '导入', '已内化', '保护', '重要', '高亮', 'feel(柔软)'}

    expected = os.environ.get("OMBRE_ADMIN_TOKEN", "").strip()
    if expected:
        provided = request.query_params.get("token") or request.headers.get("X-Admin-Token", "")
        if provided != expected:
            return JSONResponse({"error": "invalid or missing admin token"}, status_code=401)

    commit_mode = request.query_params.get("commit", "").lower() in ("1", "true", "yes")

    all_buckets = await bucket_mgr.list_all(include_archive=True)
    affected = []
    errors = []
    for b in all_buckets:
        bid = b["id"]
        meta = b.get("metadata", {}) or {}
        tags = meta.get("tags") or []
        if not isinstance(tags, list):
            continue
        removed = [t for t in tags if str(t) in PSEUDO]
        if not removed:
            continue
        cleaned = [t for t in tags if str(t) not in PSEUDO]
        record = {
            "id": bid,
            "name": meta.get("name") or bid[:12],
            "removed": removed,
            "before_len": len(tags),
            "after_len": len(cleaned),
        }
        if commit_mode:
            try:
                ok = await bucket_mgr.update(bid, tags=cleaned)
                record["written"] = bool(ok)
            except Exception as e:
                record["error"] = f"{type(e).__name__}: {str(e)[:160]}"
                errors.append(record["error"])
        affected.append(record)

    return JSONResponse({
        "mode": "commit" if commit_mode else "dry_run",
        "scanned": len(all_buckets),
        "affected_count": len(affected),
        "examples": affected[:8],
        "errors": errors[:5] if errors else None,
        "hint": ("dry_run: 加 ?commit=true 真改" if not commit_mode
                 else "已写盘 — 跑一次 GitHub Action 备份, 验证完就可以把这个端点删了"),
    })


@mcp.custom_route("/api/breath-debug", methods=["GET"])
async def api_breath_debug(request):
    """Debug endpoint: simulate breath scoring and return per-bucket breakdown."""
    from starlette.responses import JSONResponse
    query = request.query_params.get("q", "")
    q_valence = request.query_params.get("valence")
    q_arousal = request.query_params.get("arousal")
    q_valence = float(q_valence) if q_valence else None
    q_arousal = float(q_arousal) if q_arousal else None

    try:
        all_buckets = await bucket_mgr.list_all(include_archive=False)
        results = []
        w = {
            "topic": bucket_mgr.w_topic,
            "emotion": bucket_mgr.w_emotion,
            "time": bucket_mgr.w_time,
            "importance": bucket_mgr.w_importance,
            "warmth": bucket_mgr.w_warmth,  # bonus 项, 不进 w_sum 分母(跟评分公式保持一致)
        }
        w_sum = w["topic"] + w["emotion"] + w["time"] + w["importance"]

        for bucket in all_buckets:
            meta = bucket.get("metadata", {})
            bid = bucket["id"]
            try:
                topic = bucket_mgr._calc_topic_score(query, bucket) if query else 0.0
                emotion = bucket_mgr._calc_emotion_score(q_valence, q_arousal, meta)
                time_s = bucket_mgr._calc_time_score(meta)
                imp = max(1, min(10, int(meta.get("importance", 5)))) / 10.0
                # warmth bonus(只奖励 valence>0.5 的温暖桶, 不进分母)
                try:
                    b_val = float(meta.get("valence", 0.5))
                except (ValueError, TypeError):
                    b_val = 0.5
                warmth = max(0.0, b_val - 0.5)

                raw_total = (
                    topic * w["topic"]
                    + emotion * w["emotion"]
                    + time_s * w["time"]
                    + imp * w["importance"]
                    + warmth * w["warmth"]
                )
                normalized = (raw_total / w_sum) * 100 if w_sum > 0 else 0
                resolved = meta.get("resolved", False)
                if resolved:
                    normalized *= 0.3

                results.append({
                    "id": bid,
                    "name": meta.get("name", bid),
                    "domain": meta.get("domain", []),
                    "type": meta.get("type", "dynamic"),
                    "resolved": resolved,
                    "pinned": meta.get("pinned", False),
                    "protected": is_protected(meta),
                    "highlight": is_highlighted(meta),
                    "scores": {
                        "topic": round(topic, 4),
                        "emotion": round(emotion, 4),
                        "time": round(time_s, 4),
                        "importance": round(imp, 4),
                        "warmth": round(warmth, 4),
                    },
                    "weights": w,
                    "raw_total": round(raw_total, 4),
                    "normalized": round(normalized, 2),
                    "passed_threshold": normalized >= bucket_mgr.fuzzy_threshold,
                })
            except Exception:
                continue

        results.sort(key=lambda x: x["normalized"], reverse=True)
        passed = [r for r in results if r["passed_threshold"]]
        return JSONResponse({
            "query": query,
            "valence": q_valence,
            "arousal": q_arousal,
            "weights": w,
            "threshold": bucket_mgr.fuzzy_threshold,
            "total_candidates": len(results),
            "passed_count": len(passed),
            "results": results[:50],  # top 50 for debug
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/dashboard", methods=["GET"])
async def dashboard(request):
    """Serve the dashboard HTML page."""
    from starlette.responses import HTMLResponse
    import os
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    try:
        with open(dashboard_path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>dashboard.html not found</h1>", status_code=404)


# ─── v2 时间线视图(/v2) ───────────────────────────────────────────
# 静态托管 v2/ 目录(index.html + ombre-bridge.js + assets/*)。
# /v2  → v2/index.html
# /v2/<rel>  → v2/<rel>(限制在 v2/ 下,防穿越)
def _serve_v2(rel_path: str):
    from starlette.responses import Response, JSONResponse, RedirectResponse
    import os, mimetypes
    rel = (rel_path or "").lstrip("/")
    if not rel:
        rel = "index.html"
    # 安全:拒绝绝对路径 / 路径穿越
    norm = os.path.normpath(rel).replace("\\", "/")
    if norm.startswith("..") or os.path.isabs(norm):
        return JSONResponse({"error": "bad path"}, status_code=400)
    base = os.path.join(os.path.dirname(__file__), "v2")
    abs_path = os.path.join(base, norm)
    # 二次确认 abs_path 仍在 v2/ 下
    if not os.path.realpath(abs_path).startswith(os.path.realpath(base)):
        return JSONResponse({"error": "bad path"}, status_code=400)
    # console 子 tab 用真实路径(/v2/console/breath/ 等),都映射到同一个 index.html;
    # JS 端读 pathname 决定 tab,绕开浏览器初始 hash 的诡异行为
    console_base = os.path.join(base, "console")
    if abs_path.startswith(console_base) and not os.path.exists(abs_path):
        # 看是不是 /v2/console/{tab}/ 或 /v2/console/{tab}
        # 跨平台: Windows 上 abs_path 可能混用 \ 和 / (os.path.join + norm 里的 /),
        # 先统一成 / 再两头 strip, 否则 tail 会带前导 / 导致匹配失败 (本地 Windows 404)。
        tail = abs_path[len(console_base):].replace("\\", "/").strip("/")
        if tail in ("breath", "config", "import", "trash"):
            if not rel_path.endswith("/"):
                return RedirectResponse(url="/v2/console/" + tail + "/", status_code=301)
            abs_path = os.path.join(console_base, "index.html")
    # 子目录处理:/v2/cells/ → /v2/cells/index.html;
    # /v2/cells (无尾斜杠) → 301 → /v2/cells/(相对路径才会算对)
    if os.path.isdir(abs_path):
        if not rel_path.endswith("/"):
            return RedirectResponse(url="/v2/" + norm + "/", status_code=301)
        candidate = os.path.join(abs_path, "index.html")
        if os.path.isfile(candidate):
            abs_path = candidate
    if not os.path.isfile(abs_path):
        return JSONResponse({"error": "not found", "path": norm}, status_code=404)
    mime, _ = mimetypes.guess_type(abs_path)
    # 几个 mimetypes 默认认不出的兜底
    if not mime:
        if abs_path.endswith(".woff2"):
            mime = "font/woff2"
        elif abs_path.endswith(".jsx"):
            mime = "text/javascript"
        else:
            mime = "application/octet-stream"
    # Cache-Control: 朋友 Win wifi 反馈"切视图几分钟" 主要是每次重拉 4.3MB vendor JS + woff2。
    # 规则: assets/ 下 hash 命名(UUID)的资源永不变, 给一年 immutable; HTML / 手改脚本 no-cache 留 ETag 短路。
    import re as _re
    rel_under_v2 = os.path.relpath(abs_path, base).replace("\\", "/")
    fname = os.path.basename(rel_under_v2)
    is_hashed_asset = (
        "/assets/" in ("/" + rel_under_v2)
        and bool(_re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.(js|jsx|woff2)$", fname))
    )
    if is_hashed_asset:
        cache_header = "public, max-age=31536000, immutable"
    elif fname.endswith((".woff2", ".ico", ".png", ".svg")):
        # favicon/字体非 hash 命名的也按周缓存 — 改名换版本即失效
        cache_header = "public, max-age=604800"
    else:
        # index.html / ombre-bridge.js / theme-system.js / redehy-modal.{js,css} / manifest.json 等手改文件
        cache_header = "no-cache"
    with open(abs_path, "rb") as f:
        return Response(f.read(), media_type=mime, headers={"Cache-Control": cache_header})


# 手机 UA 嗅探: iPhone / iPod / Android 直接落到 /v2/mobile/,
# 否则桌面页。iPad 不在里面 — iPadOS 13+ UA 是 Macintosh, 也没必要给它 mobile 窄屏 UI。
# ?desktop=1 / ?mobile=1 query 强制覆盖。
def _is_mobile_ua(ua: str) -> bool:
    ua = (ua or "").lower()
    return ("iphone" in ua) or ("ipod" in ua) or ("android" in ua)


def _pick_v2_landing(request) -> str:
    qs = request.query_params
    if qs.get("desktop") == "1":
        return "/v2/"
    if qs.get("mobile") == "1":
        return "/v2/mobile/"
    ua = request.headers.get("user-agent", "")
    return "/v2/mobile/" if _is_mobile_ua(ua) else "/v2/"


@mcp.custom_route("/", methods=["GET"])
async def root_landing(request):
    """根域名: 手机 → /v2/mobile/, 桌面 → /v2/. 朋友手机访问根直接落 mobile 页, Safari 加桌面才能成 PWA."""
    from starlette.responses import RedirectResponse
    return RedirectResponse(url=_pick_v2_landing(request), status_code=302)


@mcp.custom_route("/v2", methods=["GET"])
async def v2_root(request):
    # 必须重定向到带尾斜杠,否则相对路径(./assets/...)的 base 会算错
    # 同时嗅 UA: 手机访问 /v2 也甩到 mobile (朋友可能 bookmark 了 /v2)
    from starlette.responses import RedirectResponse
    return RedirectResponse(url=_pick_v2_landing(request), status_code=302)


@mcp.custom_route("/v2/{rel:path}", methods=["GET"])
async def v2_rel(request):
    rel = request.path_params.get("rel", "")
    # 朋友可能 bookmark 了 /v2/ (桌面页), 手机进来仍要甩到 mobile。
    # 只在"明确请求桌面首页"时拦截 (rel == "" 或 "index.html"), 子路径 (console/cells/...) 不动。
    if rel in ("", "index.html"):
        from starlette.responses import RedirectResponse
        qs = request.query_params
        if qs.get("desktop") != "1" and _is_mobile_ua(request.headers.get("user-agent", "")):
            return RedirectResponse(url="/v2/mobile/", status_code=302)
    return _serve_v2(rel)


@mcp.custom_route("/api/config", methods=["GET"])
async def api_config_get(request):
    """Get current runtime config (safe fields only, API key masked)."""
    from starlette.responses import JSONResponse
    dehy = config.get("dehydration", {})
    emb = config.get("embedding", {})
    api_key = dehy.get("api_key", "")
    masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else ("***" if api_key else "")
    return JSONResponse({
        "dehydration": {
            "model": dehy.get("model", ""),
            "base_url": dehy.get("base_url", ""),
            "api_key_masked": masked_key,
            "max_tokens": dehy.get("max_tokens", 1024),
            "temperature": dehy.get("temperature", 0.1),
        },
        "embedding": {
            "enabled": emb.get("enabled", False),
            "model": emb.get("model", ""),
        },
        "merge_threshold": config.get("merge_threshold", 75),
        "transport": config.get("transport", "stdio"),
        "buckets_dir": config.get("buckets_dir", ""),
        "ai_name": get_ai_name(),
    })


@mcp.custom_route("/api/config", methods=["POST"])
async def api_config_update(request):
    """Hot-update runtime config. Optionally persist to config.yaml."""
    from starlette.responses import JSONResponse
    import yaml
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    updated = []

    # --- Dehydration config ---
    if "dehydration" in body:
        d = body["dehydration"]
        dehy = config.setdefault("dehydration", {})
        for key in ("model", "base_url", "max_tokens", "temperature"):
            if key in d:
                dehy[key] = d[key]
                updated.append(f"dehydration.{key}")
        if "api_key" in d and d["api_key"]:
            dehy["api_key"] = d["api_key"]
            updated.append("dehydration.api_key")
        # Hot-reload dehydrator
        dehydrator.model = dehy.get("model", "deepseek-chat")
        dehydrator.base_url = dehy.get("base_url", "")
        dehydrator.api_key = dehy.get("api_key", "")
        if hasattr(dehydrator, "client") and dehydrator.api_key:
            from openai import AsyncOpenAI
            dehydrator.client = AsyncOpenAI(
                api_key=dehydrator.api_key,
                base_url=dehydrator.base_url,
            )

    # --- Embedding config ---
    if "embedding" in body:
        e = body["embedding"]
        emb = config.setdefault("embedding", {})
        if "enabled" in e:
            emb["enabled"] = bool(e["enabled"])
            embedding_engine.enabled = emb["enabled"]
            updated.append("embedding.enabled")
        if "model" in e:
            emb["model"] = e["model"]
            embedding_engine.model = emb["model"]
            updated.append("embedding.model")

    # --- Merge threshold ---
    if "merge_threshold" in body:
        config["merge_threshold"] = int(body["merge_threshold"])
        updated.append("merge_threshold")

    # --- Persist to config.yaml if requested ---
    if body.get("persist", False):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
        try:
            save_config = {}
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    save_config = yaml.safe_load(f) or {}

            if "dehydration" in body:
                sc_dehy = save_config.setdefault("dehydration", {})
                for key in ("model", "base_url", "max_tokens", "temperature"):
                    if key in body["dehydration"]:
                        sc_dehy[key] = body["dehydration"][key]
                # Never persist api_key to yaml (use env var)

            if "embedding" in body:
                sc_emb = save_config.setdefault("embedding", {})
                for key in ("enabled", "model"):
                    if key in body["embedding"]:
                        sc_emb[key] = body["embedding"][key]

            if "merge_threshold" in body:
                save_config["merge_threshold"] = int(body["merge_threshold"])

            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(save_config, f, default_flow_style=False, allow_unicode=True)
            updated.append("persisted_to_yaml")
        except Exception as e:
            return JSONResponse({"error": f"persist failed: {e}", "updated": updated}, status_code=500)

    return JSONResponse({"updated": updated, "ok": True})


# =============================================================
# Import API — conversation history import
# 导入 API — 对话历史导入
# =============================================================

@mcp.custom_route("/api/import/upload", methods=["POST"])
async def api_import_upload(request):
    """Upload a conversation file and start import."""
    from starlette.responses import JSONResponse

    if import_engine.is_running:
        return JSONResponse({"error": "Import already running"}, status_code=409)

    content_type = request.headers.get("content-type", "")
    filename = ""

    try:
        if "multipart/form-data" in content_type:
            form = await request.form()
            file_field = form.get("file")
            if not file_field:
                return JSONResponse({"error": "No file field"}, status_code=400)
            raw_bytes = await file_field.read()
            filename = getattr(file_field, "filename", "upload")
            raw_content = raw_bytes.decode("utf-8", errors="replace")
        else:
            body = await request.body()
            raw_content = body.decode("utf-8", errors="replace")
            # Try to get filename from query params
            filename = request.query_params.get("filename", "upload")

        if not raw_content.strip():
            return JSONResponse({"error": "Empty file"}, status_code=400)

        preserve_raw = request.query_params.get("preserve_raw", "").lower() in ("1", "true")
        resume = request.query_params.get("resume", "").lower() in ("1", "true")
        # mode: 'small' (用户单独导一段, 强制至少 1 条) 或 'large' (默认, 宁缺勿滥)
        # 未知值 → 默认 large 兜底
        mode = request.query_params.get("mode", "large").lower()
        if mode not in ("small", "large"):
            mode = "large"
        try:
            max_chunks = int(request.query_params.get("max_chunks", "0") or "0")
        except (ValueError, TypeError):
            max_chunks = 0

    except Exception as e:
        return JSONResponse({"error": f"Failed to read upload: {e}"}, status_code=400)

    # Start import in background
    async def _run_import():
        try:
            await import_engine.start(raw_content, filename, preserve_raw, resume, max_chunks=max_chunks, mode=mode)
        except Exception as e:
            logger.error(f"Import failed: {e}")

    asyncio.create_task(_run_import())

    return JSONResponse({
        "status": "started",
        "filename": filename,
        "size_bytes": len(raw_content.encode()),
    })


@mcp.custom_route("/api/import/status", methods=["GET"])
async def api_import_status(request):
    """Get current import progress."""
    from starlette.responses import JSONResponse
    return JSONResponse(import_engine.get_status())


@mcp.custom_route("/api/import/pause", methods=["POST"])
async def api_import_pause(request):
    """Pause the running import."""
    from starlette.responses import JSONResponse
    if not import_engine.is_running:
        return JSONResponse({"error": "No import running"}, status_code=400)
    import_engine.pause()
    return JSONResponse({"status": "pause_requested"})


@mcp.custom_route("/api/import/patterns", methods=["GET"])
async def api_import_patterns(request):
    """Detect high-frequency patterns after import."""
    from starlette.responses import JSONResponse
    try:
        patterns = await import_engine.detect_patterns()
        return JSONResponse({"patterns": patterns})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/import/results", methods=["GET"])
async def api_import_results(request):
    """List recently imported/created buckets for review (导入工作台用).

    Query params:
      limit: 拉取上限 (默认 50)
      source: 'all' (默认, 含所有来源) | 'import' / 'ai' / 'user' (按来源过滤)
        - 工作台是用户的主整理面, 默认拉全部 (包括历史 'ai' 桶, 让用户在
          展开视图里逐条改 source). 想只看新导入传 source=import.
    """
    from starlette.responses import JSONResponse
    try:
        limit = int(request.query_params.get("limit", "50"))
        source_filter = request.query_params.get("source", "all")
        all_buckets = await bucket_mgr.list_all(include_archive=False)
        # Sort by created time, newest first
        all_buckets.sort(key=lambda b: b["metadata"].get("created", ""), reverse=True)
        # 来源过滤 (端点级) — 默认 'all' 不过滤; 传 'import'/'ai'/'user' 才过滤
        # 老桶缺 created_by 按 'ai' 计 (跟 list 端点 default 一致)
        if source_filter != "all":
            all_buckets = [
                b for b in all_buckets
                if b.get("metadata", {}).get("created_by", "ai") == source_filter
            ]
        results = []
        for b in all_buckets[:limit]:
            meta = b.get("metadata", {})
            results.append({
                "id": b["id"],
                "name": meta.get("name", ""),
                "content": (b.get("content") or "")[:300],
                "summary": meta.get("summary", ""),  # 用户编辑过的摘要,空则前端回退到 content 前 160 字
                "type": meta.get("type", "dynamic"),
                "domain": meta.get("domain", []),
                "tags": meta.get("tags", []),
                "importance": meta.get("importance", 5),
                "created": meta.get("created", ""),
                "event_time": meta.get("event_time", ""),
                "created_by": meta.get("created_by", "ai"),  # 来源 user/ai/import
                "protected": is_protected(meta),
                "highlight": is_highlighted(meta),
                "pinned": is_protected(meta) or is_highlighted(meta),
                "internalized": is_internalized(meta),
                "resolved": meta.get("resolved", False),
                "raw_source": meta.get("raw_source", ""),  # preserve_raw=1 时存的原文
                "valence": meta.get("valence", 0.5),
                "arousal": meta.get("arousal", 0.3),
                "score": decay_engine.calculate_score(meta),  # 工作台权重显示用
            })
        return JSONResponse({"buckets": results, "total": len(all_buckets)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@mcp.custom_route("/api/import/review", methods=["POST"])
async def api_import_review(request):
    """Apply review decisions: mark buckets as important/noise/pinned."""
    from starlette.responses import JSONResponse
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    decisions = body.get("decisions", [])
    if not decisions:
        return JSONResponse({"error": "No decisions provided"}, status_code=400)

    applied = 0
    errors = 0
    for d in decisions:
        bid = d.get("bucket_id", "")
        action = d.get("action", "")
        if not bid or not action:
            continue
        try:
            if action == "important":
                await bucket_mgr.update(bid, importance=9)
            elif action == "pin":
                await bucket_mgr.update(bid, pinned=True)
            elif action == "noise":
                await bucket_mgr.update(bid, resolved=True, importance=1)
            elif action == "delete":
                file_path = bucket_mgr._find_bucket_file(bid)
                if file_path:
                    os.remove(file_path)
            applied += 1
        except Exception as e:
            logger.warning(f"Review action failed for {bid}: {e}")
            errors += 1

    return JSONResponse({"applied": applied, "errors": errors})


# --- Entry point / 启动入口 ---
if __name__ == "__main__":
    transport = config.get("transport", "stdio")
    logger.info(f"Ombre Brain starting | transport: {transport}")

    # 启动时把 runtime_config['decay'] 应用到 decay_engine, 让用户改的值在重启后还在
    try:
        _reload_decay_from_runtime()
        logger.info("Decay runtime overrides applied")
    except Exception as e:
        logger.warning(f"Decay runtime overrides apply failed (using defaults): {e}")

    # 同上, runtime_config['prompts'] → dehydrator.set_prompts
    try:
        _reload_prompts_from_runtime()
        logger.info("Prompt runtime overrides applied")
    except Exception as e:
        logger.warning(f"Prompt runtime overrides apply failed (using defaults): {e}")

    # runtime_config['scoring'] → bucket_mgr.apply_runtime_scoring_overrides
    try:
        _reload_scoring_from_runtime()
        logger.info("Scoring runtime overrides applied")
    except Exception as e:
        logger.warning(f"Scoring runtime overrides apply failed (using defaults): {e}")

    # --- 数据安全自检: off-site 自动备份是否配置 ---
    # 持久盘扛重启, 但盘本身损坏/误删/平台事故会全损(维护者亲历过一次数据丢失)。
    # 推 buckets 到私有 git 的自动备份是唯一的"第二层"保命 —— 没配就显眼提醒, 别让用户裸奔。
    _backup_ready = all(
        os.environ.get(k, "").strip()
        for k in ("OMBRE_BACKUP_REPO", "OMBRE_BACKUP_TOKEN")  # USER 有默认值, 非必需
    )
    if _backup_ready:
        logger.info("✓ 自动备份已配置 — 记忆数据有 off-site 兜底")
    else:
        logger.warning(
            "⚠⚠ 未配置自动备份 (OMBRE_BACKUP_REPO/TOKEN/USER)。"
            " 数据当前只靠单块持久盘 —— 盘损坏/误删/平台事故将无法恢复。"
            " 强烈建议按 DEPLOY.md「自动备份配置」设一个 off-site 备份(几分钟, 救命)。"
        )

    if transport in ("sse", "streamable-http"):
        # ============================================================
        # fail-safe: 公网 transport 必须设 OMBRE_ADMIN_TOKEN, 否则拒绝启动。
        # 保护一键部署的用户 —— 没有这道门, 任何拿到 URL 的人可读光/删光全部记忆。
        # stdio 模式 (本地 MCP 客户端) 不受影响。可用 OMBRE_ALLOW_NO_AUTH=1 显式豁免
        # (仅限你确定服务在私网 / 反代后面已有鉴权时)。
        # ============================================================
        if not os.environ.get("OMBRE_ADMIN_TOKEN", "").strip():
            if os.environ.get("OMBRE_ALLOW_NO_AUTH", "").strip() == "1":
                logger.warning(
                    "⚠⚠⚠ OMBRE_ADMIN_TOKEN 未设, 但 OMBRE_ALLOW_NO_AUTH=1 —— 服务在无鉴权模式下启动。"
                    " 任何能访问此 URL 的人都能读/删你的全部记忆。仅限私网/反代已鉴权场景!"
                )
            else:
                logger.error(
                    "\n" + "=" * 64 + "\n"
                    "🔴 拒绝启动: 公网 transport (%s) 必须设置 OMBRE_ADMIN_TOKEN。\n"
                    "   没有它, 任何拿到本服务 URL 的人都能读取/删除你的全部记忆\n"
                    "   (含私密 feel), 还能改系统提示词 / 换 LLM 地址截走数据。\n"
                    "   → 在部署平台 (Render/Railway/Docker) 设一个强随机 OMBRE_ADMIN_TOKEN。\n"
                    "   REFUSING TO START: a public transport requires OMBRE_ADMIN_TOKEN.\n"
                    "   Set a strong random OMBRE_ADMIN_TOKEN in your env, then redeploy.\n"
                    "   (本地无鉴权调试可设 OMBRE_ALLOW_NO_AUTH=1 显式豁免。)\n"
                    + "=" * 64,
                    transport,
                )
                sys.exit(1)

        import threading
        import uvicorn
        from starlette.middleware.cors import CORSMiddleware

        # --- Application-level keepalive: ping /health every 60s ---
        # --- 应用层保活：每 60 秒 ping 一次 /health，防止 Cloudflare Tunnel 空闲断连 ---
        async def _keepalive_loop():
            await asyncio.sleep(10)  # Wait for server to fully start
            async with httpx.AsyncClient() as client:
                while True:
                    try:
                        await client.get("http://localhost:8000/health", timeout=5)
                        logger.debug("Keepalive ping OK / 保活 ping 成功")
                    except Exception as e:
                        logger.warning(f"Keepalive ping failed / 保活 ping 失败: {e}")
                    await asyncio.sleep(60)

        def _start_keepalive():
            loop = asyncio.new_event_loop()
            loop.run_until_complete(_keepalive_loop())

        t = threading.Thread(target=_start_keepalive, daemon=True)
        t.start()

        # --- Add CORS middleware so remote clients (Cloudflare Tunnel / ngrok) can connect ---
        # --- 添加 CORS 中间件，让远程客户端（Cloudflare Tunnel / ngrok）能正常连接 ---
        from starlette.middleware.gzip import GZipMiddleware
        if transport == "streamable-http":
            _app = mcp.streamable_http_app()
        else:
            _app = mcp.sse_app()

        # ============================================================
        # 全局鉴权中间件 (AuthGate) —— 默认拒绝 (default-deny)
        # ------------------------------------------------------------
        # 公网部署时, 除了静态 UI 页 + /health 外, 一切请求 (/api/*, /mcp,
        # *-hook) 都必须带正确的 X-Admin-Token header, 否则 401。
        #   · 只收 header, 不收 ?token= —— query 会进访问日志 / Referer / 浏览器历史。
        #   · 用 hmac.compare_digest 比对, 防时序侧信道。
        #   · 静态页 (/, /v2/*, /dashboard, favicon, manifest) 放行: 浏览器导航
        #     不会带自定义 header, 页面/JS/CSS/字体必须能加载; 真正的数据全部走
        #     /api/* (前端 fetch 会自动补 header), 静态页本身不含记忆内容。
        #   · 敏感前缀 (/api /mcp *-hook) 优先要门, 即便被未知/伪装路径绕也拦得住。
        #   · 配合下方启动期 fail-safe (公网无 token 直接拒启), expected 正常不会为空;
        #     万一为空, 这里对要门路径一律 401 (fail-closed, 绝不放行裸奔)。
        # ============================================================
        import hmac as _hmac
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.responses import JSONResponse as _AuthJSONResponse

        _PUBLIC_EXACT = {"/", "/health", "/dashboard"}
        _PUBLIC_PREFIXES = ("/v2", "/favicon", "/manifest")

        def _is_public_path(path: str) -> bool:
            if path in _PUBLIC_EXACT:
                return True
            return any(path == p or path.startswith(p + "/") or path.startswith(p) for p in _PUBLIC_PREFIXES)

        class AuthGate(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                path = request.url.path
                # 路径段方案 (b): 外层 McpUrlKeyPath 已用 compare_digest 校验过 /<key>/mcp
                # 的密钥并剥成 /mcp + 打标 → 这里直接放行 (它只会给 /mcp 打标, 不碰 /api/*)。
                if request.scope.get("_mcp_url_key_ok"):
                    return await call_next(request)
                sensitive = (
                    path.startswith("/api")
                    or path.startswith("/mcp")
                    or path in ("/breath-hook", "/dream-hook")
                )
                # 非敏感: 预检 OPTIONS / 已知静态页放行 (default-deny: 未知路由仍要门)
                if not sensitive and (request.method == "OPTIONS" or _is_public_path(path)):
                    return await call_next(request)
                # 要门
                expected = os.environ.get("OMBRE_ADMIN_TOKEN", "").strip()
                # 显式无鉴权豁免 (本地 demo/调试): 仅当未设 token 且显式 OMBRE_ALLOW_NO_AUTH=1
                # 时放行 —— 跟启动期 fail-safe 同一开关。不设这个开关则 fail-closed 不变 (绝不裸奔)。
                if not expected and os.environ.get("OMBRE_ALLOW_NO_AUTH", "").strip() == "1":
                    return await call_next(request)
                provided = request.headers.get("X-Admin-Token", "")
                if expected and provided and _hmac.compare_digest(provided, expected):
                    return await call_next(request)
                # cookie 兜底: dashboard 工作台等页面用 Web Worker 拉数据, 这些请求不经
                # 前端 fetch/XHR 的 header 注入(worker 读不到 localStorage), 但浏览器会自动
                # 带上同源 cookie(auth-token.js 已把 token 同步进 ombre_admin_token cookie)。
                # cookie 走 SameSite=Strict + CORS 已收紧到同源 → CSRF 风险可控。header 仍是首选。
                _cookie_tok = request.cookies.get("ombre_admin_token", "")
                if expected and _cookie_tok and _hmac.compare_digest(_cookie_tok, expected):
                    return await call_next(request)
                # --- /mcp URL-key 旁路 (opt-in, 仅 /mcp 这一条口子) ---
                # claude.ai 网页连接器只有 URL 字段、配不了自定义 header → 给 /mcp 开一条
                # "URL 带密钥"的口子, 让网页版也能连带鉴权的 OB。
                #   · 独立 env OMBRE_MCP_URL_KEY (≠ ADMIN_TOKEN): 泄漏 URL key 只给 MCP 读写删
                #     记忆的能力, 不暴露守 /api/* 销毁/config/profile 的强 header token, 可独立轮换。
                #   · 默认不设 = 纯 header 模式不变, /api/* 永不受此影响。
                #   · 两种 URL 形态并存, 连接器二选一即可:
                #     (a) query: https://host/mcp?key=<KEY>   ← 此处校验
                #     (b) path : https://host/<KEY>/mcp       ← 外层 McpUrlKeyPath 校验+改写
                #     (b) 不依赖客户端在后续请求保留 query, 对 claude.ai 这类未知行为更稳。
                #   · compare_digest 防时序侧信道。
                #   · key 已在 uvicorn access log 过滤器里脱敏 (见下方 _MaskUrlKeyFilter), 不进日志。
                if path.startswith("/mcp"):
                    url_key_expected = os.environ.get("OMBRE_MCP_URL_KEY", "").strip()
                    url_key_provided = request.query_params.get("key", "")
                    if (
                        url_key_expected
                        and url_key_provided
                        and _hmac.compare_digest(url_key_provided, url_key_expected)
                    ):
                        return await call_next(request)
                return _AuthJSONResponse(
                    {"error": "unauthorized — missing or invalid X-Admin-Token"},
                    status_code=401,
                )

        # 路径段 URL-key 方案 (b): 纯 ASGI 中间件, 放在最外层 (最先看到原始 path)。
        # 把密钥放进【路径段】—— 连接器 URL = https://host/<KEY>/mcp。密钥永远在 URL 路径里,
        # 客户端每次请求都必然带上, 不依赖 query 保留 → 对 claude.ai 这类未知行为更稳。
        # 命中(compare_digest)后剥掉 /<KEY> 前缀、把 path 改写回 /mcp 交给内层, 并打 scope 标记
        # 让 AuthGate 放行。只认 /<KEY>/mcp(及其子路径), 绝不碰 /<KEY>/api 之类 → 仍只开 /mcp。
        # ⚠ 路径段方案要求 KEY 是 URL 安全的单段 (hex / 字母数字, 无 / 与特殊字符)。
        class McpUrlKeyPath:
            def __init__(self, app):
                self.app = app

            async def __call__(self, scope, receive, send):
                if scope.get("type") == "http":
                    key = os.environ.get("OMBRE_MCP_URL_KEY", "").strip()
                    path = scope.get("path", "")
                    if key and path.startswith("/"):
                        parts = path.split("/", 2)  # ["", "<seg>", "rest..."]
                        seg = parts[1] if len(parts) > 1 else ""
                        rest = "/" + parts[2] if len(parts) > 2 else ""
                        # 必须形如 /<seg>/mcp 或 /<seg>/mcp/... 且 <seg> 是正确密钥
                        if (
                            seg
                            and (rest == "/mcp" or rest.startswith("/mcp/"))
                            and _hmac.compare_digest(seg, key)
                        ):
                            scope = dict(scope)
                            scope["path"] = rest
                            scope["raw_path"] = rest.encode("latin-1")
                            scope["_mcp_url_key_ok"] = True
                await self.app(scope, receive, send)

        # add_middleware 后加的在外层 → 先加 AuthGate (内层), 最后加 CORS (外层),
        # 让 CORS 先处理跨源预检 OPTIONS, 不会被 AuthGate 误拦。
        _app.add_middleware(AuthGate)

        # gzip: vendor JS 4.3MB + 200+ 桶 JSON 都裸传输, 朋友 Win wifi 切视图几分钟。
        # 压缩后 vendor ~1MB, JSON ~150KB, 直接 4x 提速。minimum_size=1024 跳过小响应。
        _app.add_middleware(GZipMiddleware, minimum_size=1024)

        # CORS: 原本 allow_origins=["*"] 任何网页 JS 都能跨源读响应 (没鉴权时是大洞)。
        # 收紧到 OMBRE_ALLOWED_ORIGINS (逗号分隔) 解析的白名单; 默认空 = 仅同源
        # (dashboard 由本服务同源伺服; pixel-ai / claude.ai 都是服务端调用, 无浏览器 CORS)。
        _cors_raw = os.environ.get("OMBRE_ALLOWED_ORIGINS", "").strip()
        _allowed_origins = [o.strip() for o in _cors_raw.split(",") if o.strip()]
        _app.add_middleware(
            CORSMiddleware,
            allow_origins=_allowed_origins,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["*"],
        )
        # 最后加 = 最外层: 先于一切看到原始 path, 校验/改写路径段密钥 (/<KEY>/mcp)。
        _app.add_middleware(McpUrlKeyPath)
        logger.info(
            "AuthGate + CORS + GZip middleware enabled / 已启用 鉴权 + CORS + GZip 中间件"
            f" (CORS origins={_allowed_origins or '同源 only'})"
        )

        # uvicorn access log 默认把请求行 (含 query string + path) 写进日志 → 会把
        # /mcp?key=<KEY> (query 方案) 或 /<KEY>/mcp (path 方案) 里的密钥明文留在日志 (泄漏隐患)。
        # 装一个过滤器双重脱敏: ① query 里 key= 的值 (lookbehind [?&] 防误伤 "...monkey=");
        # ② 密钥字面值本身 (覆盖 path 段形态; key 是高熵随机串, 整串替换安全)。仅作用于日志渲染。
        import logging as _logging
        import re as _re
        _MASK_KEY_RE = _re.compile(r'(?<=[?&])key=[^&\s"\']+')
        _URL_KEY_VAL = os.environ.get("OMBRE_MCP_URL_KEY", "").strip()

        class _MaskUrlKeyFilter(_logging.Filter):
            def filter(self, record):
                try:
                    if record.args and isinstance(record.args, tuple):
                        def _red(a):
                            if isinstance(a, str):
                                a = _MASK_KEY_RE.sub("key=***", a)
                                if _URL_KEY_VAL and _URL_KEY_VAL in a:
                                    a = a.replace(_URL_KEY_VAL, "***")
                            return a
                        record.args = tuple(_red(a) for a in record.args)
                except Exception:
                    pass
                return True

        _logging.getLogger("uvicorn.access").addFilter(_MaskUrlKeyFilter())

        uvicorn.run(_app, host="0.0.0.0", port=8000)
    else:
        mcp.run(transport=transport)
