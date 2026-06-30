# ============================================================
# Module: Memory Decay Engine (decay_engine.py)
# 模块：记忆衰减引擎
#
# Simulates human forgetting curve; auto-decays inactive memories and archives them.
# 模拟人类遗忘曲线，自动衰减不活跃记忆并归档。
#
# Core formula (improved Ebbinghaus + emotion coordinates):
# 核心公式（改进版艾宾浩斯遗忘曲线 + 情感坐标）：
#   Score = Importance × (activation_count^0.3) × e^(-λ×days) × emotion_weight
#
# Emotion weight (continuous coordinate, not discrete labels):
# 情感权重（基于连续坐标而非离散列举）：
#   emotion_weight = base + (arousal × arousal_boost)
#   Higher arousal → higher emotion weight → slower decay
#   唤醒度越高 → 情感权重越大 → 记忆衰减越慢
#
# Depended on by: server.py
# 被谁依赖：server.py
# ============================================================

import math
import asyncio
import logging
from datetime import datetime

from utils import is_internalized, is_protected, is_highlighted

logger = logging.getLogger("ombre_brain.decay")


class DecayEngine:
    """
    Memory decay engine — periodically scans all dynamic buckets,
    calculates decay scores, auto-archives low-activity buckets
    to simulate natural forgetting.
    记忆衰减引擎 —— 定期扫描所有动态桶，
    计算衰减得分，将低活跃桶自动归档，模拟自然遗忘。
    """

    # ---------------------------------------------------------
    # Default values — 这些是出厂默认, runtime_config 可覆盖
    # 前端"恢复默认"按钮回到这套值
    # ---------------------------------------------------------
    DEFAULTS = {
        "feel_score": 50.0,            # feel 桶基础权重 (锁定值, 防"心动时刻"被自然遗忘); 设 0 → 跟随 importance 公式
        "protected_score": 999.0,      # protected/permanent 桶分数 (对齐上游; 实质=永不衰减, 999 与 100 功能等价)
        "highlight_boost_pct": 30.0,   # highlight=true 时 score *= (1 + pct/100)
        "surface_threshold": 5.0,      # score 高于此值 → 标记为"活跃" (UI 提示用)
        "archive_threshold": 0.3,      # score 低于此值 → 自动归档
        "decay_lambda": 0.05,          # 时间衰减速率 (大→快)
        "arousal_boost": 0.8,          # arousal 对 emotion_weight 的加成系数
        "emotion_base": 1.0,           # 情感权重基线
        "resolved_factor": 0.05,       # 标 resolved/noise 时的衰减乘数
        "internalized_resolved_factor": 0.02,  # resolved + internalized 双标加速
    }

    def __init__(self, config: dict, bucket_mgr):
        # --- Load decay parameters / 加载衰减参数 ---
        decay_cfg = config.get("decay", {})
        # check_interval 不暴露 UI(后台任务调度), 沿用 config
        self.check_interval = decay_cfg.get("check_interval_hours", 24)

        # 所有可调参数初始化为默认 (后续 apply_runtime_overrides 会覆盖)
        for k, v in self.DEFAULTS.items():
            setattr(self, k, v)
        # 兼容旧字段名(仍被外部代码读)
        self.threshold = self.archive_threshold

        # config.yaml 里的初值覆盖默认 (启动时一次)
        if "lambda" in decay_cfg:
            self.decay_lambda = float(decay_cfg["lambda"])
        if "threshold" in decay_cfg:
            self.archive_threshold = float(decay_cfg["threshold"])
            self.threshold = self.archive_threshold
        emotion_cfg = decay_cfg.get("emotion_weights", {})
        if "base" in emotion_cfg:
            self.emotion_base = float(emotion_cfg["base"])
        if "arousal_boost" in emotion_cfg:
            self.arousal_boost = float(emotion_cfg["arousal_boost"])

        self.bucket_mgr = bucket_mgr

        # --- Background task control / 后台任务控制 ---
        self._task: asyncio.Task | None = None
        self._running = False

    def apply_runtime_overrides(self, overrides: dict) -> None:
        """前端 POST /api/decay-config 后调用,把 dict 里的字段设到 instance.
        无效 key 静默忽略;非法值类型转换失败也忽略不抛。"""
        if not isinstance(overrides, dict):
            return
        for k, default_v in self.DEFAULTS.items():
            if k in overrides:
                try:
                    new_v = float(overrides[k])
                    setattr(self, k, new_v)
                except (TypeError, ValueError):
                    pass
        # 同步老字段
        self.threshold = self.archive_threshold

    def current_overrides(self) -> dict:
        """读当前所有可调参数(用于 GET endpoint 返回)。"""
        return {k: getattr(self, k, v) for k, v in self.DEFAULTS.items()}

    @property
    def is_running(self) -> bool:
        """Whether the decay engine is running in the background.
        衰减引擎是否正在后台运行。"""
        return self._running

    # ---------------------------------------------------------
    # Core: calculate decay score for a single bucket
    # 核心：计算单个桶的衰减得分
    #
    # Higher score = more vivid memory; below threshold → archive
    # 得分越高 = 记忆越鲜活，低于阈值则归档
    # Permanent buckets never decay / 固化桶永远不衰减
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # Freshness bonus: continuous exponential decay
    # 新鲜度加成：连续指数衰减
    # bonus = 1.0 + 1.0 × e^(-t/36), t in hours
    # t=0 → 2.0×, t≈25h(半衰) → 1.5×, t≈72h → ≈1.14×, t→∞ → 1.0×
    # ---------------------------------------------------------
    @staticmethod
    def _calc_time_weight(days_since: float) -> float:
        """
        Freshness bonus multiplier: 1.0 + e^(-t/36), t in hours.
        新鲜度加成乘数：刚存入×2.0，~36小时半衰，72小时后趋近×1.0。
        """
        hours = days_since * 24.0
        return 1.0 + 1.0 * math.exp(-hours / 36.0)

    def calculate_score(self, metadata: dict) -> float:
        """
        Calculate current activity score for a memory bucket.
        计算一个记忆桶的当前活跃度得分。

        New model: short-term vs long-term weight separation.
        新模型：短期/长期权重分离。
        - Short-term (≤3 days): time_weight dominates, emotion amplifies
        - Long-term (>3 days): emotion_weight dominates, time decays to floor
        短期（≤3天）：时间权重主导，情感放大
        长期（>3天）：情感权重主导，时间衰减到底线
        """
        if not isinstance(metadata, dict):
            return 0.0

        # --- Protected buckets: never decay (highlight 单独不防衰减) ---
        if is_protected(metadata):
            return self.protected_score

        # --- Permanent buckets never decay ---
        if metadata.get("type") == "permanent":
            return self.protected_score

        # --- Feel buckets: 锁定 feel_score; 设为 0 时回退跟随 importance 公式 ---
        if metadata.get("type") == "feel" and self.feel_score > 0:
            return self.feel_score

        importance = max(1, min(10, int(metadata.get("importance", 5))))
        # B-03 对齐上游: float 不截断 → time-ripple 的 +0.3 涟漪(相邻桶连带激活)能真正生效
        activation_count = max(1.0, float(metadata.get("activation_count", 1)))

        # --- Days since last activation ---
        last_active_str = metadata.get("last_active", metadata.get("created", ""))
        try:
            last_active = datetime.fromisoformat(str(last_active_str))
            days_since = max(0.0, (datetime.now() - last_active).total_seconds() / 86400)
        except (ValueError, TypeError):
            days_since = 30

        # --- Emotion weight ---
        try:
            arousal = max(0.0, min(1.0, float(metadata.get("arousal", 0.3))))
        except (ValueError, TypeError):
            arousal = 0.3
        emotion_weight = self.emotion_base + arousal * self.arousal_boost

        # --- Time weight ---
        time_weight = self._calc_time_weight(days_since)

        # --- Short-term vs Long-term weight separation ---
        # 短期（≤3天）：time_weight 占 70%，emotion 占 30%
        # 长期（>3天）：emotion 占 70%，time_weight 占 30%
        if days_since <= 3.0:
            # Short-term: time dominates, emotion amplifies
            combined_weight = time_weight * 0.7 + emotion_weight * 0.3
        else:
            # Long-term: emotion dominates, time provides baseline
            combined_weight = emotion_weight * 0.7 + time_weight * 0.3

        # --- Base score ---
        base_score = (
            importance
            * (activation_count ** 0.3)
            * math.exp(-self.decay_lambda * days_since)
            * combined_weight
        )

        # --- Weight pool modifiers ---
        # resolved + internalized → accelerated fade: ×0.02
        # resolved only → ×0.05
        # 已处理+已内化 → 加速淡化:×0.02
        # 仅已处理 → ×0.05
        resolved = metadata.get("resolved", False)
        internalized = is_internalized(metadata)
        if resolved and internalized:
            resolved_factor = self.internalized_resolved_factor
        elif resolved:
            resolved_factor = self.resolved_factor
        else:
            resolved_factor = 1.0
        urgency_boost = 1.5 if (arousal > 0.7 and not resolved) else 1.0

        # --- Highlight 加成: 标记重要的桶 score 上浮 highlight_boost_pct % ---
        highlight_mult = 1.0 + (self.highlight_boost_pct / 100.0) if is_highlighted(metadata) else 1.0

        return round(base_score * resolved_factor * urgency_boost * highlight_mult, 4)

    # ---------------------------------------------------------
    # Execute one decay cycle
    # 执行一轮衰减周期
    # Scan all dynamic buckets → score → archive those below threshold
    # 扫描所有动态桶 → 算分 → 低于阈值的归档
    # ---------------------------------------------------------
    async def run_decay_cycle(self) -> dict:
        """
        Execute one decay cycle: iterate dynamic buckets, archive those
        scoring below threshold.
        执行一轮衰减：遍历动态桶，归档得分低于阈值的桶。

        Returns stats: {"checked": N, "archived": N, "lowest_score": X}
        """
        try:
            buckets = await self.bucket_mgr.list_all(include_archive=False)
        except Exception as e:
            logger.error(f"Failed to list buckets for decay / 衰减周期列桶失败: {e}")
            return {"checked": 0, "archived": 0, "lowest_score": 0, "error": str(e)}

        checked = 0
        archived = 0
        lowest_score = float("inf")
        demoted_orphans = 0

        for bucket in buckets:
            meta = bucket.get("metadata", {})

            # --- Self-heal: 孤儿固化桶（type==permanent 却没 protected）---
            # 早期 unprotect 只翻 protected 标记、没把 type 降级回 dynamic 的历史遗留。
            # 这类桶 calculate_score 恒返 999（权重卡死、永不衰减、永远霸占召回置顶）。
            # 后台衰减循环每轮扫全库，顺手把孤儿对称降级回 dynamic。
            if meta.get("type") == "permanent" and not is_protected(meta):
                try:
                    await self.bucket_mgr.update(bucket["id"], protected=False)
                    demoted_orphans += 1
                    logger.info(
                        f"Decay self-heal / 自愈降级孤儿固化桶: "
                        f"{meta.get('name', bucket['id'])} ({bucket['id']})"
                    )
                except Exception as e:
                    logger.warning(f"Decay self-heal failed / 自愈降级失败 {bucket.get('id', '?')}: {e}")
                continue

            # Skip permanent / feel / protected buckets
            # 跳过固化桶、feel 桶(心动时刻防遗忘)、保护桶(防衰减)。
            # highlight 单独不防衰减,仍参与衰减/归档,只是浮现时被推到核心准则区。
            if meta.get("type") in ("permanent", "feel") or is_protected(meta):
                continue

            checked += 1

            # 注:原本这里有 auto-resolve 机制(imp≤4 + 30 天没动 → 自动 resolved=True)
            # 在 2026-04-26 切片 3 关掉。原因:resolved 在新语义下=用户确认兑现的待办,
            # 让系统替用户宣布"这事完成了"会污染待办视图,且用户对此机制不知情。
            # 自然衰减归档(下方 score < threshold 路径)仍在工作,清理低活跃记忆的目标
            # 由那条路径承担,只是路径变成"被遗忘"而不是"被宣布完成"。

            try:
                score = self.calculate_score(meta)
            except Exception as e:
                logger.warning(
                    f"Score calculation failed for {bucket.get('id', '?')} / "
                    f"计算得分失败: {e}"
                )
                continue

            lowest_score = min(lowest_score, score)

            # --- Below threshold → archive (simulate forgetting) ---
            # --- 低于阈值 → 归档（模拟遗忘）---
            if score < self.threshold:
                try:
                    success = await self.bucket_mgr.archive(bucket["id"])
                    if success:
                        archived += 1
                        logger.info(
                            f"Decay archived / 衰减归档: "
                            f"{meta.get('name', bucket['id'])} "
                            f"(score={score:.4f}, threshold={self.threshold})"
                        )
                except Exception as e:
                    logger.warning(
                        f"Archive failed for {bucket.get('id', '?')} / "
                        f"归档失败: {e}"
                    )

        result = {
            "checked": checked,
            "archived": archived,
            "lowest_score": lowest_score if checked > 0 else 0,
            "demoted_orphans": demoted_orphans,
        }
        logger.info(f"Decay cycle complete / 衰减周期完成: {result}")
        return result

    # ---------------------------------------------------------
    # Background decay task management
    # 后台衰减任务管理
    # ---------------------------------------------------------
    async def ensure_started(self) -> None:
        """
        Ensure the decay engine is started (lazy init on first call).
        确保衰减引擎已启动（懒加载，首次调用时启动）。
        """
        if not self._running:
            await self.start()

    async def start(self) -> None:
        """Start the background decay loop.
        启动后台衰减循环。"""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._background_loop())
        logger.info(
            f"Decay engine started, interval: {self.check_interval}h / "
            f"衰减引擎已启动，检查间隔: {self.check_interval} 小时"
        )

    async def stop(self) -> None:
        """Stop the background decay loop.
        停止后台衰减循环。"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Decay engine stopped / 衰减引擎已停止")

    async def _background_loop(self) -> None:
        """Background loop: run decay → sleep → repeat.
        后台循环体：执行衰减 → 睡眠 → 重复。"""
        while self._running:
            try:
                await self.run_decay_cycle()
            except Exception as e:
                logger.error(f"Decay cycle error / 衰减周期出错: {e}")
            # --- Wait for next cycle / 等待下一个周期 ---
            try:
                await asyncio.sleep(self.check_interval * 3600)
            except asyncio.CancelledError:
                break
