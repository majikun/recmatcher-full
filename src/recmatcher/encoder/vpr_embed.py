from typing import List
import numpy as np
import logging
import os
import math
import cv2  # 用于 resize/颜色变换（Jetson 上已可用）

# ---------------- 在导入 JAX 之前，设置 XLA/JAX 环境（兼容 jaxlib==0.5.0） ----------------
def _set_xla_flags(flags: list[str]):
    # 覆盖为“已知可用”的子集，避免旧版 XLA 因未知 flag 直接 FATAL
    dedup = []
    for f in flags:
        if f and f not in dedup:
            dedup.append(f)
    os.environ["XLA_FLAGS"] = " ".join(dedup).strip()

_set_xla_flags([
    "--xla_gpu_cuda_data_dir=/usr/local/cuda",
    "--xla_gpu_autotune_level=0",   # 关闭 GEMM autotune，规避不兼容 kernel
])

# 其他建议设置（可减少显存占用和噪声）
os.environ.setdefault("JAX_PLATFORMS", "cuda")                  # 不探测 rocm/tpu
os.environ.setdefault("JAX_ENABLE_X64", "0")                    # 省显存/更快
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.70") # 预分配 70% 显存
# ------------------------------------------------------------------------------------

# ---------------- JAX debug_info 兼容层（JAX>=0.5.0 已移除 api_util.debug_info）----------------
try:
    import jax  # noqa: F401
    try:
        from jax.api_util import debug_info as _jax_debug_info  # noqa: F401
    except Exception:
        import types
        class _NoopDebugInfo:
            def __init__(self, *a, **k): pass
            def __enter__(self): return self
            def __exit__(self, exc_type, exc, tb): return False
        import jax as _jax_mod
        if not hasattr(_jax_mod, "api_util"):
            _jax_mod.api_util = types.SimpleNamespace()
        if not hasattr(_jax_mod.api_util, "debug_info"):
            _jax_mod.api_util.debug_info = lambda *a, **k: _NoopDebugInfo()
except Exception:
    pass
# ---------------------------------------------------------------------------------------------

# ---------------- 固定形状/批量参数（可用环境变量覆盖） ----------------
TARGET_T = int(os.getenv("RECMATCHER_VPR_T", "8"))     # 每段采样/填充后的帧数
TARGET_H = int(os.getenv("RECMATCHER_VPR_H", "288"))   # 高
TARGET_W = int(os.getenv("RECMATCHER_VPR_W", "512"))   # 宽
BATCH_SIZE = int(os.getenv("RECMATCHER_VPR_BS", "4"))  # 每批 clip 数
# ---------------------------------------------------------------------

def _sample_to_T(x: np.ndarray, T: int) -> np.ndarray:
    """将 [t, h, w, c] 的时间维统一到 T：不足补齐，过长均匀采样。"""
    t, h, w, c = x.shape
    if t == T:
        return x
    if t == 0:
        return np.zeros((T, h, w, c), dtype=x.dtype)
    if t > T:
        # 均匀选 T 帧
        idx = np.linspace(0, t - 1, T).round().astype(int)
        return x[idx]
    # t < T：末帧重复补齐
    pad = np.repeat(x[[-1]], T - t, axis=0)
    return np.concatenate([x, pad], axis=0)

def _resize_hw(x: np.ndarray, H: int, W: int) -> np.ndarray:
    """把每帧 resize 到 HxW。"""
    if x.shape[1] == H and x.shape[2] == W:
        return x
    out = np.empty((x.shape[0], H, W, x.shape[3]), dtype=x.dtype)
    for i in range(x.shape[0]):
        out[i] = cv2.resize(x[i], (W, H), interpolation=cv2.INTER_AREA)
    return out

class VPRemb:
    """VideoPrism wrapper with GPU 友好（固定形状+批量+JIT）并带 graceful fallback。"""

    def __init__(self, model_name: str = "videoprism_public_v1_base",
                 device: str = "cpu", threads: int | None = None, force_lite: bool = False):
        self.model_name = model_name
        self.device = (device or "cpu").lower()
        self.threads = threads
        self.force_lite = force_lite
        self.embed_dim = 768
        self._backend = None
        self._encode_fn = None     # JIT 后的批量函数
        self._init_backend()

    def _init_backend(self):
        logger = logging.getLogger(__name__)

        if self.force_lite:
            logger.info("VPR backend: forced lite embedding")
            self._backend = ("lite", None)
            self.embed_dim = 256
            return

        try:
            import jax
            import jax.numpy as jnp
            from jax import tree_util
            from videoprism import models as vp

            # 设备选择
            try:
                logger.info("JAX devices: %s", jax.devices())
            except Exception:
                pass
            all_devs = list(jax.devices())
            gpu_devs = [d for d in all_devs if getattr(d, "platform", "") == "gpu"]
            cpu_devs = [d for d in all_devs if getattr(d, "platform", "") == "cpu"]
            want_gpu = (self.device in ("gpu", "cuda")) or (self.device == "cpu" and len(gpu_devs) > 0)
            chosen = gpu_devs[0] if (want_gpu and gpu_devs) else (cpu_devs[0] if cpu_devs else (all_devs[0] if all_devs else None))
            if chosen is None:
                raise RuntimeError("No JAX devices available")
            logger.info("VPR using device: %s", chosen)

            # 加载模型与权重
            model = vp.get_model(self.model_name)
            params = vp.load_pretrained_weights(self.model_name)
            params = jax.device_put(params, chosen)

            # --- 批量 JIT 编码函数（固定 [B,T,H,W,C] 形状） ---
            def _apply(params, x_bthwc):
                # x_bthwc: [B,T,H,W,C] float32 in [0,1]
                out = model.apply(params, x_bthwc, train=False)
                leaves = tree_util.tree_leaves(out)
                emb = None
                for leaf in leaves:
                    if hasattr(leaf, "ndim") and getattr(leaf, "ndim", 0) >= 3:
                        emb = leaf  # 期望 [B,T,D]
                        break
                if emb is None:
                    raise ValueError("VideoPrism apply() did not return array-like embedding")
                if emb.ndim == 3:
                    emb = emb.mean(axis=1)  # [B,T,D] → [B,D]
                return emb  # [B,D]

            # 指定在 GPU 上编译；捐赠输入以减少拷贝
            self._encode_fn = jax.jit(_apply, donate_argnums=(1,), backend="gpu")
            # Warmup：编译一次（避免首批耗时 + tegrastats 长时间 0%）
            dummy = np.zeros((1, TARGET_T, TARGET_H, TARGET_W, 3), dtype=np.float32)
            _ = np.array(self._encode_fn(params, dummy))  # 触发编译

            self._backend = ("jax", (params, self._encode_fn))
            self.embed_dim = 768
            logger.info("VPR backend: JAX/VideoPrism (batched, jit, fixed-shape)")
        except Exception as e:
            logging.getLogger(__name__).warning(f"Falling back to lite embedding backend ({e})")
            self._backend = ("lite", None)
            self.embed_dim = 256

    def _prep_clip(self, x: np.ndarray) -> np.ndarray:
        """统一 clip 为 [TARGET_T, TARGET_H, TARGET_W, 3] float32 in [0,1]。"""
        if x.ndim == 3:
            x = x[None, ...]
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        if x.max() > 1.5:
            x = x / 255.0
        x = _sample_to_T(x, TARGET_T)
        x = _resize_hw(x, TARGET_H, TARGET_W)
        return x

    def _lite_embed(self, clip: np.ndarray) -> np.ndarray:
        # clip: [T,H,W,C] float32 0-1
        T, H, W, C = clip.shape
        x = clip
        feats = []
        feats.extend(x.mean(axis=(0, 1, 2)).tolist())
        feats.extend(x.std(axis=(0, 1, 2)).tolist())
        small = cv2.resize((x.mean(axis=0)), (64, 64))
        hsv = cv2.cvtColor((small * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        h_hist = np.histogram(hsv[:, :, 0], bins=16, range=(0, 255))[0]
        s_hist = np.histogram(hsv[:, :, 1], bins=16, range=(0, 255))[0]
        v_hist = np.histogram(hsv[:, :, 2], bins=16, range=(0, 255))[0]
        hist = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
        hist = hist / (np.linalg.norm(hist) + 1e-6)
        feats.extend(hist.tolist())
        diff = np.abs(np.diff(x.mean(axis=(2, 3)), axis=0)).mean()
        feats.append(float(diff))
        vec = np.array(feats, dtype=np.float32)
        D = 256
        if vec.shape[0] < D:
            vec = np.pad(vec, (0, D - vec.shape[0]))
        else:
            vec = vec[:D]
        return vec

    def encode_batch(self, clips: List[np.ndarray]) -> np.ndarray:
        """将一组 clip 编码为 [N,D]；内部自动做固定形状、批处理与容错。"""
        if not clips:
            return np.zeros((0, self.embed_dim), dtype=np.float32)

        backend, payload = self._backend
        out_vecs = []
        logger = logging.getLogger(__name__)

        if backend == "jax":
            params, encode_fn = payload
            batch = []
            for idx, c in enumerate(clips):
                try:
                    x = self._prep_clip(c)            # [T,H,W,C]
                    batch.append(x)
                    if len(batch) == BATCH_SIZE or idx == len(clips) - 1:
                        bx = np.stack(batch, axis=0)  # [B,T,H,W,C]
                        vecs = np.array(encode_fn(params, bx))  # [B,D]
                        if vecs.ndim != 2:
                            raise ValueError(f"unexpected VPR output shape {vecs.shape}")
                        out_vecs.append(vecs.astype(np.float32))
                        batch.clear()
                except Exception as e:
                    logger.warning(f"VPR JAX encode failed at idx={idx} ({e!r}); falling back to lite for this clip.")
                    # 单个失败回落到 lite，不影响整批
                    if 'x' not in locals():
                        x = self._prep_clip(c)
                    out_vecs.append(self._lite_embed(x)[None, ...].astype(np.float32))
            return np.concatenate(out_vecs, axis=0)

        # lite 路径（或 forced lite）
        for c in clips:
            x = self._prep_clip(c)    # 同样固定形状，便于后续对齐
            out_vecs.append(self._lite_embed(x).astype(np.float32))
        return np.stack(out_vecs, axis=0).astype(np.float32)