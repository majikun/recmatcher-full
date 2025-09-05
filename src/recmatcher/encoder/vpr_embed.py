from typing import List
import numpy as np
import logging
import os
import cv2  # Jetson 自带

# ---------------- 在导入 JAX 之前，设置 XLA/JAX 环境（兼容 jaxlib==0.5.0） ----------------
def _set_xla_flags(flags: list[str]):
    dedup = []
    for f in flags:
        if f and f not in dedup:
            dedup.append(f)
    os.environ["XLA_FLAGS"] = " ".join(dedup).strip()

_set_xla_flags([
    "--xla_gpu_cuda_data_dir=/usr/local/cuda",
    "--xla_gpu_autotune_level=0",
])

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("JAX_ENABLE_X64", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.70")
# ---------------- 固定形状/批量参数（可用环境变量覆盖） ----------------
TARGET_T = int(os.getenv("RECMATCHER_VPR_T", "8"))      # 每段帧数
TARGET_S = int(os.getenv("RECMATCHER_VPR_S", "288"))    # 方形边长 SxS
BATCH_SIZE = int(os.getenv("RECMATCHER_VPR_BS", "4"))   # 批量
PREPROC_WORKERS = int(os.getenv("RECMATCHER_VPR_PREPROC_WORKERS", "4"))  # 预处理线程数
SQUARE_MODE = os.getenv("RECMATCHER_VPR_SQUARE_MODE", "pad").lower()  # pad | crop
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
TARGET_T = int(os.getenv("RECMATCHER_VPR_T", "8"))      # 每段帧数
TARGET_S = int(os.getenv("RECMATCHER_VPR_S", "288"))    # 方形边长 SxS
BATCH_SIZE = int(os.getenv("RECMATCHER_VPR_BS", "4"))   # 批量
SQUARE_MODE = os.getenv("RECMATCHER_VPR_SQUARE_MODE", "pad").lower()  # pad | crop
if SQUARE_MODE not in ("pad", "crop"):
    SQUARE_MODE = "pad"
# ---------------------------------------------------------------------

def _sample_to_T(x: np.ndarray, T: int) -> np.ndarray:
    """将 [t,h,w,c] 的时间维统一到 T：不足补齐，过长均匀采样。"""
    t, h, w, c = x.shape
    if t == T:
        return x
    if t == 0:
        return np.zeros((T, h, w, c), dtype=x.dtype)
    if t > T:
        idx = np.linspace(0, t - 1, T).round().astype(int)
        return x[idx]
    pad = np.repeat(x[[-1]], T - t, axis=0)
    return np.concatenate([x, pad], axis=0)

def _resize_to_square(x: np.ndarray, S: int, mode: str = "pad") -> np.ndarray:
    """将每帧变换为 SxS：mode='pad' 先等比缩放再居中填充；mode='crop' 先等比放大后中心裁剪。"""
    t, h, w, c = x.shape
    if h == S and w == S:
        return x
    out = np.empty((t, S, S, c), dtype=x.dtype)
    if mode == "pad":
        # 等比缩放到不超过 S，再四周补黑边
        scale = min(S / h, S / w)
        nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        top = (S - nh) // 2
        left = (S - nw) // 2
        for i in range(t):
            resized = cv2.resize(x[i], (nw, nh), interpolation=cv2.INTER_AREA)
            frame = np.zeros((S, S, c), dtype=x.dtype)
            frame[top:top+nh, left:left+nw] = resized
            out[i] = frame
        return out
    else:
        # 等比放大到覆盖 S×S，再中心裁剪
        scale = max(S / h, S / w)
        nh, nw = max(S, int(round(h * scale))), max(S, int(round(w * scale)))
        for i in range(t):
            resized = cv2.resize(x[i], (nw, nh), interpolation=cv2.INTER_AREA)
            top = (nh - S) // 2
            left = (nw - S) // 2
            out[i] = resized[top:top+S, left:left+S]
        return out

class VPRemb:
    """VideoPrism wrapper（固定形状+方形化+批量+JIT），严格 GPU，无回退。"""

    def __init__(self, model_name: str = "videoprism_public_v1_base",
                 device: str = "cuda", threads: int | None = None):
        self.model_name = model_name
        self.device = (device or "cuda").lower()
        self.threads = threads
        self.embed_dim = 768
        self._encode_fn = None
        self._params = None
        self._init_backend()

    def _init_backend(self):
        logger = logging.getLogger(__name__)
        try:
            import jax
            import jax.numpy as jnp
            from jax import tree_util
            from videoprism import models as vp

            devs = list(jax.devices())
            logger.info("JAX devices: %s", devs)
            gpu_devs = [d for d in devs if getattr(d, "platform", "") == "gpu"]
            if not gpu_devs:
                raise RuntimeError("未检测到 JAX GPU 设备；请检查 jax/jaxlib/cuda/cuDNN 与 JAX_PLATFORMS=cuda")

            chosen = gpu_devs[0]
            logger.info("VPR using device: %s", chosen)

            model = vp.get_model(self.model_name)
            params = vp.load_pretrained_weights(self.model_name)
            params = jax.device_put(params, chosen)
            self._params = params

            def _apply(p, x_bthwc):
                out = model.apply(p, x_bthwc, train=False)
                leaves = tree_util.tree_leaves(out)
                emb = None
                for leaf in leaves:
                    if hasattr(leaf, "ndim") and getattr(leaf, "ndim", 0) >= 3:
                        emb = leaf  # 期望 [B,T,D]
                        break
                if emb is None:
                    raise ValueError("VideoPrism apply() 未返回数组型 embedding")
                if emb.ndim == 3:
                    emb = emb.mean(axis=1)  # [B,T,D] → [B,D]
                return emb  # [B,D]

            self._encode_fn = jax.jit(_apply, backend="gpu")

            # Warmup：注意要用 **方形** 输入，避免 encoders.py 里 assert h == w
            dummy = np.zeros((1, TARGET_T, TARGET_S, TARGET_S, 3), dtype=np.float32)
            _ = np.array(self._encode_fn(self._params, dummy))

            logger.info("VPR backend: JAX/VideoPrism (batched, jit, square=%dx%d, STRICT GPU)", TARGET_S, TARGET_S)
        except Exception as e:
            raise RuntimeError(
                f"VPR GPU 后端初始化失败：{e}\n"
                "请检查：\n"
                "- jax/jaxlib 是否为 0.5.0 且设备为 CudaDevice\n"
                "- CUDA/cuDNN 是否匹配\n"
                "- 输入是否已正方形（RECMATCHER_VPR_S，SQUARE_MODE=pad/crop）\n"
                "- 批量/帧数是否过大（RECMATCHER_VPR_BS / RECMATCHER_VPR_T）"
            ) from e

    def _prep_clip(self, x: np.ndarray) -> np.ndarray:
        """统一为 [T,S,S,3] float32 in [0,1]（先采样/补齐，再方形化）。"""
        if x.ndim == 3:
            x = x[None, ...]
        if x.dtype != np.float32:
            x = x.astype(np.float32)
        if x.max() > 1.5:
            x = x / 255.0
        x = _sample_to_T(x, TARGET_T)
        x = _resize_to_square(x, TARGET_S, SQUARE_MODE)
        return x

def encode_batch(self, clips: List[np.ndarray]) -> np.ndarray:
    """将一组 clip 编码为 [N,D]；固定方形+批量+JIT，任何异常直接抛出。"""
    if not clips:
        return np.zeros((0, self.embed_dim), dtype=np.float32)
    if self._encode_fn is None or self._params is None:
        raise RuntimeError("VPR GPU 后端未正确初始化")

    import jax, time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    logger = logging.getLogger(__name__)

    all_vecs = []
    # 分批处理，避免一次性占用太多内存
    for start in range(0, len(clips), BATCH_SIZE):
        chunk = clips[start:start + BATCH_SIZE]

        # 1) CPU 预处理（并行）
        t0 = time.time()
        if PREPROC_WORKERS <= 1:
            prepped = [self._prep_clip(c) for c in chunk]
        else:
            with ThreadPoolExecutor(max_workers=PREPROC_WORKERS) as ex:
                prepped = list(ex.map(self._prep_clip, chunk))
        prep_ms = (time.time() - t0) * 1000.0

        bx = np.stack(prepped, axis=0)  # [B,T,S,S,3]

        # 2) GPU 编码
        t1 = time.time()
        try:
            vecs = np.array(self._encode_fn(self._params, bx))  # [B,D]
        except Exception:
            logger.error(
                "VPR GPU 编码失败：batch_size=%d, shape=%s, devices=%s",
                len(chunk), bx.shape, jax.devices()
            )
            raise
        gpu_ms = (time.time() - t1) * 1000.0

        if vecs.ndim != 2:
            raise ValueError(f"VPR 输出维度异常：{vecs.shape}（期望 [B,D]）")
        all_vecs.append(vecs.astype(np.float32))

        # 每隔若干批打印一次：哪边在吃时间一眼就看出来
        if (start // BATCH_SIZE) % 20 == 0:
            logger.info("VPR batch%4d: prep=%.1f ms  gpu=%.1f ms  (B=%d, T=%d, S=%d)",
                        start // BATCH_SIZE, prep_ms, gpu_ms, len(chunk), TARGET_T, TARGET_S)

    return np.concatenate(all_vecs, axis=0)