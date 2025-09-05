from typing import List
import numpy as np
import logging
import os

# ---------------- 在导入 JAX 之前，设置 XLA/JAX 环境（兼容 jaxlib==0.5.0） ----------------
def _set_xla_flags(flags: list[str]):
    # 用支持的子集覆盖，避免旧版 XLA 因未知 flag 直接 FATAL
    dedup = []
    for f in flags:
        if f and f not in dedup:
            dedup.append(f)
    os.environ["XLA_FLAGS"] = " ".join(dedup).strip()

# 仅保留 0.5.0 兼容的 flag
_set_xla_flags([
    "--xla_gpu_cuda_data_dir=/usr/local/cuda",
    "--xla_gpu_autotune_level=0",   # 关闭 GEMM autotune，规避不兼容 kernel
])

# 其他建议设置（可减少显存占用和噪声）
os.environ.setdefault("JAX_PLATFORMS", "cuda")                  # 不去探测 rocm/tpu
os.environ.setdefault("JAX_ENABLE_X64", "0")                    # 省显存/更快
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75") # 预分配 75% 显存
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------

# ---------------- JAX debug_info 兼容层（JAX>=0.5.0 已移除 api_util.debug_info）----------------
# 有些老代码或三方库会引用 jax.api_util.debug_info，这里注入一个 no-op 以避免 AttributeError
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
    # 如果连 jax 都没有，下面初始化时会自动走 lite
    pass
# ---------------------------------------------------------------------------------------------

class VPRemb:
    """VideoPrism wrapper with graceful fallback.
    It attempts to import DeepMind's videoprism+jax; if not available,
    falls back to a light-weight embedding so the pipeline remains runnable.
    """
    def __init__(self, model_name: str = "videoprism_public_v1_base",
                 device: str = "cpu", threads: int | None = None, force_lite: bool = False):
        self.model_name = model_name
        self.device = (device or "cpu").lower()
        self.threads = threads
        self.force_lite = force_lite
        # 先给个默认值，实际在 _init_backend() 里按后端覆盖
        self.embed_dim = 768
        self._backend = None
        self._init_backend()

    def _init_backend(self):
        logger = logging.getLogger(__name__)

        # Optional: force lite backend via flag
        if self.force_lite:
            logger.info("VPR backend: forced lite embedding")
            self._backend = ("lite", None)
            self.embed_dim = 256
            return

        try:
            import jax
            from jax import tree_util
            from videoprism import models as vp

            # 打印可用设备，便于诊断
            try:
                logger.info("JAX devices: %s", jax.devices())
            except Exception:
                pass

            # 设备选择策略：
            # - 若用户传入 "gpu"/"cuda" 则尽量选 GPU
            # - 若用户传入 "cpu"，但系统有 GPU，则优先用 GPU（更快）
            # - 兜底选第一个设备
            all_devs = []
            try:
                all_devs = list(jax.devices())
            except Exception:
                pass
            gpu_devs = [d for d in all_devs if getattr(d, "platform", "") == "gpu"]
            cpu_devs = [d for d in all_devs if getattr(d, "platform", "") == "cpu"]

            want_gpu = (self.device in ("gpu", "cuda")) or (self.device == "cpu" and len(gpu_devs) > 0)
            if want_gpu and gpu_devs:
                chosen = gpu_devs[0]
            elif self.device == "cpu" and cpu_devs:
                chosen = cpu_devs[0]
            else:
                chosen = all_devs[0] if all_devs else None

            if chosen is None:
                raise RuntimeError("No JAX devices available")

            logger.info("VPR using device: %s", chosen)

            # 加载模型与权重
            model = vp.get_model(self.model_name)
            params = vp.load_pretrained_weights(self.model_name)

            # 将参数放到选定设备
            params = jax.device_put(params, chosen)

            # JAX 前端编码函数
            def encode_jax(inputs):
                outputs = model.apply(params, inputs, train=False)
                # Flatten any pytree (tuple/list/dict/Module outputs)
                leaves = tree_util.tree_leaves(outputs)
                emb = None
                for leaf in leaves:
                    if hasattr(leaf, "ndim") and getattr(leaf, "ndim", 0) >= 2:
                        emb = leaf
                        break
                if emb is None:
                    raise ValueError("VideoPrism apply() did not return an array-like embedding")
                # If [B,T,D], average over T → [B,D]
                if getattr(emb, "ndim", 0) == 3:
                    emb = emb.mean(axis=1)
                arr = np.array(emb, dtype=np.float32)
                return arr

            self._backend = ("jax", encode_jax)
            self.embed_dim = 768  # VPR-base 的常见维度
            logger.info("VPR backend: JAX/VideoPrism")
        except Exception as e:
            logger.warning(f"Falling back to lite embedding backend ({e})")
            self._backend = ("lite", None)
            self.embed_dim = 256

    def _lite_embed(self, clip: np.ndarray) -> np.ndarray:
        # clip: [T,H,W,C] float32 0-1
        # features: mean/std per channel + HSV hist + temporal diff energy
        T, H, W, C = clip.shape
        x = clip
        feats = []
        # RGB mean/std
        feats.extend(x.mean(axis=(0, 1, 2)).tolist())
        feats.extend(x.std(axis=(0, 1, 2)).tolist())
        # downsample to 64x64 for hist
        import cv2
        small = cv2.resize((x.mean(axis=0)), (64, 64))
        hsv = cv2.cvtColor((small * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        h_hist = np.histogram(hsv[:, :, 0], bins=16, range=(0, 255))[0]
        s_hist = np.histogram(hsv[:, :, 1], bins=16, range=(0, 255))[0]
        v_hist = np.histogram(hsv[:, :, 2], bins=16, range=(0, 255))[0]
        hist = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
        hist = hist / (np.linalg.norm(hist) + 1e-6)
        feats.extend(hist.tolist())
        # temporal energy
        diff = np.abs(np.diff(x.mean(axis=(2, 3)), axis=0)).mean()
        feats.append(float(diff))
        vec = np.array(feats, dtype=np.float32)
        # pad/trim 到 lite 维度（256）
        D = 256
        if vec.shape[0] < D:
            vec = np.pad(vec, (0, D - vec.shape[0]))
        else:
            vec = vec[:D]
        return vec

    def encode_batch(self, clips: List[np.ndarray]) -> np.ndarray:
        """Encode a list of time clips. To maximize stability on macOS/CPU,
        we avoid big-batch execution and instead run one clip per call.
        This does not change numerical results relative to a single big batch.
        Each input may be [H,W,C] or [T,H,W,C]; values will be coerced to float32 in [0,1].
        Returns: np.ndarray of shape [N, D].
        """
        if not clips:
            return np.zeros((0, self.embed_dim), dtype=np.float32)

        backend, fn = self._backend
        out_vecs = []
        logger = logging.getLogger(__name__)

        for idx, c in enumerate(clips):
            # --- normalize to [T,H,W,C] float32 in [0,1] ---
            x = c
            if x.ndim == 3:
                x = x[None, ...]
            if x.dtype != np.float32:
                x = x.astype(np.float32)
            if x.max() > 1.5:
                x = x / 255.0

            # build a per-clip batch [1,T,H,W,C]
            bx = x[None, ...]

            if backend == "jax":
                try:
                    vec = fn(bx)  # expected [1,D] or [B,D]; wrapper已对 T 做平均
                    if hasattr(vec, "ndim") and vec.ndim == 2 and vec.shape[0] == 1:
                        vec = vec[0]
                    vec = np.asarray(vec, dtype=np.float32)
                    if vec.ndim != 1:
                        raise ValueError(f"unexpected VPR output shape {vec.shape} for clip {idx}")
                    # 若偶发返回维度和预期不一致，做一次安全裁切/填充到 self.embed_dim
                    if vec.shape[0] != self.embed_dim:
                        if vec.shape[0] > self.embed_dim:
                            vec = vec[:self.embed_dim]
                        else:
                            vec = np.pad(vec, (0, self.embed_dim - vec.shape[0]))
                    out_vecs.append(vec)
                    continue
                except Exception as e:
                    logger.warning(f"VPR JAX per-clip encode failed at idx={idx} ({e!r}); falling back to lite.")
                    # 一旦当前 clip 失败，继续尝试 lite；不改变后续 clip 的策略

            # lite fallback (or forced lite backend)
            vec = self._lite_embed(x)
            # 确保维度与当前后端一致
            if vec.shape[0] != self.embed_dim:
                if vec.shape[0] > self.embed_dim:
                    vec = vec[:self.embed_dim]
                else:
                    vec = np.pad(vec, (0, self.embed_dim - vec.shape[0]))
            out_vecs.append(vec.astype(np.float32))

        return np.stack(out_vecs, axis=0).astype(np.float32)