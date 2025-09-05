from typing import List
import numpy as np
import logging
import os
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
os.environ.setdefault("JAX_PLATFORMS", "cuda")                  # 只用 CUDA，避免 rocm/tpu 噪声
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
        idx = np.linspace(0, t - 1, T).round().astype(int)  # 均匀取样
        return x[idx]
    pad = np.repeat(x[[-1]], T - t, axis=0)  # 末帧补齐
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
    """VideoPrism wrapper（固定形状+批量+JIT），**严格 GPU**，无任何 CPU/lite 回退。"""

    def __init__(self, model_name: str = "videoprism_public_v1_base",
                 device: str = "cuda", threads: int | None = None):
        self.model_name = model_name
        self.device = (device or "cuda").lower()
        self.threads = threads
        self.embed_dim = 768
        self._encode_fn = None     # JIT 后的批量函数
        self._params = None
        self._init_backend()

    def _init_backend(self):
        logger = logging.getLogger(__name__)
        try:
            import jax
            import jax.numpy as jnp
            from jax import tree_util
            from videoprism import models as vp

            # 打印设备并**强制**选择 GPU
            devs = list(jax.devices())
            logger.info("JAX devices: %s", devs)
            gpu_devs = [d for d in devs if getattr(d, "platform", "") == "gpu"]
            if not gpu_devs:
                raise RuntimeError("未检测到 JAX GPU 设备；请检查 jax/jaxlib/cuda/cuDNN 安装与 JAX_PLATFORMS=cuda")

            chosen = gpu_devs[0]
            logger.info("VPR using device: %s", chosen)

            # 加载模型与权重
            model = vp.get_model(self.model_name)
            params = vp.load_pretrained_weights(self.model_name)
            params = jax.device_put(params, chosen)
            self._params = params

            # --- 批量 JIT 编码函数（固定 [B,T,H,W,C] 形状），**无回退** ---
            def _apply(p, x_bthwc):
                # x_bthwc: [B,T,H,W,C] float32 in [0,1]
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

            self._encode_fn = jax.jit(_apply, donate_argnums=(1,), backend="gpu")

            # Warmup：编译一次；失败就直接抛错（不再回退）
            dummy = np.zeros((1, TARGET_T, TARGET_H, TARGET_W, 3), dtype=np.float32)
            _ = np.array(self._encode_fn(self._params, dummy))

            logger.info("VPR backend: JAX/VideoPrism (batched, jit, fixed-shape, STRICT GPU)")
        except Exception as e:
            # 直接失败，抛出清晰错误信息
            raise RuntimeError(
                f"VPR GPU 后端初始化失败：{e}\n"
                "请检查：\n"
                "- jax/jaxlib 版本是否为 0.5.0 且加载到 CudaDevice\n"
                "- CUDA/cuDNN 是否匹配（XLA_FLAGS 已设置 cuda_data_dir）\n"
                "- 形状是否固定（TARGET_T/H/W），批量是否过大（BATCH_SIZE）\n"
                "- 首次编译报错可尝试降低 RECMATCHER_VPR_T/BATCH_SIZE"
            ) from e

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

    def encode_batch(self, clips: List[np.ndarray]) -> np.ndarray:
        """将一组 clip 编码为 [N,D]；固定形状+批量+JIT，**任何异常直接抛出**。"""
        if not clips:
            return np.zeros((0, self.embed_dim), dtype=np.float32)

        if self._encode_fn is None or self._params is None:
            raise RuntimeError("VPR GPU 后端未正确初始化")

        import jax  # 局部导入便于报错时打印信息
        logger = logging.getLogger(__name__)

        # 批量组织
        out_vecs = []
        batch = []
        for idx, c in enumerate(clips):
            x = self._prep_clip(c)          # [T,H,W,C]
            batch.append(x)
            if len(batch) == BATCH_SIZE or idx == len(clips) - 1:
                bx = np.stack(batch, axis=0)  # [B,T,H,W,C]
                try:
                    vecs = np.array(self._encode_fn(self._params, bx))  # [B,D]
                except Exception as e:
                    # 打印更丰富的上下文再抛出
                    logger.error(
                        "VPR GPU 编码失败：batch_size=%d, shape=%s, devices=%s",
                        len(batch), bx.shape, jax.devices()
                    )
                    raise
                if vecs.ndim != 2:
                    raise ValueError(f"VPR 输出维度异常：{vecs.shape}（期望 [B,D]）")
                out_vecs.append(vecs.astype(np.float32))
                batch.clear()

        return np.concatenate(out_vecs, axis=0)