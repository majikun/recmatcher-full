from typing import List
import numpy as np
import logging

class VPRemb:
    """VideoPrism wrapper with graceful fallback.
    It attempts to import DeepMind's videoprism+jax; if not available,
    falls back to a light-weight embedding so the pipeline remains runnable.
    """
    def __init__(self, model_name: str = "videoprism_public_v1_base", device: str = "cpu", threads: int|None = None, force_lite: bool = False):
        self.model_name = model_name
        self.device = device
        self.threads = threads
        self.force_lite = force_lite
        self.embed_dim = 768
        self._backend = None
        self._init_backend()

    def _init_backend(self):
        # Optional: force lite backend via flag
        if self.force_lite:
            logging.getLogger(__name__).info("VPR backend: forced lite embedding")
            self._backend = ("lite", None)
            return
        try:
            import jax
            from jax import tree_util
            from videoprism import models as vp
            model = vp.get_model(self.model_name)
            params = vp.load_pretrained_weights(self.model_name)
            # Select device (CPU on Mac by default)
            devs = jax.devices(self.device)
            device = devs[0] if devs else jax.devices()[0]
            params = jax.device_put(params, device)

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
                return np.array(emb, dtype=np.float32)

            self._backend = ("jax", encode_jax)
            logging.getLogger(__name__).info("VPR backend: JAX/VideoPrism")
        except Exception as e:
            logging.getLogger(__name__).warning(f"Falling back to lite embedding backend ({e})")
            self._backend = ("lite", None)

    def _lite_embed(self, clip: np.ndarray) -> np.ndarray:
        # clip: [T,H,W,C] float32 0-1
        # features: mean/std per channel + HSV hist + temporal diff energy
        T,H,W,C = clip.shape
        x = clip
        feats = []
        # RGB mean/std
        feats.extend(x.mean(axis=(0,1,2)).tolist())
        feats.extend(x.std(axis=(0,1,2)).tolist())
        # downsample to 64x64 for hist
        import cv2
        small = cv2.resize((x.mean(axis=0)), (64,64))
        hsv = cv2.cvtColor((small*255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        h_hist = np.histogram(hsv[:,:,0], bins=16, range=(0,255))[0]
        s_hist = np.histogram(hsv[:,:,1], bins=16, range=(0,255))[0]
        v_hist = np.histogram(hsv[:,:,2], bins=16, range=(0,255))[0]
        hist = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
        hist = hist / (np.linalg.norm(hist) + 1e-6)
        feats.extend(hist.tolist())
        # temporal energy
        diff = np.abs(np.diff(x.mean(axis=(2,3)), axis=0)).mean()
        feats.append(float(diff))
        vec = np.array(feats, dtype=np.float32)
        # pad/trim to 256 dims
        D = 256
        if vec.shape[0] < D:
            vec = np.pad(vec, (0, D - vec.shape[0]))
        else:
            vec = vec[:D]
        return vec

    def encode_batch(self, clips: List[np.ndarray]) -> np.ndarray:
        if not clips:
            return np.zeros((0, self.embed_dim), dtype=np.float32)
        normed = []
        maxT = 0
        for c in clips:
            x = c
            if x.ndim == 3:
                x = x[None, ...]
            if x.dtype != np.float32:
                x = x.astype(np.float32)
            if x.max() > 1.5:
                x = x / 255.0
            normed.append(x)
            if x.shape[0] > maxT:
                maxT = x.shape[0]
        pads = []
        for c in normed:
            if c.shape[0] < maxT:
                k = maxT - c.shape[0]
                pad = np.concatenate([c, np.repeat(c[-1:], k, axis=0)], axis=0)
            else:
                pad = c
            pads.append(pad)
        arr = np.stack(pads, axis=0).astype(np.float32)
        backend, fn = self._backend
        if backend == "jax":
            try:
                out = fn(arr)
                return out.astype(np.float32)
            except Exception as e:
                logging.getLogger(__name__).warning(f"VPR JAX encode failed ({e}); falling back to lite embedding for this batch.")
                vecs = [self._lite_embed(c) for c in clips]
                return np.stack(vecs, axis=0).astype(np.float32)
        else:
            vecs = [self._lite_embed(c) for c in clips]
            return np.stack(vecs, axis=0).astype(np.float32)
