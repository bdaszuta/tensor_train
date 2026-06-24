"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Demonstration: compress a random function with from_dense
"""
import sys
import os as _os

_sys_dir = _os.path.dirname(_os.path.abspath(__file__))
_bind_dir = _os.path.join(_sys_dir, "..", "..", "bind_py")
if _bind_dir not in sys.path:
    sys.path.insert(0, _bind_dir)

import numpy as np
from tensor_train import from_dense, TensorTrain

# -------------------------------------------------------------------
# 1. Build a dense tensor and compress it
# -------------------------------------------------------------------
shape = [4, 5, 6, 3]
total = int(np.prod(shape))
dense = np.sin(np.arange(1, total + 1, dtype=np.float64) / total * 2.0 * np.pi)

print(f"Original: {total} values, shape {shape}")

tt = from_dense(dense, shape, eps=1e-8)
print(f"Compressed: {tt.num_params} parameters, max_rank={tt.max_rank}")
print(f"Ratio: {total / tt.num_params:.1f}x")

# -------------------------------------------------------------------
# 2. Extract cores (zero-copy views into the TT)
# -------------------------------------------------------------------
core_views = tt.cores
for k, arr in enumerate(core_views):
    print(f"  core[{k}]: shape={arr.shape}  "
          f"(r_left={arr.shape[0]}, n={arr.shape[1]}, r_right={arr.shape[2]})")

# -------------------------------------------------------------------
# 3. Save to disk
# -------------------------------------------------------------------
np.savez("demo_compressed.npz",
         **{f"c{k}": arr for k, arr in enumerate(core_views)},
         shape=np.array(shape, dtype=np.int32))
print("\nSaved to demo_compressed.npz")

# -------------------------------------------------------------------
# 4. Load from disk
# -------------------------------------------------------------------
loaded = np.load("demo_compressed.npz")
shape_loaded = list(loaded["shape"])
cores_loaded = [loaded[f"c{k}"] for k in range(len(shape_loaded))]

tt2 = TensorTrain.from_cores(cores_loaded)

# -------------------------------------------------------------------
# 5. Verify reconstruction
# -------------------------------------------------------------------
recon = tt2.to_dense()
err = np.linalg.norm(recon - dense) / np.linalg.norm(dense)
print(f"\nReconstruction error: {err:.2e}")
print(f"Shapes match: {tt2.shape == shape}")
print(f"Ranks match:  {tt2.ranks == tt.ranks}")

# -------------------------------------------------------------------
# 6. Point evaluation
# -------------------------------------------------------------------
idx = [2, 3, 1, 0]
val = tt2.eval_at(idx)
linear = ((idx[0] * shape[1] + idx[1]) * shape[2] + idx[2]) * shape[3] + idx[3]
print(f"\neval_at({idx}) = {val:.10f}")
print(f"dense[{linear}]    = {dense[linear]:.10f}")
print(f"match: {abs(val - dense[linear]) < 1e-12}")

# Cleanup
import os as _os2
_os2.unlink("demo_compressed.npz")
print("\nDone.")
#
# :D
#
