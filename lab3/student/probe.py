# 探测不同 block_DV / threads 的 shared mem 和 launch_bounds
import sys
sys.path.insert(0, ".")
import re
from student.tilelang_fwd import _gdn_naive_kernel

for bdv in [32, 64, 128]:
    for thr in [128, 256]:
        try:
            k = _gdn_naive_kernel(1, 2048, 16, 16, 128, 128, bdv, thr, 2)
            src = k.get_kernel_source()
            lb = re.findall(r"__launch_bounds__\s*\(\s*(\d+),\s*(\d+)\)", src)
            smem = re.findall(r"extern\s+__shared__.*?(\d+)\s*\[", src[:3000])
            # 找 dynamic shared memory size
            dyn = re.findall(r"shared_memory_size\s*=\s*(\d+)", src)
            print(f"bdv={bdv} thr={thr}: launch_bounds={lb} dyn_smem={dyn[:3]} srclen={len(src)}")
        except Exception as e:
            print(f"bdv={bdv} thr={thr}: ERROR {type(e).__name__}: {str(e)[:120]}")
