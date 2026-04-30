"""
Compare LocalSunFlare performance between albumentations versions.
Current config: flare_roi=(0, 0, 1, 1), src_radius=400, num_flare_circles_range=(1, 2), p=1.0
"""
import time
import numpy as np
from PIL import Image

IMG_PATH = "visuals/000000039769.jpg"


def make_test_image(size=(1200, 1920)):
    img = Image.open(IMG_PATH).convert("RGB").resize((size[1], size[0]))
    return np.array(img)


def test_version(version_name, conda_env):
    """Run in subprocess to isolate albumentations import."""
    import subprocess, json

    code = '''
import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import time, json, numpy as np
from PIL import Image
import albumentations as A

IMG_PATH = "visuals/000000039769.jpg"
np.random.seed(42)

img = np.array(Image.open(IMG_PATH).convert("RGB").resize((1920, 1200)))

params = dict(flare_roi=(0, 0, 1, 1), src_radius=400,
              num_flare_circles_range=(1, 2), p=1.0)

sunflare = A.RandomSunFlare(**params)

# Warmup
for _ in range(5):
    _ = sunflare(image=img)["image"]

# Benchmark
times = []
for _ in range(50):
    t0 = time.perf_counter()
    _ = sunflare(image=img)["image"]
    times.append(time.perf_counter() - t0)

print(json.dumps({
    "version": A.__version__,
    "mean_ms": np.mean(times) * 1000,
    "std_ms": np.std(times) * 1000,
    "min_ms": np.min(times) * 1000,
    "max_ms": np.max(times) * 1000,
}))
'''

    result = subprocess.run(
        ["bash", "-c", f"source ~/miniconda3/etc/profile.d/conda.sh && conda activate {conda_env} && python -c '{code}'"],
        capture_output=True, text=True, timeout=120, cwd="/opt/dl_workspace/algorithm/04-myself/domaingap"
    )

    if result.returncode != 0:
        print(f"[{version_name}] ERROR: {result.stderr}")
        return None

    # Extract JSON from output (may have warnings)
    for line in result.stdout.strip().split('\n'):
        line = line.strip()
        if line.startswith('{'):
            return json.loads(line)

    print(f"[{version_name}] No valid output: {result.stdout}")
    return None


if __name__ == "__main__":
    print("=== LOCAL SunFlare Benchmark (1920x1200, 50 iterations) ===\n")

    r1 = test_version("py38 (1.4.18)", "py38")
    r2 = test_version("dinov3 (2.0.8)", "dinov3")

    print(f"{'':<25} {'py38 (1.4.18)':<20} {'dinov3 (2.0.8)':<20} {'ratio':<10}")
    print("-" * 75)

    for metric in ["mean_ms", "std_ms", "min_ms", "max_ms"]:
        if r1 and r2:
            v1 = r1[metric]
            v2 = r2[metric]
            ratio = v2 / v1 if v1 > 0 else 0
            label = metric.replace("_ms", " (ms)")
            print(f"{label:<25} {v1:<20.2f} {v2:<20.2f} {ratio:<10.2f}x")

    if r1 and r2:
        print(f"\n结论: dinov3 比 py38 {'快' if r2['mean_ms'] < r1['mean_ms'] else '慢'} {abs(1 - r2['mean_ms']/r1['mean_ms'])*100:.1f}%")
