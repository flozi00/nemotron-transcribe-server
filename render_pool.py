"""Render a pool spec to the exact objects the console would apply.

The console's deploy_pool builds these from the same code; this renders them
offline so the objects can be reviewed (or applied by hand) without the
console. Pure: normalise_pool + build_pool do no cluster I/O.

    python3 render_pool.py <spec.json> <out.json>
"""

import json
import os
import sys

sys.path.insert(
    0,
    os.environ.get(
        "CONSOLE_API",
        os.path.expanduser("~/Documents/inference-console/api"),
    ),
)

import manifests  # noqa: E402


def main(argv):
    if len(argv) != 3:
        raise SystemExit(__doc__)
    spec = json.load(open(argv[1]))
    spec.pop("_comment", None)
    # The MCP tool takes env as a mapping; the console's validator wants the
    # Kubernetes shape ([{name, value}]), which is what deploy_pool sends.
    if isinstance(spec.get("env"), dict):
        spec["env"] = [{"name": k, "value": v} for k, v in spec["env"].items()]
    # Same for gpu_selectors: the tool takes card names, the validator wants
    # {key, value} pairs on the GPU product label.
    if spec.get("gpu_selectors") and isinstance(spec["gpu_selectors"][0], str):
        spec["gpu_selectors"] = [
            {"key": "nvidia.com/gpu.product", "value": name}
            for name in spec["gpu_selectors"]
        ]
    norm = manifests.normalise_pool(spec)
    objects = [obj for obj in manifests.build_pool(norm, created_by="render") if obj]
    payload = {"apiVersion": "v1", "kind": "List", "items": objects}
    with open(argv[2], "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
    print("wrote %d objects to %s" % (len(objects), argv[2]))
    for obj in objects:
        print("  %-14s %s" % (obj["kind"], obj["metadata"]["name"]))
    for warning in norm.get("warnings") or []:
        print("WARN:", warning)


if __name__ == "__main__":
    main(sys.argv)
