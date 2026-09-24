#!/usr/bin/env python3
"""Offline artifact audit for recovered embedding candidates; no inference.

This is isolated experiment tooling, never a product runtime dependency.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import re

CATALOG = Path(__file__).with_name("models.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def load_candidate(name):
    catalog = json.loads(CATALOG.read_bytes())
    require(catalog["schema"] == 1 and catalog["admission"] == "candidate-only", "invalid candidate catalog")
    candidate = catalog["models"][name]
    require(candidate["pooling"] in ("masked-mean", "cls") and candidate["normalize"] == "l2",
            "unsupported pooling contract")
    return candidate


def verify_files(directory, candidate):
    identities = {}
    for name, expected in candidate["files"].items():
        require(Path(name).name == name, "artifact names must be local filenames")
        path = directory / name
        before = path.lstat()
        require(not path.is_symlink() and path.is_file() and before.st_size == expected["bytes"],
                f"{name}: wrong size or file type")
        sha = hashlib.sha256()
        blob = hashlib.sha1(f"blob {before.st_size}\0".encode())
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                sha.update(chunk)
                blob.update(chunk)
        after = path.lstat()
        require((before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
                (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                f"{name}: changed during verification")
        actual = {"bytes": after.st_size, "sha256": sha.hexdigest(), "git_blob_sha1": blob.hexdigest()}
        for field in ("sha256", "git_blob_sha1"):
            if field in expected:
                require(actual[field] == expected[field], f"{name}: {field} differs from pinned upstream artifact")
        identities[name] = actual
    return identities


def external_weight_contract(model, files):
    # Inspect every tensor, including nested graph/attribute tensors. Never let
    # a native loader resolve unverified paths named inside the graph.
    def tensors(message):
        if message.DESCRIPTOR.full_name == "onnx.TensorProto":
            yield message
        for field, value in message.ListFields():
            if field.type == field.TYPE_MESSAGE:
                for child in value if field.is_repeated else (value,):
                    yield from tensors(child)

    locations = set()
    count = 0
    for tensor in tensors(model):
        if not tensor.external_data:
            require(tensor.data_location != 1, "external tensor has no weight location")
            continue
        require(tensor.data_location == 1, "external weights lack EXTERNAL data location")
        metadata = {item.key: item.value for item in tensor.external_data}
        require(len(metadata) == len(tensor.external_data), "duplicate external weight metadata")
        require(set(metadata) <= {"location", "offset", "length", "checksum"},
                "unsupported external weight metadata")
        location = metadata.get("location", "")
        require(location and Path(location).name == location and "\\" not in location
                and location not in ("model.onnx", "tokenizer.json") and location in files,
                "external weights must name a verified local artifact")
        require("sha256" in files[location], "external weights require a pinned SHA-256")
        size = files[location]["bytes"]
        offset = metadata.get("offset", "0")
        length = metadata.get("length", str(size))
        require(re.fullmatch(r"[0-9]+", offset) and re.fullmatch(r"[0-9]+", length),
                "invalid external weight range")
        offset = int(offset)
        length = int(length) if "length" in metadata else size - offset
        require(0 <= offset < size and 0 < length <= size - offset,
                "external weight range exceeds verified artifact")
        locations.add(location)
        count += 1
    return {"files": sorted(locations), "tensors": count}


def graph_contract(path, files):
    import onnx
    model = onnx.load(path, load_external_data=False)
    external = external_weight_contract(model, files)

    def ports(values):
        return [{"name": p.name, "type": p.type.tensor_type.elem_type,
                 "shape": [d.dim_value if d.HasField("dim_value") else d.dim_param
                           for d in p.type.tensor_type.shape.dim]} for p in values]

    return {"onnx_version": onnx.__version__, "ir_version": model.ir_version,
            "opsets": {p.domain: p.version for p in model.opset_import},
            "inputs": ports(model.graph.input), "outputs": ports(model.graph.output),
            "operator_domains": sorted({n.domain for n in model.graph.node}),
            "external_weights": external}


def audit(root, name):
    candidate = load_candidate(name)
    directory = root / name
    files = verify_files(directory, candidate)
    return {"candidate": candidate, "verified_files": files,
            "contract_sha256": hashlib.sha256(canonical(candidate)).hexdigest(),
            "graph": graph_contract(directory / "model.onnx", candidate["files"])}


def write_new(path, value):
    # Never overwrite an earlier measurement. Failed calls leave no success record.
    raw = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x") as output:
        output.write(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "output already exists")
    names = json.loads(CATALOG.read_bytes())["models"]
    result = {"schema": 1, "admission": "candidate-only", "inference_calls": 0,
              "python": platform.python_version(),
              "catalog_sha256": hashlib.sha256(CATALOG.read_bytes()).hexdigest(),
              "models": {name: audit(args.root, name) for name in names}}
    write_new(args.output, result)
    print(json.dumps({"verified_models": list(names), "inference_calls": 0}))


if __name__ == "__main__":
    main()
