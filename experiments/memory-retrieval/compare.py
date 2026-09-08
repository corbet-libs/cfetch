"""Replay two measured candidates through cfetch; no inference or admission gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile

LIMIT = 64 * 1024 * 1024
TIMEOUT_SECONDS = 120
MODES = ("bm25", "vector", "hybrid")
PURPOSE = (
    "Diagnostic ordered producer pairs and two deterministic document mixtures; "
    "not profile admission, an adversarial mixed-store gate, or convergence proof"
)


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def executable_sha256(path):
    maximum = 512 * 1024 * 1024
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum:
            raise ValueError("cfetch executable must be a regular file of at most 512 MiB")
        digest, count = hashlib.sha256(), 0
        while chunk := stream.read(min(1024 * 1024, maximum - count + 1)):
            count += len(chunk)
            if count > maximum:
                raise ValueError("cfetch executable exceeded its size bound")
            digest.update(chunk)
        after = os.fstat(stream.fileno())
        if count != before.st_size or (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size, after.st_mtime_ns, after.st_ctime_ns
        ):
            raise ValueError("cfetch executable changed while being hashed")
        return digest.hexdigest()


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"nonfinite JSON number: {value}")


def parse(raw):
    return json.loads(raw, object_pairs_hook=strict_object, parse_constant=reject_constant)


def read(path, maximum=LIMIT):
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= maximum:
            raise ValueError("input must be a nonempty bounded regular file")
        raw = stream.read(maximum + 1)
        if len(raw) != metadata.st_size:
            raise ValueError("input changed size while being read")
        return raw


def serialize(value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if len(raw) > LIMIT:
        raise ValueError("generated artifact exceeds the input size limit")
    return raw


def publish(path, raw):
    descriptor, temporary = tempfile.mkstemp(prefix=".comparison-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def load_bundle(path, manifest_hash, identifiers):
    raw = read(path)
    bundle = parse(raw)
    if bundle["schema_version"] != 1 or bundle["manifest_sha256"] != manifest_hash:
        raise ValueError("source bundle does not match the exact manifest")
    if not isinstance(bundle["provenance"], dict) or not bundle["provenance"]:
        raise ValueError("source bundle must retain its producer provenance")
    outputs = {}
    for output in bundle["outputs"]:
        if output["id"] in outputs:
            raise ValueError("duplicate source output identity")
        outputs[output["id"]] = output
    if set(outputs) != identifiers:
        raise ValueError("source bundle coverage differs from manifest, including refusals")
    return outputs, {"bundle_sha256": sha256(raw), "provenance": bundle["provenance"]}


def replay(args, bundle_path, bundle_hash, manifest_hash, executable_hash):
    command = [str(args.cfetch), "retrieval-eval", "--corpus", str(args.corpus),
               "--representation", args.representation, "--vectors", str(bundle_path)]
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        if executable_sha256(args.cfetch) != executable_hash:
            raise ValueError("cfetch executable changed before retrieval-eval")
        try:
            result = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=TIMEOUT_SECONDS)
        finally:
            if executable_sha256(args.cfetch) != executable_hash:
                raise ValueError("cfetch executable changed during retrieval-eval")
        if result.returncode:
            stderr.seek(0)
            detail = stderr.read(4096).decode("utf-8", errors="replace")
            raise RuntimeError(f"retrieval-eval exited {result.returncode}: {detail}")
        stdout.seek(0)
        raw = stdout.read(LIMIT + 1)
    if len(raw) > LIMIT:
        raise ValueError("retrieval report exceeds its size bound")
    report = parse(raw)
    if (report["schema_version"] != 1 or report["production_admission"] is not False
            or report["manifest_sha256"] != manifest_hash
            or report["vector_bundle_sha256"] != bundle_hash):
        raise ValueError("retrieval report is not bound to this exact case")
    return report, raw


def summarize(report, expected_queries):
    results = {query["id"]: query for query in report["results"]}
    if len(results) != len(report["results"]) or set(results) != set(expected_queries):
        raise ValueError("retrieval report changed query coverage")
    critical_ids = [key for key, query in expected_queries.items() if query["critical"]]
    for key, query in results.items():
        if query["critical"] != expected_queries[key]["critical"]:
            raise ValueError("retrieval report changed critical query labels")
    critical = {}
    for mode in MODES:
        failed = []
        for identifier in critical_ids:
            ranking = results[identifier]["modes"][mode]["ranking"]
            if not ranking or ranking[0]["rank"] != 1 or ranking[0]["grade"] <= 0:
                failed.append(identifier)
        critical[mode] = {"queries": len(critical_ids),
                          "relevant_at_rank_one": len(critical_ids) - len(failed),
                          "failed_query_ids": failed}
    all_rows = {row["mode"]: row for row in report["summary"] if row["category"] == "all"}
    if set(all_rows) != set(MODES) or any(
        row["queries_evaluated"] != len(expected_queries) for row in all_rows.values()
    ):
        raise ValueError("retrieval summary dropped queries from its denominators")
    return {"queries": len(results), "inputs": report["inputs"], "blocks": report["blocks"],
            "document_vectors_available": report["document_vectors_available"],
            "refused_inputs": len(report["refused"]), "metrics": report["summary"],
            "critical_first_hit": critical, "vector_measurement": report["vector_measurement"]}


def run(args):
    args.cfetch = args.cfetch.absolute()
    executable_hash = executable_sha256(args.cfetch)
    manifest_raw = read(args.manifest)
    manifest = parse(manifest_raw)
    manifest_hash = sha256(manifest_raw)
    if (manifest["schema_version"] != 1 or manifest["production_admission"] is not False
            or manifest["representation"] != args.representation.replace("-", "_")):
        raise ValueError("unsupported manifest or mismatched representation")
    if sha256(read(args.corpus, 4 * 1024 * 1024)) != manifest["corpus_sha256"]:
        raise ValueError("corpus differs from the exact manifest")
    kinds = {}
    for item in manifest["inputs"]:
        if (item["kind"] not in {"query", "document"} or item["id"] in kinds
                or item["id"] != sha256(b"cfetch-evaluation-input-v1\0" + item["text"].encode())):
            raise ValueError("invalid or duplicate manifest input")
        kinds[item["id"]] = item["kind"]
    if not 1 <= len(kinds) <= 4096 or set(kinds.values()) != {"query", "document"}:
        raise ValueError("expected bounded query and document inputs")
    queries = {query["id"]: query for query in manifest["queries"]}
    if not queries or len(queries) != len(manifest["queries"]):
        raise ValueError("invalid manifest query identities")
    sources, provenance = {}, {}
    for name in ("first", "second"):
        sources[name], provenance[name] = load_bundle(getattr(args, name), manifest_hash, set(kinds))
    args.output_directory.mkdir(parents=True, exist_ok=False)
    documents = sorted(identifier for identifier, kind in kinds.items() if kind == "document")
    mixtures = {
        "alternating": {identifier: ("first" if index % 2 == 0 else "second")
                        for index, identifier in enumerate(documents)},
        "complement": {identifier: ("second" if index % 2 == 0 else "first")
                       for index, identifier in enumerate(documents)},
    }
    cases = []
    for query_source in ("first", "second"):
        for document_source in ("first", "second", "alternating", "complement"):
            name = f"queries-{query_source}_documents-{document_source}"
            assignments = {
                identifier: query_source if kind == "query" else (
                    document_source if document_source in sources else mixtures[document_source][identifier]
                ) for identifier, kind in sorted(kinds.items())
            }
            case_provenance = {"purpose": PURPOSE, "sources": provenance,
                               "comparison_script_sha256": sha256(read(Path(__file__))),
                               "cfetch_executable_sha256": executable_hash,
                               "query_producer": query_source, "document_selection": document_source,
                               "input_sources": assignments}
            bundle = {"schema_version": 1, "manifest_sha256": manifest_hash,
                      "provenance": case_provenance,
                      "outputs": [sources[source][identifier] for identifier, source in assignments.items()]}
            bundle_raw = serialize(bundle)
            bundle_path = args.output_directory / f"{name}.bundle.json"
            publish(bundle_path, bundle_raw)
            report, report_raw = replay(args, bundle_path, sha256(bundle_raw), manifest_hash, executable_hash)
            report_path = args.output_directory / f"{name}.report.json"
            publish(report_path, report_raw)
            cases.append({"case": name, "bundle": bundle_path.name, "report": report_path.name,
                          "bundle_sha256": sha256(bundle_raw), "report_sha256": sha256(report_raw),
                          **summarize(report, queries)})
            print(f"Completed {len(cases)}/8: {name}", flush=True)
    summary = {"schema_version": 1, "production_admission": False, "purpose": PURPOSE,
               "manifest_sha256": manifest_hash, "representation": manifest["representation"],
               "cfetch_executable_sha256": executable_hash,
               "sources": provenance, "case_count": len(cases), "cases": cases}
    publish(args.output_directory / "summary.json", serialize(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("cfetch", "corpus", "manifest", "first", "second", "output-directory"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--representation", choices=("body", "heading-context", "context-payload"), required=True)
    run(parser.parse_args())
