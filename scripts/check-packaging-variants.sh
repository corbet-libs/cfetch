#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
pkgbuild="$root/packaging/arch/PKGBUILD"
catalog="$root/release/variants.json"

bash -n "$pkgbuild"
# shellcheck disable=SC1090
source "$pkgbuild"

for mapping in \
  'x86_64 linux-cfetch-cpu-x86_64 -march=x86-64 target-cpu=x86-64' \
  'aarch64 linux-cfetch-cpu-arm64 -march=armv8-a target-cpu=generic'
do
  read -r carch expected expected_c_target expected_rust_target <<<"$mapping"
  CARCH=$carch
  actual=$(_cfetch_variant)
  test "$actual" = "$expected"
  jq -e --arg id "$actual" \
    '.variants[] | select(.id == $id and .backend == "cpu")' \
    "$catalog" >/dev/null

  CFLAGS='-march=native -O3 -pipe -mtune=native -fstack-clash-protection'
  CXXFLAGS='-mcpu=native -O3 -pipe -Wp,-D_GLIBCXX_ASSERTIONS'
  RUSTFLAGS='-C target-cpu=native -C force-frame-pointers=yes'
  _cfetch_portable_build_env
  grep -Fq -- "$expected_c_target" <<<"$CFLAGS"
  grep -Fq -- "$expected_c_target" <<<"$CXXFLAGS"
  grep -Fq -- "$expected_rust_target" <<<"$RUSTFLAGS"
  ! grep -Eq -- '-m(arch|cpu|tune)=native|target-cpu=native' <<<"$CFLAGS $CXXFLAGS $RUSTFLAGS"
  grep -Fq -- '-fstack-clash-protection' <<<"$CFLAGS"
  grep -Fq -- '-Wp,-D_GLIBCXX_ASSERTIONS' <<<"$CXXFLAGS"
  grep -Fq -- '-C force-frame-pointers=yes' <<<"$RUSTFLAGS"
done

for phase in build check; do
  declare -f "$phase" | grep -Fq 'CFETCH_VARIANT="$(_cfetch_variant)"'
  declare -f "$phase" | grep -Fq '_cfetch_portable_build_env'
done

echo PACKAGING_VARIANTS_OK
