# Native Library Build ID Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent the Python solver process from ever accepting a stale, ABI-incompatible, or incomplete native `.so`, while retaining architecture-specific prebuilt binaries and automatic local compilation.

**Architecture:** Each native solver build embeds an ABI version and deterministic Build ID derived from its C++ source, required exported symbols, platform tag, and compile recipe. The loader validates candidates in a disposable subprocess, copies a valid prebuilt into a content-addressed ignored cache, and only then loads that unique path in the main process; invalid candidates trigger an atomic rebuild, never a stale fallback.

**Tech Stack:** Python 3 standard library (`ctypes`, `hashlib`, `json`, `subprocess`, `tempfile`, `os.replace`), C++17 shared libraries, pytest.

## Global Constraints

- Keep architecture-specific repository binaries such as `*.darwin_arm64.so` and `*.linux_x86_64.so`; do not solve staleness by ignoring every `.so`.
- The main Python process must not probe an unverified binary with `ctypes.CDLL`; probing occurs in a short child process.
- A candidate is valid only when it loads, exports every required function, reports ABI version `1`, and reports the exact expected Build ID.
- Missing metadata is an old binary and must be rejected.
- Invalid or missing candidates compile into a unique temporary file, pass the same validation, and are atomically promoted to a content-addressed cache path.
- Compilation failure raises a diagnostic error; it never permits an old binary to run.
- Preserve the existing uncommitted automatic-showdown source changes and training/evaluation behavior.

---

### Task 1: Specify Build Identity and Rejection Behavior

**Files:**
- Create: `tests/test_native_lib_loader.py`
- Modify: `trick_taking/solvers/native_lib_loader.py`

**Interfaces:**
- Produces: `compute_native_build_id(src, *, required_symbols, abi_version, build_recipe, target_platform=None) -> str`.
- Produces: `versioned_library_path(directory, base_name, build_id) -> str`.
- Produces: `ensure_native_library(..., required_symbols, abi_version, build_recipe) -> str`.
- Internal probe contract: `_probe_native_library(path, *, expected_build_id, expected_abi_version, required_symbols) -> tuple[bool, str]`.

- [ ] **Step 1: Write failing deterministic-identity tests**

```python
def test_build_id_changes_for_every_compatibility_input(tmp_path):
    source = tmp_path / "solver.cpp"
    source.write_text("int solver = 1;", encoding="utf-8")
    baseline = compute_native_build_id(
        str(source),
        required_symbols=("solve_native",),
        abi_version=1,
        build_recipe="cxx17-o3-v1",
        target_platform="linux_x86_64",
    )
    assert baseline != compute_native_build_id(
        str(source), required_symbols=("solve_native_with_q",), abi_version=1,
        build_recipe="cxx17-o3-v1", target_platform="linux_x86_64",
    )
    assert baseline != compute_native_build_id(
        str(source), required_symbols=("solve_native",), abi_version=2,
        build_recipe="cxx17-o3-v1", target_platform="linux_x86_64",
    )
    assert baseline != compute_native_build_id(
        str(source), required_symbols=("solve_native",), abi_version=1,
        build_recipe="cxx17-o3-v2", target_platform="linux_x86_64",
    )
    assert baseline != compute_native_build_id(
        str(source), required_symbols=("solve_native",), abi_version=1,
        build_recipe="cxx17-o3-v1", target_platform="darwin_arm64",
    )
    source.write_text("int solver = 2;", encoding="utf-8")
    assert baseline != compute_native_build_id(
        str(source), required_symbols=("solve_native",), abi_version=1,
        build_recipe="cxx17-o3-v1", target_platform="linux_x86_64",
    )
```

- [ ] **Step 2: Run the identity test and observe the missing API failure**

Run: `pytest -q tests/test_native_lib_loader.py::test_build_id_changes_for_every_compatibility_input`

Expected: collection fails because `compute_native_build_id` does not exist.

- [ ] **Step 3: Implement deterministic Build IDs and cache paths**

Hash a canonical JSON record containing schema version, ABI version, recipe, platform, and sorted required symbols, followed by the exact source bytes. Place versioned load artifacts under ignored `trick_taking/solvers/__pycache__/native/` using the full Build ID in the filename.

- [ ] **Step 4: Run the identity tests to green**

Run: `pytest -q tests/test_native_lib_loader.py -k build_id`

Expected: all selected tests pass.

- [ ] **Step 5: Write failing stale-candidate orchestration tests**

Use `monkeypatch` on the private subprocess probe boundary so the test can represent an old loadable prebuilt without loading arbitrary machine code in pytest. Assert that an old architecture binary is rejected, compilation receives the expected Build ID and ABI, the compiled temporary binary is validated, and the returned path contains that Build ID. Add a second test asserting a failed compilation raises `NativeLibraryError` rather than returning the old path.

- [ ] **Step 6: Run the stale-candidate tests and observe the signature/behavior failures**

Run: `pytest -q tests/test_native_lib_loader.py -k 'rejects or compilation'`

Expected: tests fail because `ensure_native_library` has neither metadata validation nor content-addressed compilation.

- [ ] **Step 7: Implement isolated validation and atomic promotion**

The child probe loads the candidate, checks required symbols, configures `spades_native_build_id` as `char* ()`, configures `spades_native_abi_version` as `uint32_t ()`, and compares exact values. Valid prebuilt candidates are atomically copied to the versioned cache; invalid candidates are never returned. Compilation uses a same-directory `mkstemp`, validates the result, and calls `os.replace` only after success. Always remove leftover temporary files in `finally`.

- [ ] **Step 8: Run the complete loader unit suite**

Run: `pytest -q tests/test_native_lib_loader.py`

Expected: all tests pass.

### Task 2: Embed Identity in Every Native Solver Build

**Files:**
- Modify: `trick_taking/solvers/_native_compile.py`
- Modify: `trick_taking/solvers/exact_double_dummy_cpp_fastest_core.cpp`
- Modify: `trick_taking/solvers/exact_double_dummy_cpp_native_core.cpp`
- Modify: `trick_taking/solvers/exact_double_dummy_cpp_opt1_core.cpp`
- Modify: `tests/test_native_lib_loader.py`

**Interfaces:**
- Consumes: the Build ID and ABI integer supplied by `ensure_native_library`.
- Produces: each compile function accepts `(src: str, out: str, build_id: str, abi_version: int)`.
- Produces: each shared library exports `spades_native_build_id()` and `spades_native_abi_version()`.
- Produces: `FASTEST_BUILD_RECIPE`, `NATIVE_BUILD_RECIPE`, and `OPT1_BUILD_RECIPE` strings derived from the flags used by their compiler command builders.

- [ ] **Step 1: Write a failing compile-command test**

Patch `subprocess.check_call`, invoke a compile function, and assert the command contains exact definitions equivalent to:

```text
-DSPADES_NATIVE_BUILD_ID="<64 hex characters>"
-DSPADES_NATIVE_ABI_VERSION=1
```

- [ ] **Step 2: Run the compile-command test and observe the old two-argument API failure**

Run: `pytest -q tests/test_native_lib_loader.py -k compile_command`

Expected: failure because compile functions do not accept or embed identity metadata.

- [ ] **Step 3: Refactor command construction and add metadata flags**

Make each recipe string come from the same canonical flag list used to build its compiler command, excluding environment-specific compiler executable and input/output paths. Add quoted Build ID and integer ABI macros to all three commands.

- [ ] **Step 4: Add common C++ identity exports**

Add guarded defaults so an accidentally manual unversioned build is identifiable but invalid:

```cpp
#ifndef SPADES_NATIVE_BUILD_ID
#define SPADES_NATIVE_BUILD_ID "unversioned"
#endif
#ifndef SPADES_NATIVE_ABI_VERSION
#define SPADES_NATIVE_ABI_VERSION 0
#endif

const char* spades_native_build_id() { return SPADES_NATIVE_BUILD_ID; }
uint32_t spades_native_abi_version() { return SPADES_NATIVE_ABI_VERSION; }
```

- [ ] **Step 5: Run loader and command tests**

Run: `pytest -q tests/test_native_lib_loader.py`

Expected: all tests pass.

### Task 3: Require Exact Metadata from All Python Wrappers

**Files:**
- Modify: `trick_taking/solvers/exact_double_dummy_cpp_fastest.py`
- Modify: `trick_taking/solvers/exact_double_dummy_cpp_native.py`
- Modify: `trick_taking/solvers/exact_double_dummy_cpp_opt1.py`
- Modify: `tests/test_forced_outcome_native.py`

**Interfaces:**
- Consumes: `NATIVE_LIBRARY_ABI_VERSION = 1` from `native_lib_loader.py` and recipe strings from `_native_compile.py`.
- Fastest required symbols: `solve_native`, `solve_native_with_q`, `analyze_forced_outcome_native`.
- Native/opt1 required symbols: `solve_native`, `solve_native_with_q`.

- [ ] **Step 1: Write a failing fastest-solver integration assertion**

Instantiate `ExactDoubleDummyCppFastestSolver`, assert `native_available`, call its two identity exports through ctypes, and compare them with `compute_native_build_id` for the fastest source, required symbols, ABI `1`, and `FASTEST_BUILD_RECIPE`.

- [ ] **Step 2: Run the integration assertion and observe rejection of the current unversioned binary**

Run: `pytest -q tests/test_forced_outcome_native.py -k current_build_id`

Expected: failure because the wrapper does not yet request identity metadata and the current binary has no identity exports.

- [ ] **Step 3: Pass exact solver specs to the loader**

Update all three wrappers to provide their complete required-symbol tuples, ABI version, and matching compile recipe. Preserve the existing wrapper behavior after a verified library path is returned.

- [ ] **Step 4: Run the integration test and force a verified local rebuild**

Run: `pytest -q tests/test_forced_outcome_native.py -k current_build_id`

Expected: the stale prebuilt is rejected in the child probe, a versioned cache binary is built, and the identity assertion passes.

- [ ] **Step 5: Run all native solver tests**

Run: `pytest -q tests/test_native_lib_loader.py tests/test_forced_outcome_native.py tests/test_exact_solver_thread_safety.py`

Expected: all tests pass.

### Task 4: Distribution and Regression Verification

**Files:**
- Modify: `README.md`
- Regenerate when possible: `trick_taking/solvers/_exact_double_dummy_cpp_fastest_core.darwin_arm64.so`
- Regenerate in a Linux x86_64 environment when authorized: `trick_taking/solvers/_exact_double_dummy_cpp_fastest_core.linux_x86_64.so`

**Interfaces:**
- The stable architecture paths remain distribution inputs.
- The ignored content-addressed cache remains the only path returned for main-process loading.

- [ ] **Step 1: Document binary validation and fallback behavior**

Explain that repository prebuilt binaries are validated against source and ABI, copied into an ignored versioned cache, and automatically rebuilt when stale. State that a compiler is required only when a valid platform prebuilt is unavailable.

- [ ] **Step 2: Rebuild and verify the current-platform prebuilt**

Compile with its Build ID/ABI macros, then run the isolated probe against the stable architecture path.

Expected: ABI `1`, exact Build ID, and every fastest-solver symbol are present.

- [ ] **Step 3: Verify the other tracked platform binary or report the explicit build-environment blocker**

Run the matching-platform build and isolated probe in an authorized Linux x86_64 environment. If no such environment is authorized, leave source/runtime auto-rebuild functional and report that the tracked Linux prebuilt remains intentionally rejected rather than claiming it is current.

- [ ] **Step 4: Run scoped regression suites**

Run: `pytest -q tests/test_native_lib_loader.py tests/test_forced_outcome.py tests/test_forced_outcome_native.py tests/test_exact_solver_thread_safety.py tests/test_gui_backend.py tests/test_game_server_showdown.py`

Run: `npm test -- --run` from `gui/`.

Expected: Python and frontend suites pass.

- [ ] **Step 5: Run repository hygiene checks**

Run: `git diff --check` and inspect `git status --short` plus the scoped diff.

Expected: no whitespace errors, no versioned cache artifacts in Git status, and no unrelated file staged or overwritten.
