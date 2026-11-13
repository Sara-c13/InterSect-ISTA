import random, re, subprocess, shutil, numpy as np, itertools, z3, csv, pandas as pd, os, uuid, time
from pathlib import Path
from collections import Counter
from scipy.stats import ks_2samp, kstest, chisquare, mannwhitneyu
from math import floor, isclose
from bisect import bisect_right
from z3.z3util import get_vars
from tqdm.auto import tqdm


def run_z3_and_get_model(smt_filepath, timeout_ms=None):
    """Returns (status, model_dict) where status is 'SAT', 'UNSAT', or 'UNKNOWN'."""
    with open(smt_filepath, 'r') as f:
        smt2_content = f.read()

    smt2_clean = _strip_smt2_commands_2(smt2_content)
    # smt2_clean = _strip_smt2_commands_2(smt2_content)

    solver = z3.Solver()
    if timeout_ms is not None:
        solver.set(timeout=int(timeout_ms))
    assertions = z3.parse_smt2_string(smt2_clean)
    solver.add(assertions)

    result = solver.check()
    print(f"Z3 result: {result}")
    if result == z3.sat:
        model = solver.model()
        model_dict = {model[i].name(): model[model[i]] for i in range(len(model))}
        return "SAT", model_dict
    elif result == z3.unsat:
        return "UNSAT", None
    else:
        return "UNKNOWN", None

def _tokenize_smt2(content: str) -> list[str]:
    """
    Split SMT2 content into top-level s-expressions and atoms,
    correctly handling:
      - nested parentheses
      - quoted symbols |...|
      - string literals "..."
      - line comments ;...
    """
    tokens = []
    i = 0
    n = len(content)

    while i < n:
        ch = content[i]

        # Skip whitespace between top-level tokens
        if ch in ' \t\n\r':
            i += 1
            continue

        # Line comment — skip to end of line
        if ch == ';':
            while i < n and content[i] != '\n':
                i += 1
            continue

        # Quoted symbol |...|
        if ch == '|':
            start = i
            i += 1
            while i < n and content[i] != '|':
                i += 1
            i += 1  # consume closing |
            tokens.append(content[start:i])
            continue

        # String literal "..."
        if ch == '"':
            start = i
            i += 1
            while i < n:
                if content[i] == '\\':
                    i += 2  # escaped character
                elif content[i] == '"':
                    i += 1
                    break
                else:
                    i += 1
            tokens.append(content[start:i])
            continue

        # S-expression: read until matching closing paren
        if ch == '(':
            start = i
            depth = 0
            while i < n:
                c = content[i]
                if c == ';':
                    while i < n and content[i] != '\n':
                        i += 1
                    continue
                if c == '|':
                    i += 1
                    while i < n and content[i] != '|':
                        i += 1
                elif c == '"':
                    i += 1
                    while i < n:
                        if content[i] == '\\':
                            i += 2
                            continue
                        if content[i] == '"':
                            break
                        i += 1
                elif c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
                i += 1
            tokens.append(content[start:i])
            continue

        # Bare atom (keyword, symbol, number)
        start = i
        while i < n and content[i] not in ' \t\n\r();"|\n':
            i += 1
        tokens.append(content[start:i])

    return tokens

def _get_command_name(token: str) -> str | None:
    """Extract the leading keyword from an s-expression token."""
    m = re.match(r'^\(\s*([\w-]+)', token)
    return m.group(1).lower() if m else None


# Commands that z3.parse_smt2_string understands
_ALLOWED_COMMANDS = {
    "declare-fun",
    "declare-const",
    "declare-sort",
    "declare-datatypes",
    "define-fun",
    "define-fun-rec",
    "define-funs-rec",
    "define-sort",
    "assert",
    "set-logic",
}


def _strip_smt2_commands_2(smt2_content: str) -> str:
    """
    Robustly strips all commands that z3.parse_smt2_string cannot handle,
    using a proper tokenizer instead of regex. Handles:
      - push/pop and all incremental commands
      - set-info / set-option with nested parens or multiline values
      - multiple check-sat blocks (collects ALL assertions, not just pre-first)
      - quoted symbols |...| and string literals "..."
      - line comments
    """
    tokens = _tokenize_smt2(smt2_content)
    kept = [t for t in tokens if _get_command_name(t) in _ALLOWED_COMMANDS]
    return "\n".join(kept)

def _collect_symbols(expr, visited=None):
    """Recursively collect all free constants (leaf variables) in a Z3 expression."""
    if visited is None:
        visited = set()

    if expr.get_id() in visited:
        return set()
    visited.add(expr.get_id())

    symbols = set()
    if z3.is_const(expr) and expr.decl().kind() == z3.Z3_OP_UNINTERPRETED:
        symbols.add(expr)
    for child in expr.children():
        symbols |= _collect_symbols(child, visited)
    return symbols
    
def extract_free_vars_with_type_from_smt2_file(file_path):
    free_vars = []
    with open(file_path, "r") as f:
        for line in f:
            # Match zero-arg declare-fun: (declare-fun <name> () <type>)
            m_fun = re.match(r"\(declare-fun\s+(\w+)\s+\(\s*\)\s+(.+)\)", line)
            # Match declare-const: (declare-const <name> <type>)
            m_const = re.match(r"\(declare-const\s+(\w+)\s+(.+)\)", line)
            if m_fun:
                name = m_fun.group(1)
                var_type = m_fun.group(2).strip()
                free_vars.append((name, var_type))
            elif m_const:
                name = m_const.group(1)
                var_type = m_const.group(2).strip()
                free_vars.append((name, var_type))
    return free_vars

INCREMENTAL_CMDS = re.compile(
    r"^\s*\((push|pop|check-sat|check-sat-assuming|get-model|get-unsat-core|reset|exit)[^)]*\)",
    re.IGNORECASE
)

def strip_incremental_commands(smt2_content: str) -> str:
    """Remove push/pop and other incremental commands unsupported by z3.parse_smt2_string."""
    return "\n".join(
        line for line in smt2_content.splitlines()
        if not INCREMENTAL_CMDS.match(line)
    )

def get_free_consts(formula):
    """Walk a Z3 formula AST and collect all uninterpreted applications."""
    seen = set()
    result = []
    stack = [formula]
    while stack:
        expr = stack.pop()
        if not z3.is_expr(expr):
            continue
        if z3.is_app(expr):
            decl = expr.decl()
            if decl.kind() == z3.Z3_OP_UNINTERPRETED:
                name = decl.name()
                if name not in seen:
                    seen.add(name)
                    result.append(expr)
            stack.extend(expr.children())
    return result

def wrap(x, range_size, min_value):
    """Wrap integer x into [min_value, max_value] using modular arithmetic."""
    return min_value + ((x - min_value) % range_size)

def split_into_regions(min_value, max_value, num_regions):
    step = (max_value - min_value) // num_regions
    regions = [
        (min_value + i * step, min_value + (i + 1) * step if i < num_regions - 1 else max_value)
        for i in range(num_regions)
    ]
    return regions

def std_generator(min_value, max_value, width, num_intervals, range_size):
    intervals = []
    for _ in range(num_intervals):
        # Randomly select start in [min_value, max_value]
        start = random.randint(min_value, max_value)
        # Compute end using wrapping addition
        end = wrap(start + width, range_size, min_value)
        # Handle wrapped intervals that cross the boundary; normalize as two cases:
        # 1. Simple non-wrapping interval: start <= end
        # 2. Wrapping interval: start > end → split into two intervals
        if start <= end:
            intervals.append((start, end))
        else:
            # Wrapped interval; becomes two intervals:
            # [start, max_value] and [min_value, end]
            intervals.append((start, max_value))
            intervals.append((min_value, end))
    return intervals

# ------------------------------------------------------------------------------------------------------

def multiset_from_intervals_2(intervals, min_value, max_value):
    S = Counter()

    for a, b in intervals:
        try:   
            num_samples = max(5, int(1000 * (b - a) / (max_value - min_value)))
            # if not b-a > 0 and num_samples < b-a:
            #     num_samples = b-a
            # print("num_samples =", num_samples, "for interval of size", b - a)
            norm_a = (a - min_value) / (max_value - min_value)
            norm_b = (b - min_value) / (max_value - min_value)
            pts = np.linspace(norm_a, norm_b, num_samples)
        except Exception as e:
            print("\n--- ERROR IN INTERVAL ---")
            print("a =", a, type(a))
            print("b =", b, type(b))
            print("num_samples =", num_samples)
            raise e
        for x in pts:
            S[round(x,16)] += 1
    return S
# ------------------------------------------------------------------------------------------------------

def allocate_samples_per_interval(merged_intervals, total_samples, min_per_interval=1):
    """Distribuisce total_samples proporzionalmente alla lunghezza di ogni merged interval.
       Restituisce lista di sample counts corrispondenti a merged_intervals."""
    lengths = [b-a for a,b in merged_intervals]
    total_len = sum(lengths)
    if total_len == 0:
        return [min_per_interval]*len(merged_intervals)
    # prima assegna proporzionale (float), poi arrotonda e correggi per somma esatta
    raw = [total_samples * (L / total_len) for L in lengths]
    alloc = [max(min_per_interval, int(floor(x))) for x in raw]
    # correggi differenza dovuta all'arrotondamento
    diff = total_samples - sum(alloc)
    i = 0
    # distribuisci l'eventuale differenza (+/-) uno per volta
    while diff != 0:
        if diff > 0:
            alloc[i % len(alloc)] += 1
            diff -= 1
        else:  # diff < 0
            # evita togliere sotto il minimo
            j = i % len(alloc)
            if alloc[j] > min_per_interval:
                alloc[j] -= 1
                diff += 1
        i += 1
    return alloc

def integer_uniform_sample(a, b, n):
    return [a + int((b-a) * np.random.random()) for _ in range(n)]


def uniform_multiset(domain_start, domain_end, size):
    X = np.random.uniform(0, 1, size)   # <-- normalized domain
    X = np.round(X, 6)
    return Counter(X)

def expand_multiset(M):
    arr = []
    for x, count in M.items():
        arr.extend([x] * count)
    return np.array(arr)


BIN_x_COV ={
    10: 4,
    20: 4,
    30: 4,
    40: 4,
    50: 4,
    60: 4,
    70: 7,
    80: 7,
    90: 8,
}


def kolmogorov_smirnov_test(intervals, min_value, max_value, band_width, alpha=0.05):
    S_real = multiset_from_intervals_2(intervals, min_value, max_value)
    S_ideal = uniform_multiset(min_value, max_value, sum(S_real.values()))
    
    R = expand_multiset(S_real)
    I = expand_multiset(S_ideal)
    
    if band_width == 1:
        U, p = mannwhitneyu(R, I, alternative='two-sided')
        distribution_ok = p > 0.05
    else:
        D, p = ks_2samp(R, I)
        distribution_ok = D < 0.10 and p > 0.05
    return distribution_ok

# ----------------------------------------------STRING--------------------------------------------------------

TOTAL_STRINGS = sum(26**k for k in range(1, 55))

def index_to_string(i: int) -> str:
    """
    Returns the string at 0-based position i in the lexicographic enumeration
    of all strings over {a..z} with length 1 to 54.

    Order: a, b, ..., z, aa, ab, ..., az, ba, ..., zz, aaa, ...

    Indices >= TOTAL_STRINGS are clamped via modulo to stay in range.
    """
    if i < 0:
        raise ValueError("Index must be non-negative.")

    i = i % TOTAL_STRINGS  # clamp out-of-range indices

    ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
    MAX_LEN  = 54

    # --- Step 1: determine the length of the target string ---
    length = 1
    count  = 26          # number of strings of current length (26^length)

    while i >= count:
        if length == MAX_LEN:
            raise ValueError(f"Index {i} exceeds the language (max length {MAX_LEN}).")
        i     -= count
        length += 1
        count  = 26 ** length

    # --- Step 2: decode the offset `i` in base-26 into `length` characters ---
    chars = []
    for _ in range(length):
        chars.append(ALPHABET[i % 26])
        i //= 26

    return ''.join(reversed(chars))

def sample_interval(start: int, end: int, n: int) -> list[str]:
    """
    Uniformly samples n integers from [start, end] (inclusive)
    and returns their corresponding strings.

    - If n >= number of integers in the interval, all integers are returned.
    - Samples are always inclusive of both endpoints.
    """
    if start > end:
        raise ValueError(f"start ({start}) must be <= end ({end}).")
    if n <= 0:
        raise ValueError("n must be positive.")

    interval_size = end - start + 1

    if n >= interval_size:
        # Return every integer in the interval
        indices = range(start, end + 1)
    else:
        # Spread n points uniformly across [start, end], always including both endpoints
        indices = [
            round(start + i * (interval_size - 1) / (n - 1))
            for i in range(n)
        ] if n > 1 else [start]

    return [index_to_string(idx) for idx in indices]

def sample_intervals(intervals: list[tuple[int, int]], n: int) -> list[list[str]]:
    """
    Applies sample_interval to a list of (start, end) intervals.
    Returns a list of string lists, one per interval.
    """
    return [sample_interval(start, end, n) for start, end in intervals]

def create_string_mask_with_coverage(min_value, max_value, target_coverage, numStringSamples, delta):
    if min_value > max_value:
        raise ValueError("min_value must be <= max_value")
    cardinality = max_value - min_value + 1
    band_width = max(1, int(cardinality * delta))  # 0.12% del dominio
    if band_width == 1:
        min_intervals = int(target_coverage / band_width) - 10
    else:
        min_intervals = int((target_coverage / 100) / delta)

    distribution_ok = False
    iteration = 0
    
    while (
        not distribution_ok or
        (mask_coverage > target_coverage + 2) or
        (mask_coverage < target_coverage - 2)):
        # ---- Generate intervals ----
        intervals = std_generator(min_value, max_value, band_width, min_intervals, cardinality)
        
        # ---- Merge intervals ----
        intervals.sort(key=lambda x: x[0])
        merged = []

        for interval in intervals:
            if not merged:
                merged.append(interval)
            else:
                last_start, last_end = merged[-1]
                curr_start, curr_end = interval

                if curr_start <= last_end:  # overlap
                    merged[-1] = (last_start, max(last_end, curr_end))
                else:
                    merged.append(interval)
        # ---- Calculate mask coverage ----
        mask_coverage = sum(e - s for s,e in merged)/(max_value - min_value)
        mask_coverage = mask_coverage*100

        S_real = multiset_from_intervals_2(intervals, min_value, max_value)
        S_ideal = uniform_multiset(min_value, max_value, sum(S_real.values()))
        
        R = expand_multiset(S_real)
        I = expand_multiset(S_ideal)
        
        # SMALL_SAMPLE_THRESHOLD = 50
        # if len(R) < SMALL_SAMPLE_THRESHOLD or len(I) < SMALL_SAMPLE_THRESHOLD:
        
        if band_width == 1:
            U, p = mannwhitneyu(R, I, alternative='two-sided')
            distribution_ok = p > 0.05
        else:
            D, p = ks_2samp(R, I)
            distribution_ok = D < 0.10 and p > 0.05
        
        iteration += 1
            
        if ((iteration % 5 == 0) and (mask_coverage < target_coverage + 2)):
            min_intervals += 1
        if ((mask_coverage > target_coverage + 2)):
            min_intervals -= 1
    #sample each merged interval and convert samples to strings using index_to_string
    #final_mask = []
    sample_intervals_list = sample_intervals(merged, n=numStringSamples)  # 10 samples per merged interval
    return sample_intervals_list

# ----------------------------------------------STRING--------------------------------------------------------

def create_mask_with_coverage(min_value, max_value, target_coverage, delta):
    if min_value > max_value:
        raise ValueError("min_value must be <= max_value")
    cardinality = max_value - min_value + 1
    band_width = max(1, int(cardinality * delta))  # delta del dominio
    if band_width == 1:
        min_intervals = int(target_coverage / band_width) - 10
    else:
        min_intervals = int((target_coverage / 100) / delta)

    distribution_ok = False
    iteration = 0
    while (
        not distribution_ok or
        (mask_coverage > target_coverage + 2) or
        (mask_coverage < target_coverage - 2)):
        # ---- Generate intervals ----
        intervals = std_generator(min_value, max_value, band_width, min_intervals, cardinality)
        
        # ---- Merge intervals ----
        intervals.sort(key=lambda x: x[0])
        merged = []

        for interval in intervals:
            if not merged:
                merged.append(interval)
            else:
                last_start, last_end = merged[-1]
                curr_start, curr_end = interval

                if curr_start <= last_end:  # overlap
                    merged[-1] = (last_start, max(last_end, curr_end))
                else:
                    merged.append(interval)
        # ---- Calculate mask coverage ----
        mask_coverage = sum(e - s for s,e in merged)/(max_value - min_value)
        mask_coverage = mask_coverage*100

        distribution_ok = kolmogorov_smirnov_test(
            intervals,
            min_value,
            max_value,
            band_width
        )
        
        iteration += 1
            
        if ((iteration % 5 == 0) and (mask_coverage < target_coverage + 2)):
            min_intervals += 1
        if ((mask_coverage > target_coverage + 2)):
            min_intervals -= 1
    # print("Coverage: ", mask_coverage)
    # print("Bands: ", merged)
    return merged
    
def get_sat_statuses(file_path):
    result = subprocess.run(["z3", file_path], capture_output=True, text=True)
    output = result.stdout.strip().splitlines()

    statuses = []
    count = 1

    for line in output:
        line = line.strip()
        if line in ("sat", "unsat", "unknown"):
            statuses.append((f"assert_{count}", line))
            count += 1

    return statuses

def select_free_vars(
    free_vars: list[tuple[str, str]],
    keep: list[str] | None = None,
    select_type: str | None = None,
    interactive: bool = True,
) -> list[tuple[str, str]]:
    """
    Select which free variables to keep.
    
    Args:
        free_vars: list of (name, type) tuples
        keep: optional list of variable names to keep directly (skips interaction)
        select_type: if provided, pre-filters free_vars to only variables of this type
        
    Returns:
        list of (name, type) tuples for kept variables
    """
    if select_type is not None:
        free_vars = [(name, typ) for name, typ in free_vars if typ == select_type]

    name_to_var = {name: (name, typ) for name, typ in free_vars}

    # --- Non-interactive mode ---
    if keep is not None or not interactive:
        if keep is None:
            keep = [name for name, _ in free_vars]  # tieni tutto se non specificato
        invalid = [n for n in keep if n not in name_to_var]
        if invalid:
            raise ValueError(f"Unknown variable names: {invalid}")
        selected = [name_to_var[n] for n in keep]
        discarded = [v for v in free_vars if v not in selected]
        print(f"✓ Kept    ({len(selected)}): {[n for n, _ in selected]}")
        print(f"✗ Discarded ({len(discarded)}): {[n for n, _ in discarded]}")
        return selected

    # --- Interactive mode ---
    print("\n=== Free Variable Selection ===")
    type_info = f" of type '{select_type}'" if select_type is not None else ""
    print(f"Found {len(free_vars)} free variables{type_info}:\n")
    for i, (name, typ) in enumerate(free_vars):
        print(f"  [{i}] {name:20s} : {typ}")
    print("\nOptions:")
    print("  - Enter indices to KEEP (e.g. '0 2 3')")
    print("  - Enter 'all' to keep all")
    print("  - Enter 'none' to discard all")
    print("  - Enter variable names directly (e.g. 'call_value1 sender_2')")

    while True:
        choice = input("\nYour selection: ").strip().lower()

        if choice == "all":
            selected = free_vars
            break
        elif choice == "none":
            selected = []
            break
        else:
            tokens = choice.split()
            if all(t.isdigit() for t in tokens):
                indices = [int(t) for t in tokens]
                invalid = [i for i in indices if i >= len(free_vars)]
                if invalid:
                    print(f"  Invalid indices: {invalid}. Try again.")
                    continue
                selected = [free_vars[i] for i in indices]
                break
            invalid = [t for t in tokens if t not in name_to_var]
            if invalid:
                print(f"  Unknown variable names: {invalid}. Try again.")
                continue
            selected = [name_to_var[t] for t in tokens]
            break

    discarded = [v for v in free_vars if v not in selected]
    print(f"\n✓ Kept    ({len(selected)}): {[n for n, _ in selected]}")
    print(f"✗ Discarded ({len(discarded)}): {[n for n, _ in discarded]}")
    return selected

# USAGE
# file_path = Path(__file__).parent.parent / "output" / "me4a137ba.smt2"
# free_vars = extract_free_vars_from_smt2_file(file_path)
# for free_var in free_vars:
#     print(is_brute_force_applicable(file_path, free_var, max_size=1000))

# file_path = Path(__file__).parent.parent / "output" / "m01d2485e.smt2"
# num_intervals = 5
# # Extract free variables
# free_vars = extract_free_vars_from_smt2_file(file_path)
# free_var = free_vars[1]  # Just take the first one for this example
# # For each free variable, extract domain and generate mask
# tmp_file_path_main = copy_smt_for_processing(file_path, None)
# if add_get_model_after_sat_checks(tmp_file_path_main):
#     # Cicla sulle free variables
#     # for free_var in free_vars:
#     for _ in range(1):
#         z3_models_collected = []
#         min_value, max_value = extract_domain_for_free_var(free_var, tmp_file_path_main)
#         domain = max_value - min_value + 1
#         interval_width = int(domain * 0.001)   
#         # Maschera 100 volte il dominio della free variable
#         for i in range(800):
#             # Crea una copia temporanea del file SMT per questa iterazione
#             tmp_file_path_to_process = copy_smt_for_processing(tmp_file_path_main, str(i)+"_"+free_var)
#             # Crea la maschera e aggiungila al file temporaneo
#             mask = create_mask(min_value, max_value, num_intervals, interval_width)
#             add_mask(mask, free_var, tmp_file_path_to_process)
#             # Aggiungi le blocking asserts per tutti i modelli raccolti finora
#             for model in z3_models_collected:
#                 blocking_clause = create_blocking_assert(model, or_=False)
#                 add_blocking_assert(blocking_clause, tmp_file_path_to_process)
#             # Esegui Z3 e raccogli il nuove modello
#             model = run_z3_and_collect_models(tmp_file_path_to_process)
#             #aggiungi il modello raccolto alla lista
#             if not model:
#                 continue
#             z3_models_collected.append(model[0])
#         # salva i modelli raccolti in un file csv
#         models_output_path = Path(__file__).parent.parent / "output" / f"models_{free_var}.csv"
#         with open(models_output_path, "w") as f:
#             # Scrivi header
#             if z3_models_collected:
#                 header = ",".join(z3_models_collected[0].keys())
#                 f.write(header + "\n")
#             # Scrivi modelli
#             for model in z3_models_collected:
#                 row = ",".join(model.values())
#                 f.write(row + "\n")
        


# import csv
# domain = 2**8
# start = 0
# end   =  2**8 - 1


# width_fractions = [
#     0.0001,  # 0.01%
#     0.0002,  # 0.02%
#     0.0003,  # 0.03%
#     0.0004,  # 0.04%
#     0.0005,  # 0.05%
#     0.0006,  # 0.06%
#     0.0007,  # 0.07%
#     0.0008,  # 0.08%
#     0.0009,  # 0.09%
#     0.0010,  # 0.10%  WORKS WELL
#     0.0011,  # 0.11%
#     0.0012,  # 0.12%
#     0.0013,  # 0.13%
#     0.0014,  # 0.14%
#     0.0015,  # 0.15%
#     0.0016,  # 0.16%
#     0.0017,  # 0.17%
#     0.0018,  # 0.18%
#     0.0019,  # 0.19%
#     0.0020,  # 0.20%
# ]
# width_fractions = [
#     0.1
# ]
# results = []
# for interval_numbers in range(1, 200):  # 5 → 100
#     for frac in width_fractions:
#         interval_width = int(domain * frac)

#         # Evita width = 0 per numeri piccoli
#         if interval_width < 1:
#             interval_width = 1

#         for _ in range(10):
#             # --- CALL create_mask ---
#             p, cov = create_mask(
#                 start,
#                 end,
#                 interval_numbers,
#                 interval_width
#             )

#             # --- REGISTRA IL RECORD ---
#             results.append({
#                 "interval_numbers": interval_numbers,
#                 "interval_width_fraction": frac,
#                 "interval_width": interval_width,
#                 "mask_coverage": cov,
#                 "p": p,
#             })

# # -------------------------------
# # SALVA IN CSV
# # -------------------------------

# with open("mask_uniformity_int8_results.csv", "w", newline="") as f:
#     writer = csv.DictWriter(f, fieldnames=results[0].keys())
#     writer.writeheader()
#     writer.writerows(results)

# print("\nSalvato")

# smt_filepath = Path(__file__).parent.parent / "no_axioms.smt2"
# free_vars = extract_free_vars_from_smt2_file(smt_filepath)
# hybrid_state_space_sampler(free_vars, smt_filepath, theta=256, coverage=30)
