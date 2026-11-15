import random, re, numpy as np, z3
from collections import Counter
from scipy.stats import ks_2samp, mannwhitneyu

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

def wrap(x, range_size, min_value):
    """Wrap integer x into [min_value, max_value] using modular arithmetic."""
    return min_value + ((x - min_value) % range_size)

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

def uniform_multiset(domain_start, domain_end, size):
    X = np.random.uniform(0, 1, size)   # <-- normalized domain
    X = np.round(X, 6)
    return Counter(X)

def expand_multiset(M):
    arr = []
    for x, count in M.items():
        arr.extend([x] * count)
    return np.array(arr)

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
