import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import math
from typing import List, Set, Tuple, Dict
from collections import defaultdict

# -----------------------------
# Constants
# -----------------------------
ALPHABET = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']
STANDARD_ALPHABET = ALPHABET[:20]  # enforce 20 AA for sequences
AA_3_TO_1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
    'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
    'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

# -----------------------------
# Tensor utilities
# -----------------------------

def aa20(t: torch.Tensor) -> torch.Tensor:
    """Slice logits/probs tensors to 20 standard AA channels.
    Supports (B,L,V) or (L,V); returns same rank with V==20.
    """
    if t.dim() not in (2, 3):
        raise ValueError(f"Tensor must have dim 2 or 3, got {t.dim()}")
    if t.shape[-1] < 20:
        raise ValueError(f"Expected at least 20 channels, got {t.shape[-1]}")
    return t[..., :20]

# -----------------------------
# PDB parsing
# -----------------------------

def get_pdb_info(pdb_path: str, chains_to_parse: List[str] = None) -> Tuple[str, Dict[str, int], List[str]]:
    sequence = []
    res_id_to_index = {}
    ordered_res_ids = []
    seen_residues = set()
    current_index = -1

    chains_set = set(chains_to_parse) if chains_to_parse else None

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                chain = line[21:22].strip()
                if chains_set is None or chain in chains_set:
                    res_seq_str = line[22:26].strip()
                    if not res_seq_str:
                        continue
                    res_seq = int(res_seq_str)
                    res_name = line[17:20].strip()
                    atom_name = line[12:16].strip()
                    icode = line[26:27].strip()

                    if atom_name == "CA":
                        res_id_full = (chain, res_seq, icode)
                        if res_id_full not in seen_residues:
                            seen_residues.add(res_id_full)
                            current_index += 1
                            if res_name in AA_3_TO_1:
                                sequence.append(AA_3_TO_1[res_name])
                                res_id_simple = f"{chain}{res_seq}{icode.strip()}"
                                res_id_to_index[res_id_simple] = current_index
                                ordered_res_ids.append(res_id_simple)
                            else:
                                print(f"Warning: Skipping unrecognized residue '{res_name}' at {res_id_full}")
    return "".join(sequence), res_id_to_index, ordered_res_ids

# -----------------------------
# Fixed residues (post-hoc)
# -----------------------------

def apply_fixed_residues(sequences: Set[str], native_sequence: str, fixed_indices: Set[int]) -> Set[str]:
    if not fixed_indices:
        return sequences
    corrected_sequences = set()
    native_list = list(native_sequence)
    for seq in sequences:
        seq_list = list(seq)
        if len(seq_list) != len(native_list):
            corrected_sequences.add(seq)
            continue
        for idx in fixed_indices:
            if 0 <= idx < len(seq_list):
                seq_list[idx] = native_list[idx]
        corrected_sequences.add("".join(seq_list))
    return corrected_sequences

# -----------------------------
# Core helpers
# -----------------------------

def get_probs(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Softmax over 20 AA channels. Input: (B,L,V) or (L,V). Output: (1,L,20)."""
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
    if logits.dim() != 3:
        raise ValueError("Logits tensor must have 3 dims after unsqueeze")
    logits20 = aa20(logits)
    return F.softmax(logits20 / max(1e-8, temperature), dim=-1)


def sample_sequences_from_probs(probs: torch.Tensor, num_sequences: int) -> List[str]:
    probs = probs.squeeze(0)  # (L,20)
    probs_sum = probs.sum(dim=-1, keepdim=True)
    probs = probs / (probs_sum + 1e-9)
    sampled_indices = torch.multinomial(probs, num_samples=num_sequences, replacement=True).T
    sequences = []
    for indices in sampled_indices:
        sequences.append("".join([STANDARD_ALPHABET[int(i)] for i in indices]))
    return sequences


def logits_to_greedy_sequence(logits: torch.Tensor) -> str:
    indices = torch.argmax(aa20(logits).squeeze(), dim=-1)
    return "".join([STANDARD_ALPHABET[int(i)] for i in indices])


def calculate_sequence_score(sequence: str, logits_list: List[torch.Tensor], weights: List[float], device: torch.device) -> float:
    try:
        indices = torch.tensor([STANDARD_ALPHABET.index(aa) for aa in sequence], device=device).long()
    except ValueError:
        return -float('inf')
    total_weighted_log_prob = 0.0
    for i, logits in enumerate(logits_list):
        log_probs = F.log_softmax(aa20(logits), dim=-1).squeeze(0)  # (L,20)
        seq_log_probs = torch.gather(log_probs, 1, indices.view(-1, 1)).squeeze()
        total_weighted_log_prob += float(torch.sum(seq_log_probs)) * weights[i]
    return total_weighted_log_prob

# -----------------------------
# Strategies
# -----------------------------

def weighted_average(logits_list: List[torch.Tensor], weights: List[float], num_sequences: int, temperature: float) -> Set[str]:
    print(f"--- Weighted Average (T={temperature}) ---")
    if len(logits_list) != len(weights):
        raise ValueError("Number of logits tensors must match weights.")
    weighted_logits = torch.zeros_like(aa20(logits_list[0]))
    for logit, weight in zip(logits_list, weights):
        weighted_logits += weight * aa20(logit)
    probs = get_probs(weighted_logits, temperature=temperature)
    return set(sample_sequences_from_probs(probs, num_sequences))


def product_of_experts(logits_list: List[torch.Tensor], num_sequences: int, temperature: float) -> Set[str]:
    print(f"--- Product of Experts (T={temperature}) ---")
    if not logits_list:
        return set()
    prob_list = [get_probs(logits, temperature=1.0) for logits in logits_list]
    combined_probs = prob_list[0]
    for prob in prob_list[1:]:
        combined_probs = combined_probs * prob
    combined_logits = torch.log(combined_probs + 1e-9)
    final_probs = get_probs(combined_logits, temperature=temperature)
    return set(sample_sequences_from_probs(final_probs, num_sequences))


def get_wta_logits(logits_A: torch.Tensor, logits_B: torch.Tensor) -> torch.Tensor:
    A20, B20 = aa20(logits_A), aa20(logits_B)
    max_A = torch.max(A20, dim=-1, keepdim=True)[0]
    max_B = torch.max(B20, dim=-1, keepdim=True)[0]
    mask = (max_A >= max_B).float()
    return (mask * A20) + ((1 - mask) * B20)  # (1,L,20)


def winner_take_all(logits_A: torch.Tensor, logits_B: torch.Tensor, num_sequences: int, temperature: float) -> Set[str]:
    print(f"--- Winner-Take-All (T={temperature}) ---")
    wta_logits = get_wta_logits(logits_A, logits_B)
    probs = get_probs(wta_logits, temperature=temperature)
    return set(sample_sequences_from_probs(probs, num_sequences))


def gibbs_sampling(logits_A: torch.Tensor, logits_B: torch.Tensor, num_sequences: int, iterations: int, start_temp: float, end_temp: float, use_wta_init: bool) -> Set[str]:
    print(f"--- Gibbs Sampling ({iterations} passes) ---")
    A20, B20 = aa20(logits_A), aa20(logits_B)
    num_residues = A20.shape[1]
    device = A20.device
    generated_sequences = set()
    combined_logits = (A20 + B20) / 2.0  # (1,L,20)
    for i in range(num_sequences):
        if use_wta_init:
            if i == 0: print("Initializing Gibbs with WTA sequence.")
            initial_logits = get_wta_logits(A20, B20)
            current_indices = torch.argmax(initial_logits.squeeze(0), dim=-1)
        else:
            if i == 0: print("Initializing Gibbs with random sequence.")
            current_indices = torch.randint(0, 20, (num_residues,), device=device)
        total_steps = iterations * num_residues
        for step in range(total_steps):
            progress = step / max(1, total_steps - 1)
            current_temp = start_temp * (end_temp / max(1e-8, start_temp)) ** progress
            pos = torch.randint(0, num_residues, (1,)).item()
            probs_pos = get_probs(combined_logits[:, pos:pos+1, :], temperature=current_temp)
            new_aa_index = torch.multinomial(probs_pos.squeeze(0).squeeze(0), num_samples=1)
            current_indices[pos] = new_aa_index
        final_sequence = "".join([STANDARD_ALPHABET[int(i)] for i in current_indices])
        generated_sequences.add(final_sequence)
        if (i+1) % 10 == 0 or num_sequences < 10:
            print(f"Generated sequence {len(generated_sequences)}/{num_sequences}...")
    return generated_sequences


def gumbel_softmax_optimization(
    logits_list: List[torch.Tensor], 
    weights: List[float], 
    num_sequences: int, 
    steps: int, 
    lr: float, 
    initial_tau: float, 
    min_tau: float, 
    use_wta_init: bool,
    native_sequence: str,
    fixed_indices: Set[int]
) -> Set[str]:
    print("--- Gumbel-Softmax Optimization ---")
    if len(logits_list) < 2 and use_wta_init:
        raise ValueError("WTA initialization requires at least 2 logits files.")
    if len(logits_list) != len(weights):
        raise ValueError("Number of logits tensors must match number of weights.")

    l_A = aa20(logits_list[0])
    device = l_A.device
    num_residues, num_vocab = l_A.shape[1], l_A.shape[2]  # V==20

    if use_wta_init:
        print("Initializing with WTA logits.")
        l_B = aa20(logits_list[1])
        initial_logits = get_wta_logits(l_A, l_B)  # (1,L,20)
        sequence_logits = initial_logits.clone().detach().requires_grad_(True)
    else:
        print("Initializing with random logits.")
        sequence_logits = torch.randn(1, num_residues, num_vocab, device=device, requires_grad=True)

    # Fixed-site mask once
    fixed_mask = torch.zeros_like(sequence_logits)
    if fixed_indices:
        native_idx_per_pos = torch.tensor([STANDARD_ALPHABET.index(aa) for aa in native_sequence], device=device)
        for pos in fixed_indices:
            if 0 <= pos < num_residues:
                fixed_mask[0, pos, :] = -1e9
                fixed_mask[0, pos, native_idx_per_pos[pos]] = 0.0

    optimizer = torch.optim.Adam([sequence_logits], lr=lr)

    print("--- Optimizing ---")
    for i in range(steps):
        optimizer.zero_grad()
        tau = max(min_tau, initial_tau * math.exp(-i / max(1.0, (steps / 3.5))))
        masked_logits = sequence_logits + fixed_mask
        y_soft = F.gumbel_softmax(masked_logits, tau=tau, hard=True, dim=-1)
        total_score = torch.tensor(0.0, device=device)
        for logit, weight in zip(logits_list, weights):
            logp = F.log_softmax(aa20(logit), dim=-1)
            total_score += weight * torch.sum(y_soft * logp)
        loss = -total_score
        loss.backward()
        optimizer.step()
        if (i + 1) % 50 == 0:
            print(f"Step {i+1}/{steps} | Loss: {loss.item():.4f} | Tau: {tau:.3f}")

    print("--- Optimization Finished ---")
    final_learned_logits = sequence_logits.detach() + fixed_mask
    probs = get_probs(final_learned_logits, temperature=1.0)
    return set(sample_sequences_from_probs(probs, num_sequences))

# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-state protein sequences from logits files and enforce fixed residues.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # I/O and residue fixing
    parser.add_argument("--logits_files", type=str, nargs='+', required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--pdb_for_numbering", type=str, required=True)
    parser.add_argument("--chains_to_parse", type=str, default="")
    parser.add_argument("--fixed_residues", type=str, default="")

    # General strategy args
    parser.add_argument("--strategy", type=str, default="winner_take_all", choices=['weighted_average', 'product_of_experts', 'winner_take_all', 'gibbs', 'gumbel', 'all'])
    parser.add_argument("--num_sequences", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--add_scores", action='store_true')

    # Strategy-specific args
    parser.add_argument("--weights", type=float, nargs='+')
    parser.add_argument("--gibbs_iterations", type=int, default=20)
    parser.add_argument("--gibbs_start_temp", type=float, default=2.0)
    parser.add_argument("--gibbs_end_temp", type=float, default=0.1)
    parser.add_argument("--gibbs_wta_init", action='store_true')
    parser.add_argument("--gumbel_steps", type=int, default=200)
    parser.add_argument("--gumbel_lr", type=float, default=0.1)
    parser.add_argument("--gumbel_initial_tau", type=float, default=1.0)
    parser.add_argument("--gumbel_min_tau", type=float, default=0.1)
    parser.add_argument("--gumbel_wta_init", action='store_true')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # PDB & logits loading/slicing
    _, full_res_id_to_index, _ = get_pdb_info(args.pdb_for_numbering, chains_to_parse=None)
    chains_to_design = args.chains_to_parse.split(',') if args.chains_to_parse else []
    target_native_sequence, target_res_id_to_index, target_ordered_res_ids = get_pdb_info(args.pdb_for_numbering, chains_to_design)
    print(f"Loaded target sequence of length {len(target_native_sequence)} from chains: {'All' if not chains_to_design else ','.join(chains_to_design)}")

    try:
        slice_indices = [full_res_id_to_index[res_id] for res_id in target_ordered_res_ids]
    except KeyError as e:
        sys.exit(f"Residue ID {e} from parsed chain(s) not found in the full PDB structure map.")

    all_logits_full = []
    try:
        for f_path in args.logits_files:
            if not os.path.exists(f_path):
                sys.exit(f"Logits file not found: {f_path}")
            logits = torch.from_numpy(np.load(f_path)).to(device).float()
            if logits.dim() == 2:
                logits = logits.unsqueeze(0)
            if logits.shape[1] != len(full_res_id_to_index):
                sys.exit(f"Logits {f_path} has length {logits.shape[1]}, but full PDB has {len(full_res_id_to_index)} residues.")
            all_logits_full.append(logits)
    except Exception as e:
        sys.exit(f"Error loading logits files: {e}")

    all_logits = [logit[:, slice_indices, :] for logit in all_logits_full]
    print(f"Loaded & sliced {len(all_logits)} logits files -> shape {all_logits[0].shape} (B,L,V)")

    # Fixed residues (relative to sliced indexing)
    fixed_residue_ids = set(args.fixed_residues.split()) if args.fixed_residues else set()
    fixed_indices = {target_res_id_to_index[res_id] for res_id in fixed_residue_ids if res_id in target_res_id_to_index}
    if fixed_indices:
        print(f"Fixed positions: {sorted(list(fixed_indices))}")

    # Weights processing
    processed_weights = None
    need_weights = args.strategy in ['weighted_average', 'gumbel', 'all'] or args.add_scores
    if need_weights:
        if not args.weights:
            if args.add_scores and args.strategy == 'product_of_experts':
                processed_weights = [1.0/len(all_logits)] * len(all_logits)
                print(f"Using equal weights for scoring: {processed_weights}")
            else:
                sys.exit(f"--weights required for strategy '{args.strategy}' and/or when using --add_scores.")
        else:
            num_logits = len(all_logits)
            if len(args.weights) == 1 and num_logits == 2:
                wA = args.weights[0]
                if not (0.0 <= wA <= 1.0):
                    sys.exit(f"Single weight must be in [0,1], got {wA}")
                processed_weights = [wA, 1.0 - wA]
                print(f"Weights: {processed_weights}")
            elif len(args.weights) == num_logits:
                if not math.isclose(sum(args.weights), 1.0, rel_tol=1e-6, abs_tol=1e-6):
                    sys.exit(f"--weights must sum to 1.0 (got {sum(args.weights)})")
                processed_weights = args.weights
            else:
                sys.exit(f"Number of weights ({len(args.weights)}) must match number of logits files ({num_logits}), unless providing a single weight for exactly 2 files.")

    # Execute strategies
    final_sequences_by_strategy = defaultdict(set)

    def run_weighted_average():
        seqs = weighted_average(all_logits, processed_weights, args.num_sequences, args.temperature)
        final_sequences_by_strategy['weighted_average'].update(seqs)

    def run_product_of_experts():
        seqs = product_of_experts(all_logits, args.num_sequences, args.temperature)
        final_sequences_by_strategy['product_of_experts'].update(seqs)

    def run_winner_take_all():
        if len(all_logits) != 2:
            sys.exit("'winner_take_all' requires exactly 2 logits files.")
        seqs = winner_take_all(all_logits[0], all_logits[1], args.num_sequences, args.temperature)
        final_sequences_by_strategy['winner_take_all'].update(seqs)

    def run_gibbs():
        if len(all_logits) != 2:
            sys.exit("'gibbs' requires exactly 2 logits files.")
        seqs = gibbs_sampling(all_logits[0], all_logits[1], args.num_sequences, args.gibbs_iterations, args.gibbs_start_temp, args.gibbs_end_temp, args.gibbs_wta_init)
        final_sequences_by_strategy['gibbs_sampling'].update(seqs)

    def run_gumbel():
        if args.gumbel_wta_init and len(all_logits) != 2:
            sys.exit("--gumbel_wta_init requires exactly 2 logits files.")
        seqs = gumbel_softmax_optimization(
            all_logits, processed_weights, args.num_sequences, args.gumbel_steps,
            args.gumbel_lr, args.gumbel_initial_tau, args.gumbel_min_tau, args.gumbel_wta_init,
            target_native_sequence, fixed_indices
        )
        final_sequences_by_strategy['gumbel_softmax'].update(seqs)

    if args.strategy == 'weighted_average':
        run_weighted_average()
    elif args.strategy == 'product_of_experts':
        run_product_of_experts()
    elif args.strategy == 'winner_take_all':
        run_winner_take_all()
    elif args.strategy == 'gibbs':
        run_gibbs()
    elif args.strategy == 'gumbel':
        run_gumbel()
    elif args.strategy == 'all':
        print("--- Running all strategies ---")
        if len(all_logits) == 2:
            run_winner_take_all()
            run_gibbs()
        else:
            print("Skipping WTA/Gibbs (need exactly 2 logits files).")
        run_weighted_average()
        run_product_of_experts()
        run_gumbel()

    # Post-hoc fixed residues (safety net)
    for strategy_name, sequences in final_sequences_by_strategy.items():
        final_sequences_by_strategy[strategy_name] = apply_fixed_residues(sequences, target_native_sequence, fixed_indices)

    # Write FASTA
    output_path = args.out_file
    if not (output_path.endswith('.fasta') or output_path.endswith('.fa')):
        output_path += '.fasta'
        print(f"Appending .fasta extension, final output path: {output_path}")

    total_written_count = 0
    strategy_order = ['winner_take_all', 'gibbs_sampling', 'weighted_average', 'product_of_experts', 'gumbel_softmax']

    with open(output_path, 'w') as f:
        for strategy_name in strategy_order:
            if strategy_name in final_sequences_by_strategy:
                sequences_for_strategy = list(final_sequences_by_strategy[strategy_name])
                if args.add_scores and processed_weights:
                    scored_sequences = []
                    for seq in sequences_for_strategy:
                        score = calculate_sequence_score(seq, all_logits, processed_weights, device)
                        scored_sequences.append((score, seq))
                    scored_sequences.sort(key=lambda x: x[0], reverse=True)
                else:
                    sequences_for_strategy.sort()
                    scored_sequences = [(None, seq) for seq in sequences_for_strategy]
                limit = args.num_sequences if args.strategy != 'all' else len(scored_sequences)
                for i, (score, seq) in enumerate(scored_sequences):
                    if i >= limit:
                        break
                    total_written_count += 1
                    header = f">seq_{total_written_count}|strategy={strategy_name}"
                    if score is not None:
                        header += f"|score={score:.4f}"
                    header += f"|len={len(seq)}"
                    f.write(header + "\n")
                    f.write(seq + "\n")

    if total_written_count == 0:
        print("Warning: No sequences were generated. The output file will be empty.")
    print(f"\n✅ Wrote {total_written_count} sequences to {output_path}")


if __name__ == "__main__":
    main()
