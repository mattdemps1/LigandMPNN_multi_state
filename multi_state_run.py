import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import math
from typing import List, Set, Tuple, Dict
from collections import defaultdict

# Amino acid alphabet in the order ProteinMPNN uses
ALPHABET = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']
# Standard amino acids (excluding 'X')
STANDARD_ALPHABET = ALPHABET[:20]
AA_3_TO_1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
    'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
    'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

# --- [NEW] HELPER FUNCTIONS for PDB parsing and fixing residues ---

def get_pdb_info(pdb_path: str, chains_to_parse: List[str]) -> Tuple[str, Dict[str, int]]:
    """
    Parses a PDB file to get the native sequence and a map of residue IDs to indices.
    
    Args:
        pdb_path: Path to the PDB file.
        chains_to_parse: A list of chain IDs to include. If empty, all chains are used.

    Returns:
        A tuple containing:
        - native_sequence (str): The amino acid sequence from the PDB.
        - res_id_to_index (dict): A map from "A12" -> 0, "A13" -> 1, etc.
    """
    native_sequence = []
    res_id_to_index = {}
    
    seen_residues = set()
    current_index = -1

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                chain = line[21:22].strip()
                if not chains_to_parse or chain in chains_to_parse:
                    res_seq_str = line[22:26].strip()
                    if not res_seq_str: continue # Skip if residue number is empty
                    res_seq = int(res_seq_str)
                    res_name = line[17:20].strip()
                    atom_name = line[12:16].strip()
                    icode = line[26:27].strip()

                    if atom_name == "CA": # Use C-alpha atoms to define sequence
                        res_id_full = (chain, res_seq, icode)
                        if res_id_full not in seen_residues:
                            seen_residues.add(res_id_full)
                            current_index += 1
                            
                            if res_name in AA_3_TO_1:
                                native_sequence.append(AA_3_TO_1[res_name])
                                res_id_simple = f"{chain}{res_seq}{icode.strip()}"
                                res_id_to_index[res_id_simple] = current_index
                            else:
                                print(f"Warning: Skipping unrecognized residue '{res_name}' at {res_id_full}")

    return "".join(native_sequence), res_id_to_index


def apply_fixed_residues(
    sequences: Set[str], 
    native_sequence: str, 
    fixed_indices: Set[int]
) -> Set[str]:
    """
    Enforces fixed residues on a set of generated sequences.

    Args:
        sequences: A set of sequences to process.
        native_sequence: The original sequence from the PDB.
        fixed_indices: A set of 0-based indices that should be fixed.

    Returns:
        A new set of sequences with the native amino acids at the fixed positions.
    """
    if not fixed_indices:
        return sequences

    print(f"Applying fixed residues at {len(fixed_indices)} positions...")
    corrected_sequences = set()
    native_list = list(native_sequence)

    for seq in sequences:
        seq_list = list(seq)
        if len(seq_list) != len(native_list):
            print(f"Warning: Length mismatch between generated ({len(seq_list)}) and native ({len(native_list)}) sequences. Cannot apply fixed residues.")
            corrected_sequences.add(seq)
            continue
            
        for idx in fixed_indices:
            if idx < len(seq_list):
                seq_list[idx] = native_list[idx]
        
        corrected_sequences.add("".join(seq_list))
        
    return corrected_sequences


# --- Core Helper Functions ---

def get_probs(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Converts a logits tensor to a probability tensor using softmax."""
    if logits.dim() != 3:
        raise ValueError("Input logits tensor must have 3 dimensions (batch, seq_len, vocab_size)")
    return F.softmax(logits / temperature, dim=-1)

def sample_sequences_from_probs(probs: torch.Tensor, num_sequences: int) -> List[str]:
    """Samples multiple unique sequences from a probability distribution."""
    # Ensure probabilities are for standard AAs
    probs_20 = probs.squeeze(0)[:, :20]
    
    # Normalize probabilities to ensure they sum to 1, guarding against NaNs
    probs_sum = probs_20.sum(dim=-1, keepdim=True)
    # Add a small epsilon to avoid division by zero
    probs_20 = probs_20 / (probs_sum + 1e-9)

    # Use torch.multinomial to sample indices
    sampled_indices = torch.multinomial(probs_20, num_samples=num_sequences, replacement=True).T
    
    # Convert indices to sequences
    sequences = []
    for indices in sampled_indices:
        sequences.append("".join([STANDARD_ALPHABET[i] for i in indices]))
    return sequences

def logits_to_greedy_sequence(logits: torch.Tensor) -> str:
    """Generates the most likely sequence from logits (greedy decoding)."""
    indices = torch.argmax(logits.squeeze()[:, :20], dim=-1)
    return "".join([STANDARD_ALPHABET[i] for i in indices])

def calculate_sequence_score(sequence: str, logits_list: List[torch.Tensor], weights: List[float], device: torch.device) -> float:
    """
    Calculates the weighted-average log-likelihood score for a sequence.
    A higher (less negative) score is better.
    """
    # Convert sequence string to a tensor of indices
    try:
        indices = torch.tensor([STANDARD_ALPHABET.index(aa) for aa in sequence], device=device).long()
    except ValueError as e:
        print(f"Warning: Sequence '{sequence}' contains an invalid character. Skipping score calculation. Error: {e}")
        return -float('inf') # Return a very bad score

    total_weighted_log_prob = 0.0
    
    for i, logits in enumerate(logits_list):
        # Log-probabilities for the entire vocabulary at each position
        # Use T=1.0 for an unscaled log-likelihood score
        log_probs = F.log_softmax(logits / 1.0, dim=-1).squeeze(0) # Shape: [seq_len, vocab_size]
        
        # Gather the log-probabilities for the specific amino acids in the sequence
        # indices.view(-1, 1) reshapes indices to [seq_len, 1] for gather
        seq_log_probs = torch.gather(log_probs, 1, indices.view(-1, 1)).squeeze()
        
        # Sum log-probabilities to get the score for this state
        state_score = torch.sum(seq_log_probs).item()
        
        # Add the weighted score to the total
        total_weighted_log_prob += state_score * weights[i]
        
    return total_weighted_log_prob


# --- Sequence Design Strategies ---

def weighted_average(logits_list: List[torch.Tensor], weights: List[float], num_sequences: int, temperature: float) -> Set[str]:
    """
    Generates sequences from a weighted average of logits.
    """
    print(f"--- Running Weighted Average (T={temperature}) ---")
    if len(logits_list) != len(weights):
        raise ValueError("Number of logits tensors must match the number of weights.")
    
    # Calculate weighted average
    weighted_logits = torch.zeros_like(logits_list[0])
    for logit, weight in zip(logits_list, weights):
        weighted_logits += weight * logit
    
    # Get probabilities and sample
    probs = get_probs(weighted_logits, temperature=temperature)
    return set(sample_sequences_from_probs(probs, num_sequences))

def product_of_experts(logits_list: List[torch.Tensor], num_sequences: int, temperature: float) -> Set[str]:
    """
    Generates sequences from a product of probabilities (Product of Experts).
    This method finds a consensus where an amino acid must be favorable in ALL states.
    """
    print(f"--- Running Product of Experts (T={temperature}) ---")
    if not logits_list:
        return set()

    # Convert all logits to probabilities first
    # Use a temperature of 1.0 for the initial probability calculation
    prob_list = [get_probs(logits, temperature=1.0) for logits in logits_list]

    # Multiply probabilities element-wise
    # Start with the first probability tensor
    combined_probs = prob_list[0]
    for prob in prob_list[1:]:
        combined_probs *= prob

    # The result of multiplication is not a valid probability distribution,
    # so we convert it back to logits-space to apply temperature and re-normalize
    # Add a small epsilon to prevent log(0)
    combined_logits = torch.log(combined_probs + 1e-9)

    # Now, get the final probabilities using the desired sampling temperature
    final_probs = get_probs(combined_logits, temperature=temperature)
    
    return set(sample_sequences_from_probs(final_probs, num_sequences))


def winner_take_all(logits_A: torch.Tensor, logits_B: torch.Tensor, num_sequences: int, temperature: float) -> Set[str]:
    """
    For each position, chooses logits from the state with the higher max logit value.
    """
    print(f"--- Running Winner-Take-All (T={temperature}) ---")
    wta_logits = get_wta_logits(logits_A, logits_B)
    
    # Get probabilities and sample
    probs = get_probs(wta_logits, temperature=temperature)
    return set(sample_sequences_from_probs(probs, num_sequences))

def get_wta_logits(logits_A: torch.Tensor, logits_B: torch.Tensor) -> torch.Tensor:
    """Helper function to calculate Winner-Take-All logits."""
    max_A = torch.max(logits_A, dim=-1, keepdim=True)[0]
    max_B = torch.max(logits_B, dim=-1, keepdim=True)[0]
    mask = (max_A >= max_B).float()
    return (mask * logits_A) + ((1 - mask) * logits_B)

def gibbs_sampling(logits_A: torch.Tensor, logits_B: torch.Tensor, num_sequences: int, iterations: int, start_temp: float, end_temp: float, use_wta_init: bool) -> Set[str]:
    """
    Iteratively refines sequences using Gibbs sampling with temperature annealing.
    """
    print(f"--- Running Improved Gibbs Sampling ({iterations} passes) ---")
    num_residues = logits_A.shape[1]
    device = logits_A.device
    generated_sequences = set()

    # Combined logits for sampling probability calculation (hardcoded 50/50)
    combined_logits = (logits_A + logits_B) / 2.0

    for i in range(num_sequences):
        # Initialize sequence
        if use_wta_init:
            print("Initializing Gibbs with 'Winner-Take-All' sequence.") if i == 0 else None
            initial_logits = get_wta_logits(logits_A, logits_B)
            current_indices = torch.argmax(initial_logits.squeeze()[:, :20], dim=-1)
        else:
            print("Initializing Gibbs with random sequence.") if i == 0 else None
            current_indices = torch.randint(0, 20, (num_residues,), device=device)
        
        # Main sampling loop
        total_steps = iterations * num_residues
        for step in range(total_steps):
            progress = step / max(1, total_steps - 1)
            current_temp = start_temp * (end_temp / start_temp) ** progress
            
            pos = torch.randint(0, num_residues, (1,)).item()
            
            probs_pos = get_probs(combined_logits[:, pos:pos+1, :], temperature=current_temp)
            new_aa_index = torch.multinomial(probs_pos.squeeze(0).squeeze(0)[:20], num_samples=1)
            current_indices[pos] = new_aa_index
        
        final_sequence = "".join([STANDARD_ALPHABET[i] for i in current_indices])
        generated_sequences.add(final_sequence)
        if (i+1) % 10 == 0 or num_sequences < 10:
            print(f"Generated sequence {len(generated_sequences)}/{num_sequences}...")

    return generated_sequences

def gumbel_softmax_optimization(logits_list: List[torch.Tensor], weights: List[float], num_sequences: int, steps: int, lr: float, initial_tau: float, min_tau: float, use_wta_init: bool) -> Set[str]:
    """
    Finds an optimal sequence using Gumbel-Softmax and then samples from the result.
    """
    print("--- Running Gumbel-Softmax Optimization ---")
    if len(logits_list) < 2 and use_wta_init:
        raise ValueError("Winner-Take-All initialization requires at least 2 logits files.")
    if len(logits_list) != len(weights):
        raise ValueError("Number of logits tensors must match the number of weights for loss calculation.")

    l_A = logits_list[0]
    device = l_A.device
    num_residues, num_vocab = l_A.shape[1], l_A.shape[2]

    # 1. Initialization
    if use_wta_init:
        print("Initializing optimization with 'Winner-Take-All' logits.")
        l_B = logits_list[1]
        initial_logits = get_wta_logits(l_A, l_B)
        sequence_logits = initial_logits.clone().detach().requires_grad_(True)
    else:
        print("Initializing optimization with random logits.")
        sequence_logits = torch.randn(1, num_residues, num_vocab, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([sequence_logits], lr=lr)
    
    # 2. Optimization Loop
    print("--- Starting Optimization ---")
    for i in range(steps):
        optimizer.zero_grad()
        tau = max(min_tau, initial_tau * math.exp(-i / (steps / 3.5)))
        
        y_soft = F.gumbel_softmax(sequence_logits, tau=tau, hard=True, dim=-1)
        
        total_score = torch.tensor(0.0, device=device)
        for logit, weight in zip(logits_list, weights):
            total_score += weight * torch.sum(y_soft * logit)
            
        loss = -total_score
        loss.backward()
        optimizer.step()

        if (i + 1) % 50 == 0:
            print(f"Step {i+1}/{steps} | Loss: {loss.item():.4f} | Tau: {tau:.3f}")

    print("--- Optimization Finished ---")
    
    # 3. Sample from the final learned distribution
    final_learned_logits = sequence_logits.detach()
    probs = get_probs(final_learned_logits, temperature=1.0)
    return set(sample_sequences_from_probs(probs, num_sequences))


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-state protein sequences from logits files and enforce fixed residues.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- I/O and Residue Fixing Arguments ---
    parser.add_argument("--logits_files", type=str, nargs='+', required=True, help="Space-separated paths to the input .npy logits files.")
    parser.add_argument("--out_file", type=str, required=True, help="Path to the output FASTA file.")
    parser.add_argument("--pdb_for_numbering", type=str, required=True, help="Path to one of the original PDB files, used for residue numbering and native sequence.")
    parser.add_argument("--chains_to_parse", type=str, default="", help="Comma-separated list of chains to parse from the PDB (e.g., 'A,B'). If empty, all chains are used.")
    parser.add_argument("--fixed_residues", type=str, default="", help="Space-separated list of residues to fix (e.g., 'A12 A13 B42'). Must match the PDB.")

    # --- General Strategy Arguments ---
    parser.add_argument("--strategy", type=str, default="winner_take_all", choices=['weighted_average', 'product_of_experts', 'winner_take_all', 'gibbs', 'gumbel', 'all'], help="The sequence design strategy to use.")
    parser.add_argument("--num_sequences", type=int, default=10, help="Number of unique sequences to generate.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling. Lower is more greedy. Applies to all strategies except gumbel/gibbs internal temps.")
    parser.add_argument("--add_scores", action='store_true', help="Calculate and add a log-likelihood score to each sequence header. Requires --weights.")

    # --- Strategy-Specific Arguments ---
    parser.add_argument("--weights", type=float, nargs='+', help="Weights for 'weighted_average' and 'gumbel' strategies. For 2 files, a single value is the weight for the first file (e.g., 0.8). For >2 files, provide a weight for each file. Must sum to 1.")
    parser.add_argument("--gibbs_iterations", type=int, default=20, help="Number of passes for Gibbs sampling.")
    parser.add_argument("--gibbs_start_temp", type=float, default=2.0, help="Initial temperature for Gibbs sampling annealing.")
    parser.add_argument("--gibbs_end_temp", type=float, default=0.1, help="Final temperature for Gibbs sampling annealing.")
    parser.add_argument("--gibbs_wta_init", action='store_true', help="Initialize Gibbs sampling with the Winner-Take-All sequence.")
    parser.add_argument("--gumbel_steps", type=int, default=200, help="Number of optimization steps for Gumbel-Softmax.")
    parser.add_argument("--gumbel_lr", type=float, default=0.1, help="Learning rate for Gumbel-Softmax optimization.")
    parser.add_argument("--gumbel_initial_tau", type=float, default=1.0, help="Initial temperature (tau) for Gumbel-Softmax.")
    parser.add_argument("--gumbel_min_tau", type=float, default=0.1, help="Minimum temperature (tau) for Gumbel-Softmax.")
    parser.add_argument("--gumbel_wta_init", action='store_true', help="Initialize Gumbel optimization with Winner-Take-All logits. Requires exactly 2 logits files.")

    args = parser.parse_args()

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Parse PDB for sequence and residue mapping ---
    chains = args.chains_to_parse.split(',') if args.chains_to_parse else []
    native_sequence, res_id_to_index = get_pdb_info(args.pdb_for_numbering, chains)
    
    fixed_residue_ids = set(args.fixed_residues.split())
    fixed_indices = {res_id_to_index[res_id] for res_id in fixed_residue_ids if res_id in res_id_to_index}
    
    print(f"Loaded native sequence of length {len(native_sequence)} from {args.pdb_for_numbering}")
    if fixed_indices:
        print(f"Identified {len(fixed_indices)} fixed positions from input.")
    else:
        print("No fixed residues specified or found in the PDB.")

    # --- Load and Validate Logits ---
    all_logits = []
    try:
        for f_path in args.logits_files:
            if not os.path.exists(f_path):
                sys.exit(f"Error: Logits file not found at {f_path}")
            logits = torch.from_numpy(np.load(f_path)).to(device).float()
            if logits.dim() == 2: # Add batch dimension if missing
                logits = logits.unsqueeze(0)
            all_logits.append(logits)
        
        first_shape = all_logits[0].shape
        if len(native_sequence) != first_shape[1]:
                sys.exit(f"Error: Sequence length mismatch. PDB has {len(native_sequence)} residues but logits have length {first_shape[1]}. Check --chains_to_parse.")

        for i, logit in enumerate(all_logits[1:], 1):
            if logit.shape != first_shape:
                sys.exit(f"Error: Shape mismatch. Logits file {args.logits_files[0]} has shape {first_shape} but {args.logits_files[i]} has shape {logit.shape}.")
        print(f"Successfully loaded {len(all_logits)} logits files with shape {first_shape}.")
    except Exception as e:
        sys.exit(f"Error loading or validating logits files: {e}")

    # --- Process weights based on strategy ---
    processed_weights = None
    if args.strategy in ['weighted_average', 'gumbel', 'all'] or args.add_scores:
        if not args.weights:
            # For PoE scoring, we can create equal weights if none are provided
            if args.add_scores and args.strategy == 'product_of_experts':
                 processed_weights = [1.0/len(all_logits)] * len(all_logits)
                 print(f"Creating equal weights for scoring: {processed_weights}")
            else:
                sys.exit(f"Error: --weights must be provided for the '{args.strategy}' strategy and/or when using --add_scores.")
        else:
            num_logits = len(all_logits)
            if len(args.weights) == 1 and num_logits == 2:
                weight_A = args.weights[0]
                if not (0.0 <= weight_A <= 1.0):
                    sys.exit(f"Error: A single weight must be between 0.0 and 1.0. Got: {weight_A}")
                processed_weights = [weight_A, 1.0 - weight_A]
                print(f"Using single weight input. Calculated weights: {processed_weights}")
            elif len(args.weights) == num_logits:
                if not math.isclose(sum(args.weights), 1.0):
                    sys.exit(f"Error: --weights must sum to 1.0. Current sum: {sum(args.weights)}")
                processed_weights = args.weights
            else:
                sys.exit(f"Error: The number of weights ({len(args.weights)}) must match the number of logits files ({num_logits}), unless providing a single weight for exactly 2 files.")

    # --- Execute Strategies ---
    final_sequences_by_strategy = defaultdict(set)

    def run_weighted_average():
        seqs = weighted_average(all_logits, processed_weights, args.num_sequences, args.temperature)
        final_sequences_by_strategy['weighted_average'].update(seqs)
    
    def run_product_of_experts():
        seqs = product_of_experts(all_logits, args.num_sequences, args.temperature)
        final_sequences_by_strategy['product_of_experts'].update(seqs)

    def run_winner_take_all():
        if len(all_logits) != 2:
            sys.exit("Error: 'winner_take_all' strategy requires exactly 2 logits files.")
        seqs = winner_take_all(all_logits[0], all_logits[1], args.num_sequences, args.temperature)
        final_sequences_by_strategy['winner_take_all'].update(seqs)
        
    def run_gibbs():
        if len(all_logits) != 2:
            sys.exit("Error: 'gibbs' strategy requires exactly 2 logits files.")
        seqs = gibbs_sampling(all_logits[0], all_logits[1], args.num_sequences, args.gibbs_iterations, args.gibbs_start_temp, args.gibbs_end_temp, args.gibbs_wta_init)
        final_sequences_by_strategy['gibbs_sampling'].update(seqs)

    def run_gumbel():
        if args.gumbel_wta_init and len(all_logits) != 2:
            sys.exit("Error: --gumbel_wta_init requires exactly 2 logits files.")
        seqs = gumbel_softmax_optimization(all_logits, processed_weights, args.num_sequences, args.gumbel_steps, args.gumbel_lr, args.gumbel_initial_tau, args.gumbel_min_tau, args.gumbel_wta_init)
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
            print("Skipping 'winner_take_all' and 'gibbs' as they require exactly 2 logits files.")
        run_weighted_average()
        run_product_of_experts()
        run_gumbel()

    # --- Apply fixed residues to all generated sequences ---
    for strategy_name, sequences in final_sequences_by_strategy.items():
        corrected_sequences = apply_fixed_residues(sequences, native_sequence, fixed_indices)
        final_sequences_by_strategy[strategy_name] = corrected_sequences


    # --- Write Output FASTA File ---
    output_path = args.out_file
    if not (output_path.endswith(".fasta") or output_path.endswith(".fa")):
        output_path += ".fasta"
        print(f"Appending .fasta extension, final output path: {output_path}")

    total_written_count = 0
    strategy_order = ['winner_take_all', 'gibbs_sampling', 'weighted_average', 'product_of_experts', 'gumbel_softmax']

    with open(output_path, 'w') as f:
        for strategy_name in strategy_order:
            if strategy_name in final_sequences_by_strategy:
                sequences_for_strategy = list(final_sequences_by_strategy[strategy_name])
                
                # --- SCORING AND SORTING LOGIC ---
                if args.add_scores and processed_weights:
                    scored_sequences = []
                    for seq in sequences_for_strategy:
                        score = calculate_sequence_score(seq, all_logits, processed_weights, device)
                        scored_sequences.append((score, seq))
                    
                    scored_sequences.sort(key=lambda x: x[0], reverse=True)
                else:
                    sequences_for_strategy.sort()
                    scored_sequences = [(None, seq) for seq in sequences_for_strategy]
                # ------------------------------------

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
    
    print(f"\n✅ Successfully wrote {total_written_count} sequences to {output_path}")


if __name__ == "__main__":
    main()
