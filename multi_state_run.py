import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import math
from typing import List, Set, Tuple
from collections import defaultdict

# Amino acid alphabet in the order ProteinMPNN uses
ALPHABET = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']
# Standard amino acids (excluding 'X')
STANDARD_ALPHABET = ALPHABET[:20]

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
    probs_20 = probs_20 / torch.where(probs_sum > 0, probs_sum, torch.ones_like(probs_sum))

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

    This method combines the logits from multiple conformational states into a single
    set of logits by taking a weighted average. It's a simple way to bias the
    sequence generation towards states with higher weights.

    Args:
        logits_list: A list of logits tensors, one for each state.
        weights: A list of floats representing the weight for each state.
        num_sequences: The number of sequences to generate.
        temperature: The sampling temperature.

    Returns:
        A set of unique generated sequences.
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

def winner_take_all(logits_A: torch.Tensor, logits_B: torch.Tensor, num_sequences: int, temperature: float) -> Set[str]:
    """
    For each position, chooses logits from the state with the higher max logit value.

    This is a hard-switching method. For each residue position, it compares the
    highest logit value in state A vs. state B. It then uses the *entire* set of
    21 logits from whichever state "won" that position. This can be useful for
    creating sequences that are mosaics of the most confident predictions from each state.

    Args:
        logits_A: Logits tensor for the first state.
        logits_B: Logits tensor for the second state.
        num_sequences: The number of sequences to generate.
        temperature: The sampling temperature.

    Returns:
        A set of unique generated sequences.
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

    This is a stochastic search method. It starts with a sequence (either random or
    from WTA) and iteratively tries to improve it by mutating one residue at a time.
    The decision to accept a mutation is probabilistic, guided by the logits and a
    "temperature" that cools over time, causing the search to settle on a solution.

    Args:
        logits_A: Logits tensor for the first state.
        logits_B: Logits tensor for the second state.
        num_sequences: The number of independent sampling runs to perform.
        iterations: The number of full passes over the sequence for refinement.
        start_temp: The initial high temperature for sampling.
        end_temp: The final low temperature for sampling.
        use_wta_init: If True, start from the WTA sequence instead of a random one.

    Returns:
        A set of unique generated sequences.
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
            
            # Slice with a range (e.g., pos:pos+1) to keep the dimension,
            # ensuring the tensor remains 3D for get_probs().
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

    This is a gradient-based optimization method. It learns a new set of logits that
    maximizes a weighted score across all input states. The Gumbel-Softmax trick
    allows gradients to flow through the discrete choice of amino acids. The final
    result is a probability distribution from which sequences are sampled.

    Args:
        logits_list: A list of logits tensors, one for each state.
        weights: A list of floats representing the weight for each state's score.
        num_sequences: The number of sequences to sample from the final distribution.
        steps: The number of optimization steps.
        lr: The learning rate for the Adam optimizer.
        initial_tau: The initial high temperature for the Gumbel-Softmax.
        min_tau: The final low temperature for the Gumbel-Softmax.
        use_wta_init: If True, start from the WTA logits instead of random ones.

    Returns:
        A set of unique generated sequences.
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
    """
    Generate protein sequences from multiple ProteinMPNN logits files.

    This script takes as input two or more .npy files, each containing the output
    logits from a ProteinMPNN run for a different conformational state of a protein
    (e.g., ligand-bound vs. unbound). It then uses one of several design strategies
    to generate new sequences that are optimized to be favorable across these states.

    The output is a FASTA file containing the generated unique sequences.
    """
    parser = argparse.ArgumentParser(
        description="Generate protein sequences from multiple logits files using various design strategies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- I/O Arguments ---
    parser.add_argument("--logits_files", type=str, nargs='+', required=True, help="Space-separated paths to the input .npy logits files.")
    parser.add_argument("--out_file", type=str, required=True, help="Path to the output file. The .fasta extension will be added automatically if not present.")
    
    # --- General Strategy Arguments ---
    parser.add_argument("--strategy", type=str, default="winner_take_all", choices=['weighted_average', 'winner_take_all', 'gibbs', 'gumbel', 'all'], help="The sequence design strategy to use.")
    parser.add_argument("--num_sequences", type=int, default=10, help="Number of unique sequences to generate.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling. Lower is more greedy. Applies to 'weighted_average' and 'winner_take_all'.")
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
             sys.exit(f"Error: --weights must be provided for the '{args.strategy}' strategy and/or when using --add_scores.")
        
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
        run_gumbel()

    # --- Write Output FASTA File ---
    output_path = args.out_file
    if not (output_path.endswith(".fasta") or output_path.endswith(".fa")):
        output_path += ".fasta"
        print(f"Appending .fasta extension, final output path: {output_path}")

    total_written_count = 0
    strategy_order = ['winner_take_all', 'gibbs_sampling', 'weighted_average', 'gumbel_softmax']

    with open(output_path, 'w') as f:
        for strategy_name in strategy_order:
            if strategy_name in final_sequences_by_strategy:
                sequences_for_strategy = list(final_sequences_by_strategy[strategy_name])
                
                # --- SCORING AND SORTING LOGIC ---
                if args.add_scores:
                    # Calculate scores and create a list of (score, seq) tuples
                    scored_sequences = []
                    for seq in sequences_for_strategy:
                        score = calculate_sequence_score(seq, all_logits, processed_weights, device)
                        scored_sequences.append((score, seq))
                    
                    # Sort by score in descending order (higher is better)
                    scored_sequences.sort(key=lambda x: x[0], reverse=True)
                else:
                    # If not scoring, just sort alphabetically
                    sequences_for_strategy.sort()
                    # Create a list of (None, seq) tuples to fit the loop structure
                    scored_sequences = [(None, seq) for seq in sequences_for_strategy]
                # ------------------------------------

                # When running a single strategy, limit output. When 'all', write all found.
                limit = args.num_sequences if args.strategy != 'all' else len(scored_sequences)

                for i, (score, seq) in enumerate(scored_sequences):
                    if i >= limit:
                        break
                    total_written_count += 1
                    header = f">seq_{total_written_count}|strategy={strategy_name}"
                    if score is not None:
                        header += f"|score={score:.4f}" # Format score to 4 decimal places
                    header += f"|len={len(seq)}"
                    
                    f.write(header + "\n")
                    f.write(seq + "\n")

    if total_written_count == 0:
        print("Warning: No sequences were generated. The output file will be empty.")
    
    print(f"\n✅ Successfully wrote {total_written_count} sequences to {output_path}")


if __name__ == "__main__":
    main()
