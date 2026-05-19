from utils.config_loader import Config
from models.base_model import AdamantanePeptideModel
from utils.dataset import HybridDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import List, Tuple, Dict
import random
from copy import deepcopy
import os
import argparse

# ==========================================
# Sequence generator: gradient optimization + MCMC + diversity filtering
# ==========================================
class PeptideGenerator:
    def __init__(self, 
                 model,
                 mol_tokenizer,
                 prot_tokenizer,
                 device='cuda',
                 amino_acids='ACDEFGHIKLMNPQRSTVWY',
                 fixed_positions: Dict[int, str] = None):
        """
        model: trained AdamantanePeptideModel
        mol_tokenizer: Molformer tokenizer
        prot_tokenizer: ESM-2 tokenizer
        amino_acids: allowed amino acid character set
        """
        self.model = model
        self.mol_tokenizer = mol_tokenizer
        self.prot_tokenizer = prot_tokenizer
        self.device = device
        self.amino_acids = list(amino_acids)
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.amino_acids)}
        
        # Freeze model parameters.
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        
        # Get EOS token IDs.
        self.mol_eos_id = self.model.mol_eos_id
        self.prot_eos_id = self.model.prot_eos_id
        
        self.fixed_positions = fixed_positions if fixed_positions else {}
        
        # Validate amino acids at fixed positions.
        for pos, aa in self.fixed_positions.items():
            if aa not in self.amino_acids:
                raise ValueError(f"Amino acid '{aa}' at fixed position {pos} is not allowed")
        
        if self.fixed_positions:
            print(f"[Constraint] Fixed positions: {self.fixed_positions}")
    
    # ==========================================
    # Method 1: MCMC sampling (Metropolis-Hastings)
    # ==========================================
    def mcmc_sampling(self,
                     adamantane_smiles: str,
                     initial_seq: str = None,
                     seq_length: int = 8,
                     n_iterations: int = 500,
                     temperature: float = 0.5,
                     mutation_rate: float = 0.3,
                     verbose: bool = True):
        """
        Use Metropolis-Hastings MCMC to sample high-activity sequences.
        
        Args:
            adamantane_smiles: adamantane SMILES string
            initial_seq: initial peptide sequence; randomly generated if None
            seq_length: peptide length
            n_iterations: number of MCMC iterations
            temperature: temperature parameter; lower values are greedier
            mutation_rate: amino acid mutation ratio per mutation step
            verbose: whether to print progress
            
        Returns:
            best_seq: best sequence
            best_score: best score
            accepted_samples: all accepted samples [(seq, score), ...]
        """
        if verbose:
            print("\n" + "="*60)
            print("[MCMC Sampling] Starting...")
            if self.fixed_positions:
                print(f"[Constraint] Fixed positions: {self.fixed_positions}")
            print("="*60)
        
        # Initialize sequence with constraints applied.
        if initial_seq is None:
            current_seq = self._generate_constrained_sequence(seq_length)
        else:
            current_seq = self._apply_constraints(initial_seq, seq_length)
        
        current_score = self._evaluate_sequence(adamantane_smiles, current_seq)
        
        best_seq = current_seq
        best_score = current_score
        accepted_samples = [(current_seq, current_score)]
        
        n_accepted = 0
        
        for iteration in range(n_iterations):
            # 1. Propose a new sequence with random mutations.
            proposed_seq = self._mutate_sequence(current_seq, mutation_rate)
            proposed_score = self._evaluate_sequence(adamantane_smiles, proposed_seq)
            
            # 2. Compute the acceptance probability with the Metropolis criterion.
            delta = (proposed_score - current_score) / temperature
            accept_prob = min(1.0, np.exp(delta))
            
            # 3. Accept or reject.
            if random.random() < accept_prob:
                current_seq = proposed_seq
                current_score = proposed_score
                accepted_samples.append((current_seq, current_score))
                n_accepted += 1
                
                if current_score > best_score:
                    best_score = current_score
                    best_seq = current_seq
                    if verbose and iteration % 50 == 0:
                        print(f"  Iter {iteration:3d}: {current_seq} -> Score: {current_score:.4f} (new best)")
        
        if verbose:
            print(f"\nAcceptance rate: {n_accepted/n_iterations*100:.1f}%")
            print(f"Best sequence: {best_seq} -> Score: {best_score:.4f}")
        
        return best_seq, best_score, accepted_samples
    
    # ==========================================
    # Method 2: Gradient-based sequence optimization
    # ==========================================
    def gradient_based_optimization(self, 
                                    adamantane_smiles: str,
                                    initial_seq: str = None,
                                    seq_length: int = 8,
                                    n_iterations: int = 100,
                                    lr: float = 0.5,
                                    inner_steps: int = 3,
                                    verbose: bool = True):
        """
        Optimize sequence embeddings with gradient ascent.
        
        Strategy:
        1. Convert the sequence into learnable embeddings.
        2. Maximize predicted activity with gradient ascent.
        3. Map embeddings back to the nearest amino acids using a Gumbel-Softmax relaxation.
        
        Args:
            adamantane_smiles: adamantane SMILES
            initial_seq: initial sequence
            seq_length: peptide length
            n_iterations: number of outer iterations
            lr: learning rate
            inner_steps: number of inner gradient steps per outer iteration
            verbose: whether to print progress
        """
        if verbose:
            print("\n" + "="*60)
            print("[Gradient Optimization] Starting...")
            if self.fixed_positions:
                print(f"[Constraint] Fixed positions: {self.fixed_positions}")
            print("="*60)
        
        # Initialize sequence with constraints applied.
        if initial_seq is None:
            current_seq = self._generate_constrained_sequence(seq_length)
        else:
            current_seq = self._apply_constraints(initial_seq, seq_length)
        
        # Tokenize adamantane, which remains fixed.
        mol_inputs = self.mol_tokenizer(
            adamantane_smiles, 
            padding='max_length', 
            truncation=True, 
            max_length=50, 
            return_tensors="pt"
        )
        mol_ids = mol_inputs['input_ids'].to(self.device)
        mol_mask = mol_inputs['attention_mask'].to(self.device)
        
        best_seq = current_seq
        best_score = self._evaluate_sequence(adamantane_smiles, current_seq)
        
        # Get the ESM-2 embedding layer.
        try:
            embedding_layer = self.model.prot_model.esm.embeddings.word_embeddings
        except AttributeError:
            try:
                embedding_layer = self.model.prot_model.embeddings.word_embeddings
            except AttributeError:
                raise RuntimeError("Could not find the ESM-2 embedding layer")
        
        for iteration in range(n_iterations):
            # 1. Get token IDs and original embeddings for the current sequence.
            prot_inputs = self.prot_tokenizer(
                current_seq, 
                padding='max_length', 
                truncation=True, 
                max_length=30, 
                return_tensors="pt"
            )
            prot_ids = prot_inputs['input_ids'].to(self.device)
            prot_mask = prot_inputs['attention_mask'].to(self.device)
            
            # Validate tokenization correctness.
            decoded_tokens = self.prot_tokenizer.convert_ids_to_tokens(prot_ids[0])
            
            # Find real amino acid token positions, skipping special tokens.
            aa_token_indices = []
            for idx, token in enumerate(decoded_tokens):
                # ESM-2 amino acid tokens are usually single characters.
                if len(token) == 1 and token in self.amino_acids:
                    aa_token_indices.append(idx)
            
            # Validate that the amino acid count matches sequence length.
            if len(aa_token_indices) != len(current_seq):
                print(f"Warning: tokenization mismatch! Sequence length={len(current_seq)}, "
                    f"token count={len(aa_token_indices)}")
                print(f"Tokens: {decoded_tokens[:15]}")
                # Fall back to random mutation.
                current_seq = self._mutate_sequence(best_seq, 0.2)
                continue
            
            
            with torch.no_grad():
                original_embedding = embedding_layer(prot_ids)  # (1, seq_len, hidden_dim)
            
            # 2. Create learnable embeddings, optimizing only valid token positions.
            # Use clean_mask to identify valid amino acid positions.
            clean_mask = self.model.create_clean_mask(prot_ids, prot_mask, self.prot_eos_id)
            
            # Copy embeddings and enable gradients.
            optimizable_embedding = original_embedding.clone().detach()
            optimizable_embedding.requires_grad = True
            
            optimizer = torch.optim.Adam([optimizable_embedding], lr=lr)
            
            # 3. Inner optimization loop.
            for step in range(inner_steps):
                optimizer.zero_grad()
                
                # Manual forward pass, bypassing the tokenizer and using embeddings directly.
                score = self._forward_with_embedding(
                    mol_ids, mol_mask, 
                    optimizable_embedding, prot_mask, prot_ids
                )
                
                # Maximize score with gradient ascent.
                loss = -score.mean()
                loss.backward()
                
                # Only update gradients at valid positions.
                with torch.no_grad():
                    mask_expanded = clean_mask.unsqueeze(-1).float()
                    if optimizable_embedding.grad is not None:
                        optimizable_embedding.grad *= mask_expanded
                
                optimizer.step()
            
            # 4. Map optimized embeddings back to an amino acid sequence.
            with torch.no_grad():
                # Get embeddings for all amino acids.
                all_aa_tokens = [self.prot_tokenizer.encode(aa, add_special_tokens=False)[0] 
                                for aa in self.amino_acids]
                all_aa_ids = torch.tensor(all_aa_tokens).to(self.device)
                all_aa_embeddings = embedding_layer(all_aa_ids)  # (20, hidden_dim)
                
                # Find valid token positions.
                # valid_positions = torch.where(clean_mask[0])[0]
                
                # For each valid position, find the nearest amino acid.
                new_seq_list = []
                for seq_pos, token_idx in enumerate(aa_token_indices):
                    if seq_pos >= seq_length:
                        break
                    
                    # Check whether this is a fixed position.
                    if seq_pos in self.fixed_positions:
                        # Use the specified amino acid for fixed positions.
                        new_seq_list.append(self.fixed_positions[seq_pos])
                    else:
                        # Use gradient optimization results for variable positions.
                        pos_embedding = optimizable_embedding[0, token_idx]
                        
                        # Compute cosine similarity to all amino acids.
                        pos_norm = F.normalize(pos_embedding.unsqueeze(0), dim=-1)
                        aa_norm = F.normalize(all_aa_embeddings, dim=-1)
                        similarities = torch.mm(pos_norm, aa_norm.T)  # (1, 20)
                        
                        # Select the most similar amino acid.
                        best_aa_idx = similarities.argmax().item()
                        new_seq_list.append(self.amino_acids[best_aa_idx])
                
                # If the generated sequence is too short, fill from the original sequence.
                while len(new_seq_list) < seq_length:
                    new_seq_list.append(current_seq[len(new_seq_list)])
                
                new_seq = ''.join(new_seq_list[:seq_length])
            
            # 5. Evaluate the new sequence.
            new_score = self._evaluate_sequence(adamantane_smiles, new_seq)
            
            if new_score > best_score:
                best_score = new_score
                best_seq = new_seq
                if verbose and iteration % 20 == 0:
                    print(f"  Iter {iteration:3d}: {new_seq} -> Score: {new_score:.4f} ✓")
            
            # Update the current sequence with some randomness to avoid getting stuck.
            if random.random() < 0.8:  # 80% chance to accept the new sequence
                current_seq = new_seq
            else:
                current_seq = self._mutate_sequence(best_seq, 0.2)
        
        if verbose:
            print(f"\nBest sequence: {best_seq} -> Score: {best_score:.4f}")
        
        return best_seq, best_score
    
    # ==========================================
    # Ensemble method: MCMC + gradient optimization + diversity filtering
    # ==========================================
    def ensemble_generation(self,
                           adamantane_smiles: str,
                           n_candidates: int = 50,
                           seq_length: int = 8,
                           min_score_threshold: float = 0.8,
                           diversity_threshold: float = 0.2,
                           n_mcmc_runs: int = 5,
                           mcmc_iterations: int = 200,
                           n_top_for_refinement: int = 10,
                           gradient_iterations: int = 50):
        """
        Generate high-quality, diverse peptide sequences with a hybrid strategy.
        
        Workflow:
        1. Run multiple MCMC sampling rounds to build a candidate pool.
        2. Select top candidates for gradient optimization refinement.
        3. Filter for diversity to remove similar sequences.
        4. Return the final candidate list.
        
        Args:
            adamantane_smiles: adamantane SMILES
            n_candidates: number of final candidates to return
            seq_length: peptide length
            diversity_threshold: sequence similarity threshold; above this is treated as duplicate
            n_mcmc_runs: number of independent MCMC runs
            mcmc_iterations: number of iterations per MCMC run
            n_top_for_refinement: number of candidates selected for gradient optimization
            gradient_iterations: number of gradient optimization iterations
        """
        print("\n" + "="*80)
        print("Ensemble generation strategy: MCMC sampling -> gradient refinement -> diversity filtering")
        print("="*80)
        
        all_candidates = []
        
        # ==========================================
        # Stage 1: Generate a candidate pool with MCMC.
        # ==========================================
        print("\n[Stage 1/3] Generating candidate pool with MCMC sampling...")
        print("-"*80)
        
        for run in range(n_mcmc_runs):
            print(f"\nMCMC Run {run+1}/{n_mcmc_runs}")
            
            # Use a different initial sequence for each run.
            initial_seq = ''.join(random.choices(self.amino_acids, k=seq_length))
            
            _, _, samples = self.mcmc_sampling(
                adamantane_smiles,
                initial_seq=initial_seq,
                seq_length=seq_length,
                n_iterations=mcmc_iterations,
                temperature=0.3,
                mutation_rate=0.2,
                verbose=False
            )
            
            high_quality_samples = [(seq, score) for seq, score in samples 
                               if score >= min_score_threshold]
        
            for seq, score in high_quality_samples:
                if not any(s == seq for s, _ in all_candidates):
                    all_candidates.append((seq, score))
            
            print(f"  Generated {len(samples)} samples, with {len(high_quality_samples)} above the threshold")
            print(f"  Current candidate pool: {len(all_candidates)} high-quality sequences")
        
        print(f"\nStage 1 complete. Candidate pool contains {len(all_candidates)} sequences")
        
        # ==========================================
        # Stage 2: Refine top candidates with gradient optimization.
        # ==========================================
        print("\n[Stage 2/3] Refining top candidates with gradient optimization...")
        print("-"*80)
        
        # Sort by score and select top candidates.
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = all_candidates[:n_top_for_refinement]
        
        optimized = []
        for i, (seq, score) in enumerate(top_candidates, 1):
            print(f"\nOptimizing candidate {i}/{n_top_for_refinement}: {seq} (Score: {score:.4f})")
            
            opt_seq, opt_score = self.gradient_based_optimization(
                adamantane_smiles,
                initial_seq=seq,
                seq_length=seq_length,
                n_iterations=gradient_iterations,
                lr=0.5,
                inner_steps=3,
                verbose=False
            )
            
            if opt_score >= min_score_threshold:
                optimized.append((opt_seq, opt_score))
                print(f"  After optimization: {opt_seq} (Score: {opt_score:.4f})")
            else:
                print(f"  After optimization: {opt_seq} (Score: {opt_score:.4f}) below threshold")
        
        # Merge original and optimized candidates.
        all_candidates.extend(optimized)
        all_candidates = list(set(all_candidates))  # Deduplicate.
        
        all_candidates = [(seq, score) for seq, score in all_candidates 
                        if score >= min_score_threshold]
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nStage 2 complete. Current candidate pool: {len(all_candidates)} sequences")
        
        # ==========================================
        # Stage 3: Diversity filtering.
        # ==========================================
        print("\n[Stage 3/3] Diversity filtering...")
        print("-"*80)
        
        final_candidates = self._diversity_filter(
            all_candidates, 
            diversity_threshold,
            max_candidates=n_candidates
        )
        
        print(f"\nGenerated {len(final_candidates)} diverse final candidate sequences")
        
        # Sort by score.
        final_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return final_candidates
    
    # ==========================================
    # Helper functions
    # ==========================================
    def _evaluate_sequence(self, smiles: str, sequence: str) -> float:
        """
        Evaluate the predicted activity of a single sequence.
        Correctly handle masks by removing CLS, EOS, and PAD tokens.
        """
        mol_inputs = self.mol_tokenizer(
            smiles, 
            padding='max_length', 
            truncation=True, 
            max_length=50, 
            return_tensors="pt"
        )
        prot_inputs = self.prot_tokenizer(
            sequence, 
            padding='max_length', 
            truncation=True, 
            max_length=30, 
            return_tensors="pt"
        )
        
        with torch.no_grad():
            pred = self.model(
                mol_inputs['input_ids'].to(self.device),
                mol_inputs['attention_mask'].to(self.device),
                prot_inputs['input_ids'].to(self.device),
                prot_inputs['attention_mask'].to(self.device)
            )
        return pred.item()
    
    def _forward_with_embedding(self, mol_ids, mol_mask, prot_embedding, prot_mask, prot_ids):
        """
        Run a forward pass with custom peptide embeddings.
        Adapted to the newer model architecture (TransformerFusion).
        """
        # 1. Extract adamantane features.
        mol_outputs = self.model.mol_model(
            input_ids=mol_ids, 
            attention_mask=mol_mask
        )
        mol_clean_mask = self.model.create_clean_mask(mol_ids, mol_mask, self.mol_eos_id)
        mol_emb = self.model.mol_embedding(mol_outputs, mol_clean_mask)
        
        # 2. Extract peptide features using custom embeddings.
        # Manually pass through the ESM-2 encoder.
        prot_outputs = self.model.prot_model(
            inputs_embeds=prot_embedding,
            attention_mask=prot_mask
        )
        prot_clean_mask = self.model.create_clean_mask(prot_ids, prot_mask, self.prot_eos_id)
        
        # Extract features according to the model embedding type.
        prot_emb_type = self.model.config.get('model.prot_embedding.type', 'transformer')
        
        if prot_emb_type == "transformer":
            # Return full sequence representations.
            prot_emb = prot_outputs.last_hidden_state
        elif prot_emb_type in ["mean", "max", "attention"]:
            # Apply pooling.
            from models.embeddings import ProteinEmbedding
            pooling_layer = ProteinEmbedding(prot_emb_type, self.model.prot_dim)
            prot_emb = pooling_layer(prot_outputs, prot_clean_mask)
        else:
            prot_emb = prot_outputs.last_hidden_state
        
        # 3. Project to a shared dimension.
        if len(mol_emb.shape) == 2:  # (B, D)
            mol_feat = self.model.mol_projector(mol_emb).unsqueeze(1)  # (B, 1, D)
        else:  # (B, L, D)
            mol_feat = self.model.mol_projector(mol_emb)
        
        if len(prot_emb.shape) == 2:  # (B, D)
            prot_feat = self.model.prot_projector(prot_emb).unsqueeze(1)  # (B, 1, D)
        else:  # (B, L, D)
            prot_feat = self.model.prot_projector(prot_emb)
        
        # 4. Fuse features.
        combined_feat = torch.cat([mol_feat, prot_feat], dim=1)
        batch_size = mol_feat.shape[0]
        
        mol_valid_mask = torch.ones(batch_size, mol_feat.shape[1]).to(mol_feat.device)
        combined_mask = torch.cat([mol_valid_mask, prot_clean_mask], dim=1)
        
        # Use the model's fusion module (TransformerFusion).
        fused_feat = self.model.fusion(combined_feat, combined_mask)
        
        # 5. Pooling.
        pooled_output = self.model.pooling(fused_feat, mask=combined_mask)
        
        # 6. Prediction.
        prediction = self.model.regressor(pooled_output)
        
        return prediction
    
    def _mutate_sequence(self, seq: str, mutation_rate: float) -> str:
        """
        Randomly mutate a sequence, skipping fixed positions.
        
        Args:
            seq: original sequence
            mutation_rate: mutation ratio
            
        Returns:
            mutated sequence
        """
        seq_list = list(seq)
        
        # Get mutable positions, excluding fixed positions.
        mutable_positions = self._get_mutable_positions(len(seq))
        
        if not mutable_positions:
            return seq  # All positions are fixed and cannot mutate.
        
        # Compute the number of mutations.
        n_mutations = max(1, int(len(mutable_positions) * mutation_rate))
        n_mutations = min(n_mutations, len(mutable_positions))
        
        # Randomly select mutable positions.
        positions = random.sample(mutable_positions, n_mutations)
        
        for pos in positions:
            seq_list[pos] = random.choice(self.amino_acids)
        
        return ''.join(seq_list)
    
    def _diversity_filter(self, 
                         candidates: List[Tuple[str, float]], 
                         threshold: float,
                         max_candidates: int = 50) -> List[Tuple[str, float]]:
        """
        Filter similar sequences to ensure diversity.
        Use a greedy algorithm: keep high-scoring sequences first and reject candidates too similar to selected ones.
        """
        if not candidates:
            return []
        
        # Sort by score.
        sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        
        filtered = [sorted_candidates[0]]  # Keep the best sequence.
        
        for seq, score in sorted_candidates[1:]:
            if len(filtered) >= max_candidates:
                break
            
            # Check whether this sequence is too similar to selected sequences.
            is_diverse = True
            for existing_seq, _ in filtered:
                similarity = self._sequence_similarity(seq, existing_seq)
                if similarity > (1 - threshold):  # Similarity > (1 - threshold) is treated as duplicate.
                    is_diverse = False
                    break
            
            if is_diverse:
                filtered.append((seq, score))
        
        return filtered
    
    def _sequence_similarity(self, seq1: str, seq2: str) -> float:
        """
        Compute sequence similarity using normalized edit distance.
        Return value ranges from 0.0 (completely different) to 1.0 (identical).
        """
        if len(seq1) != len(seq2):
            # Use edit distance for different lengths.
            return 1.0 - (self._levenshtein_distance(seq1, seq2) / max(len(seq1), len(seq2)))
        
        # For equal lengths, compute the match ratio directly.
        matches = sum(a == b for a, b in zip(seq1, seq2))
        return matches / len(seq1)
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Compute edit distance."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _apply_constraints(self, sequence: str, seq_length: int) -> str:
        """
        Apply positional constraints to a sequence.
        
        Args:
            sequence: original sequence
            seq_length: target length
            
        Returns:
            constrained sequence
        """
        seq_list = list(sequence[:seq_length])
        
        # Ensure the sequence is long enough.
        while len(seq_list) < seq_length:
            seq_list.append(random.choice(self.amino_acids))
        
        # Apply fixed-position constraints.
        for pos, aa in self.fixed_positions.items():
            if pos < len(seq_list):
                seq_list[pos] = aa
        
        return ''.join(seq_list)
    
    def _generate_constrained_sequence(self, seq_length: int) -> str:
        """
        Generate a random sequence that satisfies constraints.
        """
        seq_list = [random.choice(self.amino_acids) for _ in range(seq_length)]
        
        # Apply fixed positions.
        for pos, aa in self.fixed_positions.items():
            if pos < seq_length:
                seq_list[pos] = aa
        
        return ''.join(seq_list)
    
    def _get_mutable_positions(self, seq_length: int) -> List[int]:
        """
        Get mutable positions, excluding fixed positions.
        """
        return [i for i in range(seq_length) if i not in self.fixed_positions]
    
    def save_candidates(self, candidates: List[Tuple[str, float]], filepath: str):
        """Save candidate sequences to a file."""
        with open(filepath, 'w') as f:
            f.write("Rank\tSequence\tPredicted_Activity\n")
            for i, (seq, score) in enumerate(candidates, 1):
                f.write(f"{i}\t{seq}\t{score:.6f}\n")
        print(f"\nCandidate sequences saved to: {filepath}")


# ==========================================
# Main program: load model and generate sequences
# ==========================================
def main():
    parser = argparse.ArgumentParser(description='Generate high-activity peptide sequences')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to the configuration file')
    parser.add_argument('--model_path', type=str, default='checkpoints/model.pt',
                       help='Path to trained model weights')
    parser.add_argument('--n_candidates', type=int, default=10,
                       help='Number of final candidates to return')
    parser.add_argument('--seq_length', type=int, default=9,
                       help='Peptide length')
    parser.add_argument('--fixed_positions', type=str, default=None,
                       help='Fixed positions, format: "0:M,3:C,7:K" means position 0 is M, position 3 is C, and position 7 is K')
    parser.add_argument('--n_mcmc_runs', type=int, default=5,
                       help='Number of independent MCMC runs')
    parser.add_argument('--mcmc_iterations', type=int, default=200,
                       help='Number of iterations per MCMC run')
    parser.add_argument('--output', type=str, default='generated_peptides.txt',
                       help='Output file path')
    args = parser.parse_args()
    
    # Load configuration.
    config = Config(args.config)
    
    # Set up environment.
    os.environ["CUDA_VISIBLE_DEVICES"] = config.get('device.cuda_visible_devices', '0')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ==========================================
    # 1. Load tokenizers.
    # ==========================================
    print("\nLoading tokenizers...")
    mol_tokenizer = AutoTokenizer.from_pretrained(
        config.get('model.mol_model_name'), 
        trust_remote_code=True
    )
    prot_tokenizer = AutoTokenizer.from_pretrained(
        config.get('model.prot_model_name')
    )
    
    # ==========================================
    # 2. Load the trained model.
    # ==========================================
    print("\nLoading trained model...")
    model = AdamantanePeptideModel(config).to(device)
    
    # Load trained weights.
    if os.path.exists(args.model_path):
        print(f"Loading model weights from {args.model_path}...")
        checkpoint = torch.load(args.model_path, map_location=device)
        
        # Correctly extract model_state_dict.
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Compatible with checkpoints that directly save weights.
            model.load_state_dict(checkpoint)
        
        print("Model weights loaded successfully.")
    else:
        print(f"Warning: model file not found: {args.model_path}")
        print("Using an untrained model for testing only.")
    
    model.eval()
    
    fixed_positions = {}
    if args.fixed_positions:
        try:
            for item in args.fixed_positions.split(','):
                pos_str, aa = item.strip().split(':')
                pos = int(pos_str)
                aa = aa.upper().strip()
                
                # Validate position range.
                if pos < 0 or pos >= args.seq_length:
                    raise ValueError(f"Position {pos} is outside sequence length range [0, {args.seq_length})")
                
                fixed_positions[pos] = aa
            
            print("\n" + "="*80)
            print("Positional constraints:")
            print("="*80)
            for pos, aa in sorted(fixed_positions.items()):
                print(f"  Position {pos}: {aa}")
            print("="*80)
        except Exception as e:
            print(f"Error: failed to parse fixed position arguments - {e}")
            print("Example format: --fixed_positions '0:M,3:C,7:K'")
            return
    
    # ==========================================
    # 3. Initialize generator.
    # ==========================================
    print("\nInitializing sequence generator...")
    generator = PeptideGenerator(
        model=model,
        mol_tokenizer=mol_tokenizer,
        prot_tokenizer=prot_tokenizer,
        device=device,
        amino_acids='ACDEFGHIKLMNPQRSTVWY',
        fixed_positions=fixed_positions
    )
    
    # ==========================================
    # 4. Set adamantane SMILES.
    # ==========================================
    adamantane_smiles = "C1C2CC3CC1CC(C2)C3"
    
    # ==========================================
    # 5. Generate candidate sequences.
    # ==========================================
    print("\n" + "="*80)
    print("Starting high-activity PDC peptide candidate generation")
    print("="*80)
    
    candidates = generator.ensemble_generation(
        adamantane_smiles=adamantane_smiles,
        n_candidates=args.n_candidates,
        seq_length=args.seq_length,
        min_score_threshold=0.9,
        diversity_threshold=0.1,
        n_mcmc_runs=args.n_mcmc_runs * 5,
        mcmc_iterations=args.mcmc_iterations * 2,
        n_top_for_refinement=10,
        gradient_iterations=100
    )
    
    # ==========================================
    # 6. Output results.
    # ==========================================
    print("\n" + "="*80)
    print(f"Top {len(candidates)} candidate sequences:")
    print("="*80)
    print(f"{'Rank':<6}{'Sequence':<15}{'Predicted Activity':<20}")
    print("-"*80)
    
    for i, (seq, score) in enumerate(candidates, 1):
        print(f"{i:<6}{seq:<15}{score:.6f}")
    
    # ==========================================
    # 7. Save results.
    # ==========================================
    generator.save_candidates(candidates, args.output)
    
    # ==========================================
    # 8. Statistical analysis.
    # ==========================================
    print("\n" + "="*80)
    print("Statistical analysis:")
    print("="*80)
    
    scores = [score for _, score in candidates]
    print(f"Mean predicted activity: {np.mean(scores):.4f}")
    print(f"Highest predicted activity: {np.max(scores):.4f}")
    print(f"Lowest predicted activity: {np.min(scores):.4f}")
    print(f"Standard deviation: {np.std(scores):.4f}")
    
    # Amino acid frequency statistics.
    aa_counts = {aa: 0 for aa in generator.amino_acids}
    for seq, _ in candidates:
        for aa in seq:
            if aa in aa_counts:
                aa_counts[aa] += 1
    
    print("\nAmino acid frequency (Top 10):")
    sorted_aa = sorted(aa_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for aa, count in sorted_aa:
        freq = count / (len(candidates) * args.seq_length) * 100
        print(f"  {aa}: {count:3d} times ({freq:.1f}%)")
    
    print("\nGeneration complete.")


if __name__ == "__main__":
    main()
