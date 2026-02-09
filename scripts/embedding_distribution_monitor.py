"""
Embedding Distribution Monitor
Analyzes and compares embedding distributions between real and generated images.
Monitors diversity, alignment, and distribution gaps for RDM + StyleGAN2 training.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import argparse
import json
from tqdm import tqdm


class EmbeddingDistributionMonitor:
    """Monitor embedding statistics for real vs generated images."""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.stats = {
            'real': {'diversity': [], 'alignment': [], 'eigenvalues': []},
            'generated': {'diversity': [], 'alignment': [], 'eigenvalues': []},
        }
    
    def compute_diversity_metrics(self, seg_tokens, seg_pad_mask=None):
        """
        Compute diversity metrics for segment embeddings.
        
        Args:
            seg_tokens: [B, N, C] segment embeddings
            seg_pad_mask: [B, N] boolean mask (True = padded)
            
        Returns:
            dict with diversity metrics
        """
        B, N, C = seg_tokens.shape
        
        # Remove padding
        if seg_pad_mask is not None:
            mask = ~seg_pad_mask
        else:
            mask = torch.ones(B, N, dtype=torch.bool, device=seg_tokens.device)
        
        metrics = {
            'pairwise_similarity': [],
            'covariance_eigenvalues': [],
            'rank': [],
            'condition_number': [],
        }
        
        for i in range(B):
            valid_tokens = seg_tokens[i, mask[i]]  # [N_valid, C]
            if valid_tokens.shape[0] < 2:
                continue
            
            # 1. Pairwise similarity (want low for diversity)
            seg_norm = F.normalize(valid_tokens, dim=1)
            sim_matrix = torch.mm(seg_norm, seg_norm.t())  # [N_valid, N_valid]
            n_valid = valid_tokens.shape[0]
            off_diag = sim_matrix - torch.eye(n_valid, device=sim_matrix.device)
            avg_sim = off_diag.sum() / (n_valid * (n_valid - 1))
            metrics['pairwise_similarity'].append(avg_sim.item())
            
            # 2. Covariance eigenvalues (want well-distributed for diversity)
            mean = valid_tokens.mean(dim=0, keepdim=True)
            centered = valid_tokens - mean
            cov = (centered.T @ centered) / valid_tokens.shape[0]
            
            try:
                eigenvalues = torch.linalg.eigvalsh(cov).cpu().numpy()
                eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order
                
                metrics['covariance_eigenvalues'].append(eigenvalues)
                
                # Effective rank (number of significant eigenvalues)
                total = eigenvalues.sum()
                if total > 1e-8:
                    normalized = eigenvalues / total
                    effective_rank = np.exp(-np.sum(normalized * np.log(normalized + 1e-10)))
                    metrics['rank'].append(effective_rank)
                
                # Condition number (ratio of largest to smallest eigenvalue)
                if eigenvalues[-1] > 1e-8:
                    cond = eigenvalues[0] / eigenvalues[-1]
                    metrics['condition_number'].append(cond)
            except Exception as e:
                print(f"Warning: Failed to compute eigenvalues: {e}")
        
        # Aggregate
        summary = {}
        for key, values in metrics.items():
            if len(values) > 0:
                if key == 'covariance_eigenvalues':
                    # Average eigenvalue spectrum
                    summary[key] = {
                        'mean_spectrum': np.mean(values, axis=0).tolist(),
                        'std_spectrum': np.std(values, axis=0).tolist(),
                    }
                else:
                    summary[key] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                    }
        
        return summary
    
    def compute_alignment_metrics(self, global_emb, seg_tokens, seg_pad_mask=None):
        """
        Compute alignment between global and segment embeddings.
        
        Args:
            global_emb: [B, C_g] global embeddings
            seg_tokens: [B, N, C_s] segment embeddings
            seg_pad_mask: [B, N] boolean mask
            
        Returns:
            dict with alignment metrics
        """
        B, N, C_s = seg_tokens.shape
        C_g = global_emb.shape[1]
        
        # Normalize
        global_norm = F.normalize(global_emb, dim=1)
        seg_norm = F.normalize(seg_tokens, dim=2)
        
        # Remove padding
        if seg_pad_mask is not None:
            mask = ~seg_pad_mask
        else:
            mask = torch.ones(B, N, dtype=torch.bool, device=seg_tokens.device)
        
        alignments = []
        for i in range(B):
            valid_segs = seg_norm[i, mask[i]]  # [N_valid, C_s]
            if valid_segs.shape[0] == 0:
                continue
            
            # If dimensions don't match, we can't compute alignment directly
            # This would require a projection layer (like in the loss)
            if C_g == C_s:
                # Average segment embedding
                seg_mean = valid_segs.mean(dim=0)  # [C_s]
                
                # Cosine similarity
                alignment = F.cosine_similarity(
                    global_norm[i].unsqueeze(0),
                    seg_mean.unsqueeze(0),
                    dim=1
                ).item()
                
                alignments.append(alignment)
        
        if len(alignments) == 0:
            return {}
        
        return {
            'cosine_similarity': {
                'mean': np.mean(alignments),
                'std': np.std(alignments),
                'min': np.min(alignments),
                'max': np.max(alignments),
            }
        }
    
    def update_stats(self, source, global_emb=None, seg_tokens=None, seg_pad_mask=None):
        """
        Update statistics for real or generated embeddings.
        
        Args:
            source: 'real' or 'generated'
            global_emb: [B, C_g] global embeddings (optional)
            seg_tokens: [B, N, C_s] segment embeddings
            seg_pad_mask: [B, N] boolean mask
        """
        assert source in ['real', 'generated']
        
        # Diversity metrics
        if seg_tokens is not None:
            diversity = self.compute_diversity_metrics(seg_tokens, seg_pad_mask)
            self.stats[source]['diversity'].append(diversity)
        
        # Alignment metrics (only if both global and segment available)
        if global_emb is not None and seg_tokens is not None:
            alignment = self.compute_alignment_metrics(global_emb, seg_tokens, seg_pad_mask)
            self.stats[source]['alignment'].append(alignment)
    
    def compute_distribution_gap(self):
        """
        Compute gap between real and generated distributions.
        
        Returns:
            dict with gap metrics
        """
        gaps = {}
        
        # Compare diversity
        if (len(self.stats['real']['diversity']) > 0 and 
            len(self.stats['generated']['diversity']) > 0):
            
            real_div = self.stats['real']['diversity'][-1]
            gen_div = self.stats['generated']['diversity'][-1]
            
            for metric in ['pairwise_similarity', 'rank', 'condition_number']:
                if metric in real_div and metric in gen_div:
                    real_val = real_div[metric]['mean']
                    gen_val = gen_div[metric]['mean']
                    gap = abs(real_val - gen_val)
                    gaps[f'diversity_{metric}_gap'] = gap
        
        # Compare alignment
        if (len(self.stats['real']['alignment']) > 0 and 
            len(self.stats['generated']['alignment']) > 0):
            
            real_align = self.stats['real']['alignment'][-1]
            gen_align = self.stats['generated']['alignment'][-1]
            
            if 'cosine_similarity' in real_align and 'cosine_similarity' in gen_align:
                real_val = real_align['cosine_similarity']['mean']
                gen_val = gen_align['cosine_similarity']['mean']
                gap = abs(real_val - gen_val)
                gaps['alignment_gap'] = gap
        
        return gaps
    
    def print_summary(self):
        """Print summary of current statistics."""
        print("\n" + "="*60)
        print("EMBEDDING DISTRIBUTION SUMMARY")
        print("="*60)
        
        for source in ['real', 'generated']:
            print(f"\n{source.upper()} EMBEDDINGS:")
            
            if len(self.stats[source]['diversity']) > 0:
                div = self.stats[source]['diversity'][-1]
                print(f"  Diversity:")
                if 'pairwise_similarity' in div:
                    print(f"    Pairwise similarity: {div['pairwise_similarity']['mean']:.4f} ± {div['pairwise_similarity']['std']:.4f}")
                if 'rank' in div:
                    print(f"    Effective rank: {div['rank']['mean']:.2f} ± {div['rank']['std']:.2f}")
                if 'condition_number' in div:
                    print(f"    Condition number: {div['condition_number']['mean']:.2e}")
            
            if len(self.stats[source]['alignment']) > 0:
                align = self.stats[source]['alignment'][-1]
                print(f"  Alignment:")
                if 'cosine_similarity' in align:
                    print(f"    Cosine similarity: {align['cosine_similarity']['mean']:.4f} ± {align['cosine_similarity']['std']:.4f}")
        
        # Distribution gap
        gaps = self.compute_distribution_gap()
        if gaps:
            print(f"\nDISTRIBUTION GAP:")
            for key, value in gaps.items():
                print(f"  {key}: {value:.4f}")
        
        print("="*60 + "\n")
    
    def save_stats(self, output_path):
        """Save statistics to JSON file."""
        # Convert to serializable format
        serializable_stats = {}
        for source in ['real', 'generated']:
            serializable_stats[source] = {
                'diversity': self.stats[source]['diversity'],
                'alignment': self.stats[source]['alignment'],
            }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_stats, f, indent=2)
        
        print(f"Statistics saved to: {output_path}")


def main():
    """Example usage of the monitor."""
    parser = argparse.ArgumentParser(description='Monitor embedding distributions')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--output', type=str, default='embedding_stats.json', help='Output file')
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = EmbeddingDistributionMonitor(device=args.device)
    
    # Example: Simulate some embeddings
    print("Simulating embeddings for demonstration...")
    
    # Real embeddings (more diverse)
    real_seg = torch.randn(16, 180, 256, device=args.device)
    real_global = torch.randn(16, 256, device=args.device)
    real_mask = torch.zeros(16, 180, dtype=torch.bool, device=args.device)
    real_mask[:, 150:] = True  # Pad last 30
    
    monitor.update_stats('real', real_global, real_seg, real_mask)
    
    # Generated embeddings (potentially less diverse)
    gen_seg = torch.randn(16, 180, 256, device=args.device) * 0.8  # Slightly less variance
    gen_global = torch.randn(16, 256, device=args.device)
    gen_mask = torch.zeros(16, 180, dtype=torch.bool, device=args.device)
    gen_mask[:, 150:] = True
    
    monitor.update_stats('generated', gen_global, gen_seg, gen_mask)
    
    # Print and save
    monitor.print_summary()
    monitor.save_stats(args.output)


if __name__ == '__main__':
    main()
