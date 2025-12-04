#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

def plot_junction_analysis(csv_file, output_prefix='junction_analysis'):
    """Create comprehensive visualizations of junction analysis"""
    
    print(f"Loading data from: {csv_file}")
    df = pd.read_csv(csv_file)
    
    print(f"Loaded {len(df):,} junctions")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Read count distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(np.log10(df['read_count'] + 1), bins=50, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Log10(Read Count + 1)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Junction Read Counts')
    
    # Plot 2: Intron length distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(np.log10(df['intron_length'] + 1), bins=50, edgecolor='black', alpha=0.7, color='green')
    ax2.set_xlabel('Log10(Intron Length + 1)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Intron Lengths')
    
    # Plot 3: Chromosome distribution
    ax3 = fig.add_subplot(gs[0, 2])
    chrom_counts = df['chrom'].value_counts().head(20)
    chrom_counts.plot(kind='bar', ax=ax3, color='coral')
    ax3.set_xlabel('Chromosome')
    ax3.set_ylabel('Number of Junctions')
    ax3.set_title('Top 20 Chromosomes by Junction Count')
    ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Donor score distribution
    if 'donor_score' in df.columns:
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.hist(df['donor_score'], bins=50, edgecolor='black', alpha=0.7, color='purple')
        ax4.axvline(0.5, color='red', linestyle='--', label='High confidence threshold')
        ax4.set_xlabel('SpliceAI Donor Score')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Donor Scores')
        ax4.legend()
    
    # Plot 5: Acceptor score distribution
    if 'acceptor_score' in df.columns:
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.hist(df['acceptor_score'], bins=50, edgecolor='black', alpha=0.7, color='orange')
        ax5.axvline(0.5, color='red', linestyle='--', label='High confidence threshold')
        ax5.set_xlabel('SpliceAI Acceptor Score')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Distribution of Acceptor Scores')
        ax5.legend()
    
    # Plot 6: Donor vs Acceptor scores
    if 'donor_score' in df.columns and 'acceptor_score' in df.columns:
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.scatter(df['donor_score'], df['acceptor_score'], alpha=0.3, s=10)
        ax6.plot([0, 1], [0, 1], 'r--', alpha=0.5)
        ax6.set_xlabel('Donor Score')
        ax6.set_ylabel('Acceptor Score')
        ax6.set_title('Donor vs Acceptor Scores')
        ax6.grid(True, alpha=0.3)
    
    # Plot 7: Read count vs Donor score
    if 'donor_score' in df.columns:
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.scatter(np.log10(df['read_count'] + 1), df['donor_score'], alpha=0.3, s=10, c=df['donor_score'], cmap='viridis')
        ax7.set_xlabel('Log10(Read Count + 1)')
        ax7.set_ylabel('Donor Score')
        ax7.set_title('Read Count vs Donor Score')
        ax7.grid(True, alpha=0.3)
    
    # Plot 8: Read count vs Acceptor score
    if 'acceptor_score' in df.columns:
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.scatter(np.log10(df['read_count'] + 1), df['acceptor_score'], alpha=0.3, s=10, c=df['acceptor_score'], cmap='plasma')
        ax8.set_xlabel('Log10(Read Count + 1)')
        ax8.set_ylabel('Acceptor Score')
        ax8.set_title('Read Count vs Acceptor Score')
        ax8.grid(True, alpha=0.3)
    
    # Plot 9: Intron length vs splice scores
    if 'donor_score' in df.columns and 'acceptor_score' in df.columns:
        ax9 = fig.add_subplot(gs[2, 2])
        avg_score = (df['donor_score'] + df['acceptor_score']) / 2
        ax9.scatter(np.log10(df['intron_length'] + 1), avg_score, alpha=0.3, s=10, c=avg_score, cmap='coolwarm')
        ax9.set_xlabel('Log10(Intron Length + 1)')
        ax9.set_ylabel('Average Splice Score')
        ax9.set_title('Intron Length vs Avg Splice Score')
        ax9.grid(True, alpha=0.3)
    
    plt.savefig(f'{output_prefix}_comprehensive.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_prefix}_comprehensive.png")
    plt.close()
    
    # Create summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"\nTotal junctions: {len(df):,}")
    print(f"Unique chromosomes: {df['chrom'].nunique()}")
    print(f"\nRead counts:")
    print(f"  Mean: {df['read_count'].mean():.1f}")
    print(f"  Median: {df['read_count'].median():.1f}")
    print(f"  Max: {df['read_count'].max():,}")
    print(f"\nIntron lengths:")
    print(f"  Mean: {df['intron_length'].mean():.1f} bp")
    print(f"  Median: {df['intron_length'].median():.1f} bp")
    print(f"  Max: {df['intron_length'].max():,} bp")
    
    if 'donor_score' in df.columns:
        print(f"\nDonor scores:")
        print(f"  Mean: {df['donor_score'].mean():.4f}")
        print(f"  Median: {df['donor_score'].median():.4f}")
        print(f"  High confidence (>0.5): {(df['donor_score'] > 0.5).sum():,} ({100*(df['donor_score'] > 0.5).sum()/len(df):.1f}%)")
        print(f"  Very high (>0.8): {(df['donor_score'] > 0.8).sum():,} ({100*(df['donor_score'] > 0.8).sum()/len(df):.1f}%)")
    
    if 'acceptor_score' in df.columns:
        print(f"\nAcceptor scores:")
        print(f"  Mean: {df['acceptor_score'].mean():.4f}")
        print(f"  Median: {df['acceptor_score'].median():.4f}")
        print(f"  High confidence (>0.5): {(df['acceptor_score'] > 0.5).sum():,} ({100*(df['acceptor_score'] > 0.5).sum()/len(df):.1f}%)")
        print(f"  Very high (>0.8): {(df['acceptor_score'] > 0.8).sum():,} ({100*(df['acceptor_score'] > 0.8).sum()/len(df):.1f}%)")
    
    # Top junctions by read count
    print("\n" + "="*70)
    print("TOP 10 JUNCTIONS BY READ COUNT")
    print("="*70)
    top_junctions = df.nlargest(10, 'read_count')
    print(top_junctions[['chrom', 'donor', 'acceptor', 'intron_length', 'read_count']].to_string())
    
    if 'donor_score' in df.columns and 'acceptor_score' in df.columns:
        # Top junctions by splice score
        print("\n" + "="*70)
        print("TOP 10 JUNCTIONS BY AVERAGE SPLICE SCORE")
        print("="*70)
        df_copy = df.copy()
        df_copy['avg_score'] = (df_copy['donor_score'] + df_copy['acceptor_score']) / 2
        top_scored = df_copy.nlargest(10, 'avg_score')
        print(top_scored[['chrom', 'donor', 'acceptor', 'read_count', 'donor_score', 'acceptor_score', 'avg_score']].to_string())


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python visualize_results.py <results.csv> [output_prefix]")
        print("\nExample:")
        print("  python visualize_results.py results/spliceai_junction_analysis.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    output_prefix = sys.argv[2] if len(sys.argv) > 2 else 'junction_analysis'
    
    plot_junction_analysis(csv_file, output_prefix)
