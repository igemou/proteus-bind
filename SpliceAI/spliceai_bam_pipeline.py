#!/usr/bin/env python3

import argparse
import pysam
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
import sys

# For SpliceAI
from keras.models import load_model
from pkg_resources import resource_filename

def one_hot_encode(seq):
    """One-hot encode a DNA sequence"""
    map = np.asarray([[0, 0, 0, 0],
                      [1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
    seq = seq.upper().replace('A', '\x01').replace('C', '\x02')
    seq = seq.replace('G', '\x03').replace('T', '\x04').replace('N', '\x00')
    return map[np.frombuffer(seq.encode('latin-1'), np.int8) % 5]


class SpliceJunctionExtractor:
    """Extract splice junctions from BAM file"""
    
    def __init__(self, bam_file):
        self.bam_file = bam_file
        self.junctions = defaultdict(int)
    
    def extract_junctions(self, min_reads=2):

        print(f"Reading BAM file: {self.bam_file}")
        bamfile = pysam.AlignmentFile(self.bam_file, "rb")
        
        read_count = 0
        for read in bamfile.fetch():
            read_count += 1
            if read_count % 100000 == 0:
                print(f"  Processed {read_count:,} reads...")
            
            # Skip unmapped reads
            if read.is_unmapped:
                continue
            
            # Get reference name (chromosome)
            chrom = bamfile.get_reference_name(read.reference_id)
            
            # Parse CIGAR string to find splice junctions (N operations)
            cigar = read.cigartuples
            if cigar is None:
                continue
            
            # Track position in reference
            ref_pos = read.reference_start
            
            for operation, length in cigar:
                # N operation indicates skipped region (intron)
                if operation == 3:  # 3 = CIGAR 'N' (skipped region from reference)
                    # Junction coordinates
                    donor = ref_pos  # Last base of upstream exon
                    acceptor = ref_pos + length  # First base of downstream exon
                    
                    # Store junction
                    junction_key = (chrom, donor, acceptor, length)
                    self.junctions[junction_key] += 1
                
                # Move reference position
                if operation in [0, 2, 3, 7, 8]:  # Operations that consume reference
                    ref_pos += length
        
        bamfile.close()
        print(f"✓ Processed {read_count:,} total reads")
        
        # Convert to DataFrame
        junction_data = []
        for (chrom, donor, acceptor, intron_len), count in self.junctions.items():
            if count >= min_reads:
                junction_data.append({
                    'chrom': chrom,
                    'donor': donor,
                    'acceptor': acceptor,
                    'intron_length': intron_len,
                    'read_count': count
                })
        
        df = pd.DataFrame(junction_data)
        df = df.sort_values(['chrom', 'donor'])
        
        print(f"✓ Found {len(df):,} junctions with ≥{min_reads} supporting reads")
        
        return df


class SpliceAIAnalyzer:
    
    def __init__(self):
        print("Loading SpliceAI models...")
        paths = [resource_filename('spliceai', f'models/spliceai{x}.h5') for x in range(1, 6)]
        self.models = [load_model(path, compile=False) for path in paths]
        print("✓ Models loaded")
    
    def predict_splice_sites(self, sequence, context=10000):
        """Predict splice sites in a sequence"""
        seq_len = len(sequence)
        padded_seq = 'N' * (context // 2) + sequence.upper() + 'N' * (context // 2)
        x = one_hot_encode(padded_seq)[None, :]
        
        predictions = [model.predict(x, verbose=0) for model in self.models]
        avg_pred = np.mean(predictions, axis=0)
        
        return {
            'acceptor_scores': avg_pred[0, :, 1],
            'donor_scores': avg_pred[0, :, 2]
        }


def extract_sequences_around_junctions(junctions_df, reference_fasta, context=200):

    from pyfaidx import Fasta
    
    print(f"Loading reference genome: {reference_fasta}")
    genome = Fasta(reference_fasta)
    
    sequences = []
    
    for idx, row in junctions_df.iterrows():
        if idx % 100 == 0:
            print(f"  Extracting sequences: {idx}/{len(junctions_df)}")
        
        chrom = row['chrom']
        donor = row['donor']
        acceptor = row['acceptor']
        
        # Extract donor sequence (around donor site)
        try:
            donor_seq = genome[chrom][donor-context:donor+context].seq
        except:
            donor_seq = 'N' * (2 * context)
        
        # Extract acceptor sequence (around acceptor site)
        try:
            acceptor_seq = genome[chrom][acceptor-context:acceptor+context].seq
        except:
            acceptor_seq = 'N' * (2 * context)
        
        sequences.append({
            'donor_seq': donor_seq,
            'acceptor_seq': acceptor_seq
        })
    
    seq_df = pd.DataFrame(sequences)
    result_df = pd.concat([junctions_df.reset_index(drop=True), seq_df], axis=1)
    
    print("✓ Sequences extracted")
    return result_df


def analyze_junctions_with_spliceai(junctions_df, spliceai_analyzer):

    print("Running SpliceAI predictions...")
    
    results = []
    
    for idx, row in junctions_df.iterrows():
        if idx % 10 == 0:
            print(f"  Analyzing junction {idx}/{len(junctions_df)}")
        
        # Predict on donor sequence
        donor_pred = spliceai_analyzer.predict_splice_sites(row['donor_seq'])
        
        # Predict on acceptor sequence
        acceptor_pred = spliceai_analyzer.predict_splice_sites(row['acceptor_seq'])
        
        # Get scores at the junction position (middle of sequence)
        mid_pos = len(row['donor_seq']) // 2
        
        results.append({
            'donor_score': float(donor_pred['donor_scores'][mid_pos]),
            'acceptor_score': float(acceptor_pred['acceptor_scores'][mid_pos]),
            'donor_max_score': float(np.max(donor_pred['donor_scores'])),
            'acceptor_max_score': float(np.max(acceptor_pred['acceptor_scores']))
        })
    
    results_df = pd.DataFrame(results)
    output_df = pd.concat([junctions_df.reset_index(drop=True), results_df], axis=1)
    
    print("✓ SpliceAI analysis complete")
    return output_df


def main():
    parser = argparse.ArgumentParser(description='Extract splice junctions from BAM and analyze with SpliceAI')
    parser.add_argument('--bam', required=True, help='Input BAM file')
    parser.add_argument('--ref', required=False, help='Reference genome FASTA (optional, for sequence extraction)')
    parser.add_argument('--output', default='./spliceai_results/', help='Output directory')
    parser.add_argument('--min-reads', type=int, default=2, help='Minimum reads supporting junction')
    parser.add_argument('--context', type=int, default=200, help='Context length around junctions (bp)')
    parser.add_argument('--no-spliceai', action='store_true', help='Skip SpliceAI analysis')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Splice Junction Extraction and SpliceAI Analysis Pipeline")
    print("="*70)
    
    # Step 1: Extract junctions from BAM
    print("\n[Step 1] Extracting splice junctions from BAM...")
    extractor = SpliceJunctionExtractor(args.bam)
    junctions_df = extractor.extract_junctions(min_reads=args.min_reads)
    
    # Save junctions
    junctions_file = output_dir / 'splice_junctions.csv'
    junctions_df.to_csv(junctions_file, index=False)
    print(f"✓ Junctions saved to: {junctions_file}")
    
    # Step 2: Extract sequences (if reference provided)
    if args.ref:
        print("\n[Step 2] Extracting sequences around junctions...")
        junctions_df = extract_sequences_around_junctions(
            junctions_df, 
            args.ref, 
            context=args.context
        )
        
        # Save with sequences
        seq_file = output_dir / 'junctions_with_sequences.csv'
        junctions_df.to_csv(seq_file, index=False)
        print(f"✓ Sequences saved to: {seq_file}")
        
        # Step 3: Run SpliceAI (if not disabled)
        if not args.no_spliceai:
            print("\n[Step 3] Running SpliceAI analysis...")
            analyzer = SpliceAIAnalyzer()
            results_df = analyze_junctions_with_spliceai(junctions_df, analyzer)
            
            # Save results
            results_file = output_dir / 'spliceai_junction_analysis.csv'
            results_df.to_csv(results_file, index=False)
            print(f"✓ Results saved to: {results_file}")
            
            # Print summary
            print("\n" + "="*70)
            print("SUMMARY")
            print("="*70)
            print(f"Total junctions analyzed: {len(results_df):,}")
            print(f"High-confidence donors (score > 0.5): {(results_df['donor_score'] > 0.5).sum()}")
            print(f"High-confidence acceptors (score > 0.5): {(results_df['acceptor_score'] > 0.5).sum()}")
            print(f"\nMean donor score: {results_df['donor_score'].mean():.4f}")
            print(f"Mean acceptor score: {results_df['acceptor_score'].mean():.4f}")
    else:
        print("\nSkipping sequence extraction (no reference genome provided)")
        print("To extract sequences and run SpliceAI, provide --ref hg19.fa")
    
    print("\n" + "="*70)
    print("Pipeline complete!")
    print("="*70)


if __name__ == '__main__':
    main()
