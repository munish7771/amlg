import os
import sys
import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd

# Append the root of the Git repository to the path.
git_root = os.popen(cmd="git rev-parse --show-toplevel").read().strip()
sys.path.append(git_root)

from src.exercises.perturbation_data_analysis import pertdata as pt

def generate_plots():
    print("Loading data...")
    try:
        norman = pt.PertData.from_repo(name="norman", save_dir="data")
    except Exception as e:
        # Fallback if running from script location
        os.makedirs("data", exist_ok=True)
        norman = pt.PertData.from_repo(name="norman", save_dir="data")
    
    print("Doing plot settings...")
    sc.set_figure_params(dpi=300, facecolor='white')
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 20
    })
    
    adata = norman.adata

    
    print("Preprocessing and Analyzing...")
    # QC Metrics
    print("Calculating QC metrics...")
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    
    # Define report_dir
    report_dir = os.path.join(git_root, "src/projects/project_report")
    os.makedirs(report_dir, exist_ok=True)

    # QC Plot
    print("Generating QC Plot...")
    plt.clf()
    sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], 
                 jitter=0.4, multi_panel=True, show=False)
    save_path_qc = os.path.join(report_dir, "figure_qc.png")
    plt.savefig(save_path_qc, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved QC plot to {save_path_qc}")

    # HVGs
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    
    # Custom HVG Plot (Normalized only)
    print("Generating custom HVG Plot...")
    plt.clf()
    # Plot all genes first
    plt.scatter(adata.var['means'], adata.var['dispersions_norm'], s=3, c='grey', label='Non-variable', alpha=0.5)
    # Overlay highly variable
    hvg = adata.var[adata.var['highly_variable']]
    plt.scatter(hvg['means'], hvg['dispersions_norm'], s=3, c='red', label='Highly variable')
    
    plt.xlabel('Mean expression')
    plt.ylabel('Normalized dispersion')
    plt.title('Highly Variable Genes (Normalized Dispersion)')
    plt.legend(frameon=False)
    plt.tight_layout()
    
    # sc.pl.highly_variable_genes(adata, show=False) # Replaced with custom plot
    
    save_path_hvg = os.path.join(report_dir, "figure0_hvg.png")
    plt.savefig(save_path_hvg, dpi=300, bbox_inches='tight')
    plt.close() # Close the figure to free memory
    print(f"Saved HVG plot to {save_path_hvg}")

    adata = adata[:, adata.var['highly_variable']]
    
    # Dim Red & Clustering
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.leiden(adata)
    sc.tl.umap(adata)
    
    # Ensure report directory exists
    # report_dir = os.path.join(git_root, "src/projects/project_report") # Already defined above
    # os.makedirs(report_dir, exist_ok=True) # Already created above
    
    print("Generating Figure 1: UMAP...")
    # UMAP Plot
    # When plotting multiple colors, scanpy creates subplots. We should not pass a single ax.
    # We clear any existing plots first
    plt.clf()
    sc.pl.umap(adata, color='leiden', legend_loc='on data', show=False)
    plt.title("UMAP: Leiden Clusters")
    save_path_umap = os.path.join(report_dir, "figure1_umap.png")
    plt.savefig(save_path_umap, dpi=300, bbox_inches='tight')
    plt.close() # Close the figure to free memory
    print(f"Saved UMAP to {save_path_umap}")

    # Ctrl vs Rest UMAP
    print("Generating Figure: UMAP Ctrl vs Rest...")
    adata.obs['ctrl_vs_rest'] = adata.obs['condition'].apply(lambda x: 'ctrl' if x == 'ctrl' else 'non-ctrl')
    
    # Custom Palette
    palette = {'ctrl': '#d62728', 'non-ctrl': '#d3d3d3'} # Red vs LightGrey
    
    plt.clf()
    # Plot non-ctrl first (background) - Light, small, transparent
    # We grab the axis from the first plot to overlay the second
    ax = sc.pl.umap(adata[adata.obs['ctrl_vs_rest'] == 'non-ctrl'], 
                    color='ctrl_vs_rest', palette=palette,
                    show=False, size=20, alpha=0.1) # increased size slightly but very low alpha for 'fog' effect
                    
    # Overlay ctrl - Bold, large, opaque
    sc.pl.umap(adata[adata.obs['ctrl_vs_rest'] == 'ctrl'], ax=ax, 
               color='ctrl_vs_rest', palette=palette, 
               size=50, alpha=1.0, show=False)
               
    plt.title("UMAP: Ctrl vs Rest")

    # The clobbering call has been removed.

    
    save_path_umap_ctrl = os.path.join(report_dir, "figure_umap_ctrl.png")
    plt.savefig(save_path_umap_ctrl, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Ctrl UMAP to {save_path_umap_ctrl}")

    # Composition Plot
    print("Generating Figure: Perturbation Composition...")
    # Create crosstab
    composition = pd.crosstab(adata.obs['leiden'], adata.obs['condition'], normalize='index')
    
    plt.clf()
    
    # Selection strategy for plot:
    # 1. Always include 'ctrl'
    # 2. Include specific key perturbations mentioned in text
    # 3. Include the top condition for each cluster (driver)
    # 4. Include top N globally frequent to fill up if needed
    
    selected_conditions = set(['ctrl'])
    
    # explicit inclusions
    all_conditions = adata.obs['condition'].unique()
    for cond in all_conditions:
        if 'TGFBR2' in cond or 'TP53' in cond:
            selected_conditions.add(cond)
            
    # cluster drivers (max in each row)
    for cluster in composition.index:
        # get condition with max proportion in this cluster
        top_cond = composition.loc[cluster].idxmax()
        selected_conditions.add(top_cond)
        
    # top global frequency (to ensure we cover the major ones)
    top_global = adata.obs['condition'].value_counts().head(10).index.tolist()
    selected_conditions.update(top_global)
    
    selected_conditions = list(selected_conditions)
    
    # Filter composition dataframe
    composition_plot = composition.copy()
    other_cols = [c for c in composition_plot.columns if c not in selected_conditions]
    if other_cols:
        composition_plot['Other'] = composition_plot[other_cols].sum(axis=1)
        composition_plot = composition_plot.drop(columns=other_cols)
    
    # Sort columns for consistent coloring (Other last)
    cols = sorted([c for c in composition_plot.columns if c != 'Other'])
    if 'Other' in composition_plot.columns:
        cols.append('Other')
    composition_plot = composition_plot[cols]
    
    # Print dominant perturbations for report text
    with open("cluster_info.txt", "w") as f:
        f.write("--- Dominant Perturbations per Cluster ---\n")
        print("--- Dominant Perturbations per Cluster ---")
        for cluster in composition.index:
            # Get the original composition row (all conditions)
            row = composition.loc[cluster]
            top_cond = row.idxmax()
            top_prop = row.max()
            msg = f"Cluster {cluster}: {top_cond} ({top_prop:.2f})"
            print(msg)
            f.write(msg + "\n")
        f.write("------------------------------------------\n")
        print("------------------------------------------")

    # Plot
    ax = composition_plot.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='tab20')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Perturbation', fontsize='small')
    plt.title("Cluster Composition by Perturbation")
    plt.xlabel("Leiden Cluster")
    plt.ylabel("Proportion")
    plt.tight_layout()
    
    save_path_composition = os.path.join(report_dir, "figure_composition.png")
    plt.savefig(save_path_composition, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Composition Plot to {save_path_composition}")

    

    print("Generating Figure 2: Heatmap...")
    # Markers
    # sc.tl.rank_genes_groups(adata, 'leiden', method='t-test') # Original
    sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon') # Updated to Wilcoxon
    
    # Heatmap
    plt.clf()
    sc.pl.rank_genes_groups_heatmap(adata, n_genes=5, groupby='leiden', show=False)
    save_path_heatmap = os.path.join(report_dir, "figure2_heatmap.png")
    plt.savefig(save_path_heatmap, bbox_inches='tight', dpi=300)
    plt.close() # Close the figure to free memory
    print(f"Saved Heatmap to {save_path_heatmap}")

    # Specific Comparison: 9 vs 12
    print("Generating 9 vs 12 Comparison...")
    try:
        # Check if clusters 9 and 12 exist
        if '9' in adata.obs['leiden'].cat.categories and '12' in adata.obs['leiden'].cat.categories:
            # Ensure the previous rank_genes_groups result is cleared or overwritten
            sc.tl.rank_genes_groups(adata, 'leiden', groups=['9'], reference='12', method='wilcoxon', key_added='rank_genes_9vs12')
            
            plt.clf()
            # Restore to rank genes groups plot as per request ("use the previous")
            sc.pl.rank_genes_groups(adata, n_genes=20, sharey=False, show=False, key='rank_genes_9vs12')
                          
            save_path_9vs12 = os.path.join(report_dir, "figure_comparison_9vs12.png")
            plt.savefig(save_path_9vs12, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Saved 9vs12 Ranking Plot to {save_path_9vs12}")
            
            # Export top genes to text for report
            result = adata.uns['rank_genes_9vs12']
            groups = result['names'].dtype.names
            top_genes_df = pd.DataFrame(
                {group + '_' + key[:1]: result[key][group]
                for group in groups for key in ['names', 'pvals_adj', 'logfoldchanges']}).head(5)
            
            with open(os.path.join(report_dir, "comparison_9vs12_genes.txt"), "w") as f:
                f.write("--- Top Genes for Cluster 9 vs Cluster 12 ---\n")
                f.write(top_genes_df.to_string())
                f.write("\n\nFull list of top 5 genes for Cluster 9 (vs 12):\n")
                
                genes_9 = result['names']['9'][:5]
                f.write(str(genes_9))
                f.write("\n\n--- Gene Symbols ---\n")
                
                # Check for symbol column
                symbol_col = None
                for col in ['gene_name', 'gene_symbols', 'feature_name']:
                    if col in adata.var.columns:
                        symbol_col = col
                        break
                
                if symbol_col:
                    f.write(f"Using column '{symbol_col}' for symbols.\n")
                    for gid in genes_9:
                        if gid in adata.var_names:
                            sym = adata.var.loc[gid, symbol_col]
                            f.write(f"{gid}: {sym}\n")
                        else:
                            f.write(f"{gid}: Not found in var\n")
                else:
                    f.write("No symbol column found in adata.var.\n")
                    f.write(f"Columns: {list(adata.var.columns)}\n")

                f.write("\n------------------------------------------\n")

        else:
            print("Clusters '9' or '12' not found in adata.obs['leiden']. Skipping 9 vs 12 comparison.")
    except Exception as e:
        print(f"Could not run 9vs12 comparison: {e}")


if __name__ == "__main__":
    try:
        generate_plots()
    except Exception:
        import traceback
        with open("error_log.txt", "w") as f:
            f.write(traceback.format_exc())
        traceback.print_exc()
