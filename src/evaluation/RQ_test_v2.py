import pandas as pd
import numpy as np
from pathlib import Path
import sys
from scipy.stats import wilcoxon, spearmanr, norm
from numpy import log, median

def rank_biserial(diff):
    """Calculate rank-biserial for paired samples."""
    pos = np.sum(diff > 0)
    neg = np.sum(diff < 0)
    return (pos - neg) / len(diff)

def fisher_z(r):
    """Fisher's r-to-z transformation."""
    return 0.5 * np.log((1 + r) / (1 - r))

def format_p_value(p_value):
    """Format p-values for LaTeX tables."""
    if p_value < 0.001:
        return "p$<$0.001"
    else:
        return f"p={p_value:.3f}"

def format_with_bold(value, p_value, is_effect_size=False):
    """Format values with bold if p-value is significant."""
    if p_value < 0.05:
        if is_effect_size:
            return f"\\textbf{{{value:.3f}}}"
        else:
            return f"\\textbf{{{value:.1f}}}"
    else:
        if is_effect_size:
            return f"{value:.3f}"
        else:
            return f"{value:.1f}"

def analyze_data(df_path):
    """Analyze data and generate statistics for all research questions."""
    # Load data
    df = pd.read_pickle(df_path)
    
    # Define column mapping for Chamfer Distance only
    cd_cols = {
        "input":     "input_chamfer_distance",
        "baseline":  "baseline_chamfer_distance",
        "naip":      "naip_chamfer_distance",
        "uavsar":    "uavsar_chamfer_distance",
        "fused":     "combined_chamfer_distance",
    }
    
    # Initialize result storage
    results = {}
    
    # Descriptive Statistics for all models
    descriptive_stats = {}
    for model_name, col_name in cd_cols.items():
        # Remove any infinite or NaN values for clean statistics
        clean_data = df[col_name].replace([np.inf, -np.inf], np.nan).dropna()
        
        descriptive_stats[model_name] = {
            "mean": clean_data.mean(),
            "median": clean_data.median(),
            "std": clean_data.std(),
            "q25": clean_data.quantile(0.25),
            "q75": clean_data.quantile(0.75),
            "iqr": clean_data.quantile(0.75) - clean_data.quantile(0.25),
            "n": len(clean_data)
        }
    
    results["descriptive_stats"] = descriptive_stats
    
    # RQ1: NAIP vs baseline
    d_naip = df[cd_cols["baseline"]] - df[cd_cols["naip"]]  # Positive if NAIP is better
    w_naip, p_naip = wilcoxon(df[cd_cols["naip"]], df[cd_cols["baseline"]])
    median_naip = median(d_naip)  # Median improvement of NAIP over baseline
    pct_naip = 100 * median_naip / median(df[cd_cols['baseline']])
    rb_naip = rank_biserial(d_naip)  # Positive if NAIP tends to be better
    
    results["naip_vs_baseline"] = {
        "median_change": median_naip,
        "percent_change": pct_naip,
        "effect_size": rb_naip,
        "p_value": p_naip
    }
    
    # RQ1: UAVSAR vs baseline
    d_uav = df[cd_cols["baseline"]] - df[cd_cols["uavsar"]]  # Positive if UAVSAR is better
    w_uav, p_uav = wilcoxon(df[cd_cols["uavsar"]], df[cd_cols["baseline"]])
    median_uav = median(d_uav)  # Median improvement of UAVSAR over baseline
    pct_uav = 100 * median_uav / median(df[cd_cols['baseline']])
    rb_uav = rank_biserial(d_uav)  # Positive if UAVSAR tends to be better
    
    results["uavsar_vs_baseline"] = {
        "median_change": median_uav,
        "percent_change": pct_uav,
        "effect_size": rb_uav,
        "p_value": p_uav
    }
    
    # RQ2: Fused vs NAIP (no Holm's correction needed)
    d_fused = df[cd_cols["naip"]] - df[cd_cols["fused"]]  # Positive if fused is better
    w_fus_naip, p_fus_naip = wilcoxon(df[cd_cols["fused"]], df[cd_cols["naip"]])
    
    median_fused = median(d_fused)  # Median improvement of fused over NAIP
    pct_fused = 100 * median_fused / median(df[cd_cols['naip']])
    rb_fused = rank_biserial(d_fused)  # Positive if fused tends to be better
    
    results["fused_vs_naip"] = {
        "median_change": median_fused,
        "percent_change": pct_fused,
        "effect_size": rb_fused,
        "p_value": p_fus_naip
    }
    
    # RQ3: Correlation with |ΔH|
    numeric_cols = [
        "net_canopy_height_change",
        cd_cols["baseline"],
        cd_cols["naip"],
        cd_cols["uavsar"],
        cd_cols["fused"],
    ]
    
    clean = (
        df[numeric_cols]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    N_clean = len(clean)
    
    abs_dh = clean["net_canopy_height_change"].abs().values
    
    def rho(colname):
        return spearmanr(abs_dh, clean[colname].values)
    
    rho_base, p_base = rho(cd_cols["baseline"])
    rho_naip, p_naip_rho = rho(cd_cols["naip"])
    rho_uav, p_uav_rho = rho(cd_cols["uavsar"])
    rho_fused, p_fused_rho = rho(cd_cols["fused"])
    
    # Fisher r-to-z comparison baseline vs fused
    z = (fisher_z(rho_base) - fisher_z(rho_fused)) / np.sqrt(2 / (N_clean - 3))
    p_z = 2 * norm.sf(abs(z))
    
    # Fisher r-to-z comparison baseline vs naip
    z_naip = (fisher_z(rho_base) - fisher_z(rho_naip)) / np.sqrt(2 / (N_clean - 3))
    p_z_naip = 2 * norm.sf(abs(z_naip))
    
    # Fisher r-to-z comparison baseline vs uavsar
    z_uav = (fisher_z(rho_base) - fisher_z(rho_uav)) / np.sqrt(2 / (N_clean - 3))
    p_z_uav = 2 * norm.sf(abs(z_uav))
    
    results["correlation"] = {
        "rho_baseline": rho_base,
        "rho_naip": rho_naip,
        "rho_uavsar": rho_uav,
        "rho_fused": rho_fused,
        "p_baseline": p_base,
        "p_naip": p_naip_rho,
        "p_uavsar": p_uav_rho,
        "p_fused": p_fused_rho,
        "z_value": z,
        "p_z": p_z,
        "z_naip": z_naip,
        "p_z_naip": p_z_naip,
        "z_uav": z_uav,
        "p_z_uav": p_z_uav
    }
    
    # RQ3 Extended: Separate analysis for canopy gains and losses
    # Create masks for gains and losses
    gain_mask = clean["net_canopy_height_change"] >= 0
    loss_mask = ~gain_mask
    
    # Function to calculate correlations for a specific mask
    def calculate_correlations(mask, label):
        abs_dh_subset = clean.loc[mask, "net_canopy_height_change"].abs().values
        n_subset = mask.sum()
        
        # Calculate correlations for each model
        correlations = {}
        for name, col in cd_cols.items():
            if name == 'input':
                continue  # Skip input as it's not a model
            rho_val, p_val = spearmanr(abs_dh_subset, clean.loc[mask, col])
            correlations[f"rho_{name}"] = rho_val
            correlations[f"p_{name}"] = p_val
        
        # Calculate Fisher z comparisons for baseline vs. each model
        rho_base_subset = correlations["rho_baseline"]
        for name in ["naip", "uavsar", "fused"]:
            rho_model = correlations[f"rho_{name}"]
            z_val = (fisher_z(rho_base_subset) - fisher_z(rho_model)) / np.sqrt(2/(n_subset-3))
            p_z_val = 2 * norm.sf(abs(z_val))
            correlations[f"z_{name}"] = z_val
            correlations[f"p_z_{name}"] = p_z_val
        
        correlations["n"] = n_subset
        return correlations
    
    # Calculate correlations for gains and losses
    results["correlation_gains"] = calculate_correlations(gain_mask, "gains")
    results["correlation_losses"] = calculate_correlations(loss_mask, "losses")
    
    return results

def generate_latex_tables(results):
    """Generate LaTeX tables for all research questions."""
    tables = {}
    
    # Descriptive Statistics Table
    desc_table = """\\begin{table}[htbp]
\\centering
\\caption{Descriptive Statistics for Chamfer Distance Across All Models}
\\begin{tabular}{lcccc}
\\toprule
\\textbf{Model} & \\textbf{Mean CD (m)} & \\textbf{Median CD (m)} & \\textbf{Std Dev (m)} & \\textbf{IQR (m)} \\\\
\\midrule
"""
    
    # Model order for display
    model_display_order = [
        ("input", "Input"),
        ("baseline", "Baseline"), 
        ("naip", "NAIP"),
        ("uavsar", "UAVSAR"),
        ("fused", "Fused")
    ]
    
    desc_stats = results["descriptive_stats"]
    for model_key, model_label in model_display_order:
        stats = desc_stats[model_key]
        desc_table += f"{model_label} & {stats['mean']:.3f} & {stats['median']:.3f} & {stats['std']:.3f} & {stats['iqr']:.3f} \\\\\n"
    
    desc_table += """\\bottomrule
\\end{tabular}
\\label{tab:descriptive_stats}
\\end{table}
"""
    
    tables["Descriptive_Stats"] = desc_table
    
    # Table for RQ1
    rq1_table = """\\begin{table}[htbp]
\\centering
\\caption{RQ1: Effect of Individual Modalities on Reconstruction Error}
\\begin{tabular}{lcc}
\\toprule
\\textbf{Comparison} & \\textbf{Median Change (\\%)} & \\textbf{Effect Size} \\\\
\\midrule
"""
    
    # NAIP vs Baseline
    naip = results["naip_vs_baseline"]
    p = naip["p_value"]
    
    rq1_table += f"NAIP vs Baseline & {format_with_bold(naip['percent_change'], p)} ({format_p_value(p)}) & {format_with_bold(naip['effect_size'], p, True)} \\\\\n"
    
    # UAVSAR vs Baseline
    uav = results["uavsar_vs_baseline"]
    p = uav["p_value"]
    
    rq1_table += f"UAVSAR vs Baseline & {format_with_bold(uav['percent_change'], p)} ({format_p_value(p)}) & {format_with_bold(uav['effect_size'], p, True)} \\\\\n"
    
    rq1_table += """\\bottomrule
\\end{tabular}
\\label{tab:rq1_results}
\\end{table}
"""
    
    tables["RQ1"] = rq1_table
    
    # Table for RQ2
    rq2_table = """\\begin{table}[htbp]
\\centering
\\caption{RQ2: Effect of Fusing Multiple Modalities on Reconstruction Error}
\\begin{tabular}{lcc}
\\toprule
\\textbf{Comparison} & \\textbf{Median Change (\\%)} & \\textbf{Effect Size} \\\\
\\midrule
"""
    
    # Fused vs NAIP
    fused = results["fused_vs_naip"]
    p = fused["p_value"]
    
    rq2_table += f"Fused vs NAIP & {format_with_bold(fused['percent_change'], p)} ({format_p_value(p)}) & {format_with_bold(fused['effect_size'], p, True)} \\\\\n"
    
    rq2_table += """\\bottomrule
\\end{tabular}
\\label{tab:rq2_results}
\\end{table}
"""
    
    tables["RQ2"] = rq2_table
    
    # Table for RQ3
    rq3_table = """\\begin{table}[htbp]
\\centering
\\caption{RQ3: Correlation Between Reconstruction Error and Canopy Height Change}
\\begin{tabular}{lcc}
\\toprule
\\textbf{Model} & \\textbf{Spearman ρ} & \\textbf{p-value} \\\\
\\midrule
"""
    
    # Correlations for each model
    corr = results["correlation"]
    
    # Baseline
    p = corr["p_baseline"]
    rq3_table += f"Baseline & {format_with_bold(corr['rho_baseline'], p, True)} & {format_p_value(p)} \\\\\n"
    
    # NAIP
    p = corr["p_naip"]
    rq3_table += f"NAIP & {format_with_bold(corr['rho_naip'], p, True)} & {format_p_value(p)} \\\\\n"
    
    # UAVSAR
    p = corr["p_uavsar"]
    rq3_table += f"UAVSAR & {format_with_bold(corr['rho_uavsar'], p, True)} & {format_p_value(p)} \\\\\n"
    
    # Fused
    p = corr["p_fused"]
    rq3_table += f"Fused & {format_with_bold(corr['rho_fused'], p, True)} & {format_p_value(p)} \\\\\n"
    
    rq3_table += "\\midrule\n"
    
    # Baseline vs NAIP comparison
    z_naip = corr["z_naip"]
    p_z_naip = corr["p_z_naip"]
    rq3_table += f"Baseline vs NAIP (z) & {format_with_bold(z_naip, p_z_naip, True)} & {format_p_value(p_z_naip)} \\\\\n"
    
    # Baseline vs UAVSAR comparison
    z_uav = corr["z_uav"]
    p_z_uav = corr["p_z_uav"]
    rq3_table += f"Baseline vs UAVSAR (z) & {format_with_bold(z_uav, p_z_uav, True)} & {format_p_value(p_z_uav)} \\\\\n"
    
    # Baseline vs Fused comparison
    z = corr["z_value"]
    p_z = corr["p_z"]
    rq3_table += f"Baseline vs Fused (z) & {format_with_bold(z, p_z, True)} & {format_p_value(p_z)} \\\\\n"
    
    rq3_table += """\\bottomrule
\\end{tabular}
\\label{tab:rq3_results}
\\end{table}
"""
    
    tables["RQ3"] = rq3_table
    
    # Table for RQ3 Extended (gains vs losses)
    rq3_ext_table = """\\begin{table}[htbp]
\\centering
\\caption{RQ3 Extended: Correlation Between Reconstruction Error and Canopy Height Changes (Gains vs. Losses)}
\\begin{tabular}{lcccc}
\\toprule
\\multirow{2}{*}{\\textbf{Model}} & \\multicolumn{2}{c}{\\textbf{Canopy Gains}} & \\multicolumn{2}{c}{\\textbf{Canopy Losses}} \\\\
\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}
 & \\textbf{Spearman ρ} & \\textbf{p-value} & \\textbf{Spearman ρ} & \\textbf{p-value} \\\\
\\midrule
"""
    
    # Get gain and loss data
    gains = results["correlation_gains"]
    losses = results["correlation_losses"]
    
    # Add counts to table caption
    n_gains = gains["n"]
    n_losses = losses["n"]
    
    # Update caption with sample sizes
    rq3_ext_table = rq3_ext_table.replace("Canopy Gains", f"Canopy Gains (N={n_gains})")
    rq3_ext_table = rq3_ext_table.replace("Canopy Losses", f"Canopy Losses (N={n_losses})")
    
    # Model data rows
    for model in ["baseline", "naip", "uavsar", "fused"]:
        # Gains
        g_rho = gains[f"rho_{model}"]
        g_p = gains[f"p_{model}"]
        g_formatted = f"{format_with_bold(g_rho, g_p, True)} & {format_p_value(g_p)}"
        
        # Losses
        l_rho = losses[f"rho_{model}"]
        l_p = losses[f"p_{model}"]
        l_formatted = f"{format_with_bold(l_rho, l_p, True)} & {format_p_value(l_p)}"
        
        # Full row
        rq3_ext_table += f"{model.capitalize()} & {g_formatted} & {l_formatted} \\\\\n"
    
    rq3_ext_table += "\\midrule\n"
    
    # Fisher z comparisons
    for model in ["naip", "uavsar", "fused"]:
        comparison = f"Baseline vs {model.capitalize()} (z)"
        
        # Gains
        g_z = gains[f"z_{model}"]
        g_p_z = gains[f"p_z_{model}"]
        g_formatted = f"{format_with_bold(g_z, g_p_z, True)} & {format_p_value(g_p_z)}"
        
        # Losses
        l_z = losses[f"z_{model}"]
        l_p_z = losses[f"p_z_{model}"]
        l_formatted = f"{format_with_bold(l_z, l_p_z, True)} & {format_p_value(l_p_z)}"
        
        # Full row
        rq3_ext_table += f"{comparison} & {g_formatted} & {l_formatted} \\\\\n"
    
    rq3_ext_table += """\\bottomrule
\\end{tabular}
\\label{tab:rq3_extended_results}
\\end{table}
"""
    
    tables["RQ3_Extended"] = rq3_ext_table
    
    return tables

def save_tables_to_file(tables, output_file):
    """Save all LaTeX tables to a file."""
    with open(output_file, 'w') as f:
        f.write("% LaTeX tables for research questions\n\n")
        f.write("% Required packages:\n")
        f.write("% \\usepackage{booktabs}\n")
        f.write("% \\usepackage{multirow}\n\n")
        
        for rq, table in tables.items():
            f.write(f"% {rq} Table\n")
            f.write(table)
            f.write("\n\n")

def main(df_path, output_file=None):
    """Main function to run analysis and generate tables."""
    results = analyze_data(df_path)
    tables = generate_latex_tables(results)
    
    if output_file:
        save_tables_to_file(tables, output_file)
    else:
        for rq, table in tables.items():
            print(f"\n--- {rq} Table ---")
            print(table)

if __name__ == "__main__":
    # Configure parameters here
    df_path = "data/processed/model_data/point_cloud_comparison_df_0516_e100.pkl"
    output_file = None #"results_tables.tex"  # Set to None to print to console instead
    main(df_path, output_file)