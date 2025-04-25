import pandas as pd

def compute_dom_stats(csv_path, output_path='dom_stats.txt'):
    df = pd.read_csv(csv_path, usecols=['dom distance'])
    zero_count = (df['dom distance'] == 0).sum()
    non_zero = df['dom distance'][df['dom distance'] >= 1]

    count_non_zero = non_zero.count()
    if count_non_zero > 0:
        mean_val   = non_zero.mean()
        min_val    = non_zero.min()
        max_val    = non_zero.max()
        median_val = non_zero.median()
    else:
        mean_val = min_val = max_val = median_val = float('nan')

    with open(output_path, 'w') as f:
        f.write(f"Rows with dom distance = 0: {zero_count}\n")
        f.write(f"Rows with dom distance ≥ 1: {count_non_zero}\n")
        f.write("Statistics for dom distance ≥ 1:\n")
        f.write(f"    Mean:   {mean_val:.2f}\n")
        f.write(f"    Min:    {min_val}\n")
        f.write(f"    Max:    {max_val}\n")
        f.write(f"    Median: {median_val:.2f}\n")

    print(f"Written DOM stats to {output_path}")

if __name__ == "__main__":
    compute_dom_stats("data_main.csv")