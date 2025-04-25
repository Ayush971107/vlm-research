import pandas as pd

INPUT = 'data_main.csv'
CHUNK = 10_000

non_html_count = 0
offenders = set()

for chunk in pd.read_csv(INPUT, chunksize=CHUNK):
    mask = ~chunk['file name'].str.endswith('.html', na=False)
    if mask.any():
        non_html_count += mask.sum()
        offenders.update(chunk.loc[mask, 'file name'].unique())

if non_html_count == 0:
    print("All rows have '.html' file names ✅")
else:
    print(f"{non_html_count} non-HTML rows found:")
    print("Offending extensions:", list(offenders)[:5])