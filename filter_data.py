# import pandas as pd
# import sys
# import os

# INPUT  = 'data_main.csv'
# OUTPUT = 'data_main_filtered.csv'
# CHUNK  = 5_000  # adjust for your memory/speed trade-off

# # Remove old output if it exists
# if os.path.exists(OUTPUT):
#     os.remove(OUTPUT)

# reader = pd.read_csv(INPUT, chunksize=CHUNK)

# rows_seen = 0
# header_written = False

# for chunk in reader:
#     # filter .html rows
#     filtered = chunk[chunk['file name'].str.endswith('.html', na=False)]
#     # append to output
#     filtered.to_csv(OUTPUT, 
#                     mode='a', 
#                     index=False, 
#                     header=not header_written)
#     header_written = True

#     # update & show progress
#     rows_seen += len(chunk)
#     sys.stdout.write(f'\rRows scanned: {rows_seen}')
#     sys.stdout.flush()

# print("\nDone — filtered file is", OUTPUT)

# os.replace(OUTPUT, INPUT)

import pandas as pd

df = pd.read_csv("data_main.csv")

print(len(df))
print(list(df.columns))


