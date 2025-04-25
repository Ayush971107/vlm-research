import pandas as pd
import multiprocessing as mp
import os
from lxml import html
from zss import Node, simple_distance
from multiprocessing import Pool

# Dom helpers
def parse_html_to_dom(html_str):
    try:
        return html.fromstring(html_str)
    except Exception:
        return html.fromstring("<html><body></body></html>")

def build_zss_tree(dom_node):
    node = Node(dom_node.tag)
    for child in dom_node:
        if isinstance(child.tag, str):
            node.addkid(build_zss_tree(child))
    return node

def compute_tree_edit_distance(pair):
    before, after = pair
    if before == after:
        return 0
    try:
        dom1 = parse_html_to_dom(before)
        dom2 = parse_html_to_dom(after)
        t1 = build_zss_tree(dom1)
        t2 = build_zss_tree(dom2)
        return simple_distance(t1, t2)
    except Exception:
        return None  


if __name__ == "__main__":
    INPUT    = "data_main.csv"
    TEMP     = "data_main.tmp.csv"
    CHUNKSZ  = 5000
    TASK_SIZE = 100           # rows sent to each worker
    WORKERS  = 2             

    pool   = Pool(WORKERS, maxtasksperchild=200)
    reader = pd.read_csv(INPUT, chunksize=CHUNKSZ)

    first_chunk = True
    rows_done   = 0

    for chunk in reader:
        # computes distances
        pairs = list(zip(chunk["page before commit"], chunk["page after commit"]))
        dists = pool.map(compute_tree_edit_distance, pairs, chunksize=TASK_SIZE)
        chunk["dom distance"] = dists

        # write or append
        chunk.to_csv(
            TEMP,
            mode='w' if first_chunk else 'a',
            index=False,
            header=first_chunk
        )
        first_chunk = False

        rows_done += len(chunk)
        print(f"\rRows processed: {rows_done}", end='', flush=True)

    pool.close()
    pool.join()

    # replace original with updated
    os.replace(TEMP, INPUT)
    print("\n✅ data_main.csv updated with dom distance column")