import argparse
import os
import time
from multiprocessing.pool import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from raw import Possession

p = argparse.ArgumentParser()
p.add_argument('--input-dir', required=True)
p.add_argument('--output-dir', required=True)
args = p.parse_args()

# raw_dir = '/mldata/basketball/possession_data'


def process_data(args):
    poss = Possession(args[0], args[1], args[2])
    poss.read_files()
    return poss.create_df()


def list_all_possessions(raw_dir):
    p = Path(raw_dir).rglob('Team_*/possession_*_optical')
    all_poss = [[raw_dir,
                 int(f.parent.name[5:]),
                 int(f.name.split('_')[1])] for f in p]
    return all_poss


# Multiprocessing
all_poss = list_all_possessions(args.input_dir)
pool = Pool(processes=8)
results = []
good_poss = []
bad_poss = []
start_time = time.time()
for info, result in tqdm(pool.imap_unordered(process_data, all_poss),
                         total=len(all_poss)):
    results.append(result)
    if info[2] == '':
        good_poss.append(info)
    else:
        bad_poss.append(info)
pool.close()
pool.join()

print('Aggregating results...')
df = pd.concat(results).reset_index(drop=True)
print(f'Finished in {time.time() - start_time:.2f}s')
print(f'Shape: {df.shape}')
print(f'Writing poss files...')
with open(os.path.join(args.output_dir, 'good_poss'), 'w') as f:
    for row in good_poss:
        f.write(f'{row[0]},{row[1]}\n')
with open(os.path.join(args.output_dir, 'bad_poss'), 'w') as f:
    for row in bad_poss:
        f.write(f'{row[0]},{row[1]},{row[2]!s}\n')

# Set dtypes
int_cols = [
    'team_no', 'possession_no', 'outcome', 'teamID_A', 'teamID_B',
    'teamA_direction', 'possession', 'frame_start', 'frame_end', 'pbp_start',
    'pbp_end', 'pbp1', 'pbp2', 'pbp4', 'bid1', 'bid2', 'bid3', 'bid4', 'bid5',
    'teamA_p1_ID', 'teamA_p2_ID', 'teamA_p3_ID', 'teamA_p4_ID', 'teamA_p5_ID',
    'teamB_p1_ID', 'teamB_p2_ID', 'teamB_p3_ID', 'teamB_p4_ID', 'teamB_p5_ID',
    'bh_ID', 'def1_ID', 'def2_ID', 'def3_ID', 'def4_ID', 'def5_ID',
    'shoot_label'
]
df[int_cols] = df[int_cols].astype(np.int32)
print(f'Writing clean raw pickle file...')
df.to_pickle(os.path.join(args.output_dir, 'cleaned_raw.pkl'))

df = pd.read_pickle(os.path.join(args.output_dir, 'cleaned_raw.pkl'))

print(f'Original Shape: {df.shape}')
_, counts = np.unique(df['shoot_label'], return_counts=True)
print(f'Original label percentages: {counts / df.shape[0]}')

# Discard unused columns
cols = [
    'team_no', 'possession_no', 'teamID_A', 'teamID_B', 'timestamps', 'bh_ID',
    'bh_x', 'bh_y', 'bh_dist_from_ball', 'bh_dist_from_basket',
    'bh_angle_from_basket', 'def1_ID', 'def2_ID', 'def3_ID', 'def4_ID',
    'def5_ID', 'def1_dist_from_bh', 'def2_dist_from_bh', 'def3_dist_from_bh',
    'def4_dist_from_bh', 'def5_dist_from_bh', 'def1_rel_angle_from_bh',
    'def2_rel_angle_from_bh', 'def3_rel_angle_from_bh',
    'def4_rel_angle_from_bh', 'def5_rel_angle_from_bh', 'def1_trunc_x',
    'def1_trunc_y', 'def2_trunc_x', 'def2_trunc_y', 'def3_trunc_x',
    'def3_trunc_y', 'def4_trunc_x', 'def4_trunc_y', 'def5_trunc_x',
    'def5_trunc_y', 'shoot_label'
]
df = df[cols]

# Players must have at least n attempted shots (sum(shoot_label == 1) >= n)
df = df[df.groupby('bh_ID').shoot_label.transform('sum') >= 50]

# Reindex bh_ID into a
df['a'] = df['bh_ID'].astype('category').cat.codes

# Reorder columns
cols = [
    'team_no', 'possession_no', 'teamID_A', 'teamID_B', 'timestamps', 'a',
    'bh_ID', 'bh_x', 'bh_y', 'bh_dist_from_ball', 'bh_dist_from_basket',
    'bh_angle_from_basket', 'def1_ID', 'def2_ID', 'def3_ID', 'def4_ID',
    'def5_ID', 'def1_dist_from_bh', 'def2_dist_from_bh', 'def3_dist_from_bh',
    'def4_dist_from_bh', 'def5_dist_from_bh', 'def1_rel_angle_from_bh',
    'def2_rel_angle_from_bh', 'def3_rel_angle_from_bh',
    'def4_rel_angle_from_bh', 'def5_rel_angle_from_bh', 'def1_trunc_x',
    'def1_trunc_y', 'def2_trunc_x', 'def2_trunc_y', 'def3_trunc_x',
    'def3_trunc_y', 'def4_trunc_x', 'def4_trunc_y', 'def5_trunc_x',
    'def5_trunc_y', 'shoot_label'
]
df = df[cols]

print(f'New Shape: {df.shape}')
_, counts = np.unique(df['shoot_label'], return_counts=True)
print(f'New label percentages: {counts / df.shape[0]}')
# print(
#     f"Bh_pos: max= {df.loc[:, 'bh_x':'bh_y'].max()}, min = {df.loc[:, 'bh_x':'bh_y'].min()}"
# )
# print(
#     f"Def_pos_x: max= {df.filter(like='trunc_x').max(axis=0)}, min = {df.filter(like='trunc_x')[df.filter(like='trunc_x') != -100.].min(axis=0)}"
# )
# print(
#     f"Def_pos_y: max= {df.filter(like='trunc_y').max(axis=0)}, min = {df.filter(like='trunc_y')[df.filter(like='trunc_y') != -100.].min(axis=0)}"
# )
# print(
#     f"At least one def for every row: {(df.filter(like='trunc') != -100.).any(axis=1).sum() == df.shape[0]}"
# )

print(f'Writing preprocessed pickle file...')
df.to_pickle(os.path.join(args.output_dir, 'full_data.pkl'))
