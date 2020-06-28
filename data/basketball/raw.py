import os

import numpy as np
import pandas as pd


class Possession:
    def __init__(self, raw_dir, team_no, poss_no):
        self.raw_dir = raw_dir
        self.team_no = team_no
        self.poss_no = poss_no

        self.optical = None
        self.pbp = None
        self.timestamps = None
        self.meta = None

    def read_files(self):
        self._read_optical()
        self._read_meta()
        self._read_pbp()
        self._read_timestamps()

        self._normalize_coords()

    def _read_optical(self):
        self.optical = np.genfromtxt(os.path.join(
            self.raw_dir, f'Team_{self.team_no}',
            f'possession_{self.poss_no}_optical'),
                                     delimiter=',')

    def _read_pbp(self):
        self.pbp = np.genfromtxt(os.path.join(
            self.raw_dir, f'Team_{self.team_no}',
            f'possession_{self.poss_no}_pbp'),
                                 delimiter=',')

    def _read_timestamps(self):
        self.timestamps = np.genfromtxt(os.path.join(
            self.raw_dir, f'Team_{self.team_no}',
            f'possession_{self.poss_no}_timeStamps'),
                                        delimiter=',')

    def _read_meta(self):
        self.meta = np.genfromtxt(os.path.join(
            self.raw_dir, f'Team_{self.team_no}',
            f'possession_{self.poss_no}_meta'),
                                  delimiter=',')

    def create_df(self):
        col_names = [
            'team_no', 'possession_no', 'outcome', 'teamID_A', 'teamID_B',
            'teamA_direction', 'possession', 'frame_start', 'frame_end',
            'pbp_start', 'pbp_end', 'duration', 'timestamps', 'pbp1', 'pbp2',
            'pbp3', 'pbp4', 'ball_x', 'ball_y', 'ball_info3', 'bid1', 'bid2',
            'bid3', 'bid4', 'bid5', 'teamA_p1_x', 'teamA_p1_y', 'teamA_p2_x',
            'teamA_p2_y', 'teamA_p3_x', 'teamA_p3_y', 'teamA_p4_x',
            'teamA_p4_y', 'teamA_p5_x', 'teamA_p5_y', 'teamB_p1_x',
            'teamB_p1_y', 'teamB_p2_x', 'teamB_p2_y', 'teamB_p3_x',
            'teamB_p3_y', 'teamB_p4_x', 'teamB_p4_y', 'teamB_p5_x',
            'teamB_p5_y', 'teamA_p1_ID', 'teamA_p2_ID', 'teamA_p3_ID',
            'teamA_p4_ID', 'teamA_p5_ID', 'teamB_p1_ID', 'teamB_p2_ID',
            'teamB_p3_ID', 'teamB_p4_ID', 'teamB_p5_ID', 'bh_ID', 'bh_x',
            'bh_y', 'bh_dist_from_ball', 'bh_dist_from_basket',
            'bh_angle_from_basket', 'def1_ID', 'def2_ID', 'def3_ID', 'def4_ID',
            'def5_ID', 'def1_x', 'def1_y', 'def2_x', 'def2_y', 'def3_x',
            'def3_y', 'def4_x', 'def4_y', 'def5_x', 'def5_y',
            'def1_rel_x_from_bh', 'def1_rel_y_from_bh', 'def2_rel_x_from_bh',
            'def2_rel_y_from_bh', 'def3_rel_x_from_bh', 'def3_rel_y_from_bh',
            'def4_rel_x_from_bh', 'def4_rel_y_from_bh', 'def5_rel_x_from_bh',
            'def5_rel_y_from_bh', 'def1_dist_from_bh', 'def2_dist_from_bh',
            'def3_dist_from_bh', 'def4_dist_from_bh', 'def5_dist_from_bh',
            'def1_rel_angle_from_bh', 'def2_rel_angle_from_bh',
            'def3_rel_angle_from_bh', 'def4_rel_angle_from_bh',
            'def5_rel_angle_from_bh', 'def1_trunc_x', 'def1_trunc_y',
            'def2_trunc_x', 'def2_trunc_y', 'def3_trunc_x', 'def3_trunc_y',
            'def4_trunc_x', 'def4_trunc_y', 'def5_trunc_x', 'def5_trunc_y',
            'shoot_label'
        ]

        msg = [self.team_no, self.poss_no, '']

        # Exceptions
        # Pbp 1-dimensional
        if len(self.pbp.shape) == 1:
            msg[2] = 'PBP 1-dimensional'
            return msg, pd.DataFrame(columns=col_names)

        # # Remove duplicate rows from self.pbp while preserving order
        # _, idx = np.unique(self.pbp, axis=0, return_index=True)
        # if len(idx) != self.pbp.shape[0]:
        #     print(self.team_no, self.poss_no)
        # self.pbp = self.pbp[np.sort(idx)]

        # Check if pbp has a shooting event that is not the last frame and
        # truncate frames
        argwhere_res = np.argwhere((self.pbp[:, 1] == 3.)
                                   | (self.pbp[:, 1] == 4.))
        if argwhere_res.size != 0:
            pbp_idx = argwhere_res[-1].item()
            # if argwhere_res.size != 1:
            #     print(self.team_no, self.poss_no)
            if pbp_idx != self.pbp.shape[0] - 1:
                timestamp_idx = (self.timestamps > self.pbp[pbp_idx, 2]).sum()
                self.pbp = self.pbp[:pbp_idx + 1, :]
                self.optical = self.optical[:timestamp_idx + 1, :]
                self.timestamps = self.timestamps[:timestamp_idx + 1]
                msg[2] = 'Truncated shooting'

        df = pd.DataFrame(index=range(self.timestamps.shape[0]),
                          columns=col_names)

        df.loc[:, 'team_no'] = self.team_no
        df.loc[:, 'possession_no'] = self.poss_no
        df.loc[:, 'outcome':'duration'] = self.meta
        df.loc[:, 'timestamps'] = self.timestamps

        # Get number of repetitions for pbp
        reps = np.zeros(self.pbp.shape[0], dtype=int)
        idx = 0
        for i in range(1, self.pbp.shape[0]):
            rep = 0
            while self.timestamps[idx] > self.pbp[i, 2]:
                idx += 1
                rep += 1
            reps[i - 1] = rep

        # If there are two events associated with last frame, use second to
        # last event ID
        if reps[-2] == 0 and self.timestamps.shape[0] - np.sum(reps) == 1:
            reps[-2] = self.timestamps.shape[0] - np.sum(reps)
        reps[-1] = self.timestamps.shape[0] - np.sum(reps)

        # Set pbp
        df.loc[:, 'pbp1':'pbp4'] = np.repeat(self.pbp, reps, axis=0)

        # Set optical
        df.loc[:, 'ball_x':'teamB_p5_ID'] = self.optical

        # Set dtypes
        int_cols = [
            'team_no',
            'possession_no',
            'outcome',
            'teamID_A',
            'teamID_B',
            'teamA_direction',
            'possession',
            'frame_start',
            'frame_end',
            'pbp_start',
            'pbp_end',
            'pbp1',
            'pbp2',
            'pbp4',
            'pbp1',
            'pbp2',
            'pbp4',
            'bid1',
            'bid2',
            'bid3',
            'bid4',
            'bid5',
            'teamA_p1_ID',
            'teamA_p2_ID',
            'teamA_p3_ID',
            'teamA_p4_ID',
            'teamA_p5_ID',
            'teamB_p1_ID',
            'teamB_p2_ID',
            'teamB_p3_ID',
            'teamB_p4_ID',
            'teamB_p5_ID',
        ]
        float_cols = [
            'duration',
            'timestamps',
            'pbp3',
            'ball_x',
            'ball_y',
            'ball_info3',
            'teamA_p1_x',
            'teamA_p1_y',
            'teamA_p2_x',
            'teamA_p2_y',
            'teamA_p3_x',
            'teamA_p3_y',
            'teamA_p4_x',
            'teamA_p4_y',
            'teamA_p5_x',
            'teamA_p5_y',
            'teamB_p1_x',
            'teamB_p1_y',
            'teamB_p2_x',
            'teamB_p2_y',
            'teamB_p3_x',
            'teamB_p3_y',
            'teamB_p4_x',
            'teamB_p4_y',
            'teamB_p5_x',
            'teamB_p5_y',
        ]
        df[int_cols] = df[int_cols].astype(int)
        df[float_cols] = df[float_cols].astype(float)

        # Filters
        # Filter: sum of bid1~bid5 == 1.
        # Use bid1~bid5 to get bh_ID
        df = df[df[['bid1', 'bid2', 'bid3', 'bid4', 'bid5']].sum(axis=1) == 1]
        # Exceptions
        if df.shape[0] == 0:
            msg[2] = 'Sum(bid1,...,bid5) != 1'
            return msg, pd.DataFrame(columns=col_names)

        # Delete possession if ball_x=46.9980, ball_y=25, ball_info3=0.25
        # (out of bounds)
        ball_out_of_bounds = pd.Series({
            'ball_x': 46.998,
            'ball_y': 25,
            'ball_info3': 0.25
        })
        if (df[['ball_x', 'ball_y', 'ball_info3']]
                == ball_out_of_bounds).any(axis=1).sum() > 0:
            msg[2] = 'Out of bounds ball_info'
            return msg, pd.DataFrame(columns=col_names)

        # Delete frames if pos_x not in boundaries [0, 40) for all players and
        # ball
        x_cols = [
            'ball_x', 'teamA_p1_x', 'teamA_p2_x', 'teamA_p3_x', 'teamA_p4_x',
            'teamA_p5_x', 'teamB_p1_x', 'teamB_p2_x', 'teamB_p3_x',
            'teamB_p4_x', 'teamB_p5_x'
        ]
        df = df.loc[(df.loc[:, x_cols] >= 0).all(axis=1)
                    & (df.loc[:, x_cols] < 40).all(axis=1)]

        # Delete frames if pos_y not in boundaries [0, 50) for all players and
        # ball
        y_cols = [
            'ball_y', 'teamA_p1_y', 'teamA_p2_y', 'teamA_p3_y', 'teamA_p4_y',
            'teamA_p5_y', 'teamB_p1_y', 'teamB_p2_y', 'teamB_p3_y',
            'teamB_p4_y', 'teamB_p5_y'
        ]
        df = df.loc[(df.loc[:, y_cols] >= 0).all(axis=1)
                    & (df.loc[:, y_cols] < 50).all(axis=1)]

        # Ball_pos != 0
        df = df.loc[(df['ball_x'] != 0.) & (df['ball_y'] != 0.)]
        # Exceptions
        if df.shape[0] == 0:
            msg[2] = 'Ball info == 0'
            return msg, pd.DataFrame(columns=col_names)

        # Events == 3, 4, 5, 6, 7, 21, 23, 24
        allowed_event_IDs = [3, 4, 5, 6, 7, 21, 23, 24]
        df = df.loc[df['pbp2'].isin(allowed_event_IDs)]
        # Exceptions
        if df.shape[0] == 0:
            msg[2] = 'Invalid Event IDs'
            return msg, pd.DataFrame(columns=col_names)

        # df = df.reset_index(drop=True)

        # Calculate new columns
        # Calc bh_ID
        def calc_bh_ID(row, teamA_ID_start_idx, teamB_ID_start_idx):
            idx = row.loc['bid1':'bid5'].values.nonzero()[0].item()
            if row['possession'] == 1.:
                return row.iloc[teamA_ID_start_idx + idx]
            else:
                return row.iloc[teamB_ID_start_idx + idx]

        teamA_ID_start_idx = df.columns.get_loc('teamA_p1_ID')
        teamB_ID_start_idx = df.columns.get_loc('teamB_p1_ID')
        df['bh_ID'] = df.apply(lambda row: calc_bh_ID(row, teamA_ID_start_idx,
                                                      teamB_ID_start_idx),
                               axis=1)

        def calc_bh_pos(row, teamA_pos_start_idx, teamB_pos_start_idx):
            idx = row.loc['bid1':'bid5'].values.nonzero()[0].item()
            if row['possession'] == 1.:
                return row.iloc[teamA_pos_start_idx +
                                2 * idx], row.iloc[teamA_pos_start_idx +
                                                   2 * idx + 1]
            else:
                return row.iloc[teamB_pos_start_idx +
                                2 * idx], row.iloc[teamB_pos_start_idx +
                                                   2 * idx + 1]

        def dist(x1, y1, x2, y2):
            return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        def angle(x, y, origin_x, origin_y):
            return np.degrees(np.arctan2((y - origin_y), (x - origin_x)))

        # Calculate bh_pos, bh_dist_from_ball
        teamA_pos_start_idx = df.columns.get_loc('teamA_p1_x')
        teamB_pos_start_idx = df.columns.get_loc('teamB_p1_x')
        df['bh_x'], df['bh_y'] = zip(*df.apply(lambda row: calc_bh_pos(
            row, teamA_pos_start_idx, teamB_pos_start_idx),
                                               axis=1))
        df['bh_dist_from_ball'] = dist(df['bh_x'], df['bh_y'], df['ball_x'],
                                       df['ball_y'])

        # Calculate bh_dist_from_basket
        df['bh_dist_from_basket'] = dist(df['bh_x'], df['bh_y'], 5.2493, 25.)

        # Calculate angle of bh from basket
        df['bh_angle_from_basket'] = angle(df['bh_x'], df['bh_y'], 5.2493, 25.)

        # Calculate def_pos, def_ID of defenders
        def_cols = [
            'def1_x', 'def1_y', 'def2_x', 'def2_y', 'def3_x', 'def3_y',
            'def4_x', 'def4_y', 'def5_x', 'def5_y', 'def1_ID', 'def2_ID',
            'def3_ID', 'def4_ID', 'def5_ID'
        ]
        teamA_cols = [
            'teamA_p1_x', 'teamA_p1_y', 'teamA_p2_x', 'teamA_p2_y',
            'teamA_p3_x', 'teamA_p3_y', 'teamA_p4_x', 'teamA_p4_y',
            'teamA_p5_x', 'teamA_p5_y', 'teamA_p1_ID', 'teamA_p2_ID',
            'teamA_p3_ID', 'teamA_p4_ID', 'teamA_p5_ID'
        ]
        teamB_cols = [
            'teamB_p1_x', 'teamB_p1_y', 'teamB_p2_x', 'teamB_p2_y',
            'teamB_p3_x', 'teamB_p3_y', 'teamB_p4_x', 'teamB_p4_y',
            'teamB_p5_x', 'teamB_p5_y', 'teamB_p1_ID', 'teamB_p2_ID',
            'teamB_p3_ID', 'teamB_p4_ID', 'teamB_p5_ID'
        ]
        if self.meta[4] == 1.:
            df[def_cols] = df[teamB_cols]
        else:
            df[def_cols] = df[teamA_cols]

        # Calculate def_dist_from_bh
        df['def1_dist_from_bh'] = dist(df['bh_x'], df['bh_y'], df['def1_x'],
                                       df['def1_y'])
        df['def2_dist_from_bh'] = dist(df['bh_x'], df['bh_y'], df['def2_x'],
                                       df['def2_y'])
        df['def3_dist_from_bh'] = dist(df['bh_x'], df['bh_y'], df['def3_x'],
                                       df['def3_y'])
        df['def4_dist_from_bh'] = dist(df['bh_x'], df['bh_y'], df['def4_x'],
                                       df['def4_y'])
        df['def5_dist_from_bh'] = dist(df['bh_x'], df['bh_y'], df['def5_x'],
                                       df['def5_y'])

        # Calculate def_rel_pos_from_bh
        def calc_rot_matrix(theta):
            rad = np.radians(theta)
            cosines, sines = np.cos(rad), np.sin(rad)
            return np.array([[[c, -s], [s, c]]
                             for c, s in zip(cosines, sines)])

        rot_matrices = calc_rot_matrix(90 + df['bh_angle_from_basket'])
        bh_rot_pos = np.einsum('abc,ac->ab', rot_matrices,
                               df.loc[:, 'bh_x':'bh_y'])
        def_rot_pos = np.einsum(
            'abc,adc->adb', rot_matrices,
            df.loc[:, 'def1_x':'def5_y'].to_numpy().reshape(-1, 5, 2))

        def_rel_cols = [
            'def1_rel_x_from_bh', 'def1_rel_y_from_bh', 'def2_rel_x_from_bh',
            'def2_rel_y_from_bh', 'def3_rel_x_from_bh', 'def3_rel_y_from_bh',
            'def4_rel_x_from_bh', 'def4_rel_y_from_bh', 'def5_rel_x_from_bh',
            'def5_rel_y_from_bh'
        ]
        df.loc[:,
               def_rel_cols] = (def_rot_pos - bh_rot_pos[:, None, :]).reshape(
                   -1, 10)

        # Calculate def_rel_angle_from_bh
        df['def1_rel_angle_from_bh'] = angle(df['def1_rel_x_from_bh'],
                                             df['def1_rel_y_from_bh'], 0, 0)
        df['def2_rel_angle_from_bh'] = angle(df['def2_rel_x_from_bh'],
                                             df['def2_rel_y_from_bh'], 0, 0)
        df['def3_rel_angle_from_bh'] = angle(df['def3_rel_x_from_bh'],
                                             df['def3_rel_y_from_bh'], 0, 0)
        df['def4_rel_angle_from_bh'] = angle(df['def4_rel_x_from_bh'],
                                             df['def4_rel_y_from_bh'], 0, 0)
        df['def5_rel_angle_from_bh'] = angle(df['def5_rel_x_from_bh'],
                                             df['def5_rel_y_from_bh'], 0, 0)

        # Calculate def_trunc_pos
        def def_in_boundaries(x, y):
            return (x >= -6) & (x < 6) & (y >= -2) & (y < 10)

        keep = False
        for d in ['def1', 'def2', 'def3', 'def4', 'def5']:
            in_boundaries = def_in_boundaries(df.loc[:, f'{d}_rel_x_from_bh'],
                                              df.loc[:, f'{d}_rel_y_from_bh'])
            df.loc[:,
                   f'{d}_trunc_x'] = np.where(in_boundaries,
                                              df.loc[:, f'{d}_rel_x_from_bh'],
                                              -100.)
            df.loc[:,
                   f'{d}_trunc_y'] = np.where(in_boundaries,
                                              df.loc[:, f'{d}_rel_y_from_bh'],
                                              -100.)
            keep = keep | in_boundaries

        # Calculate shoot_label (shoots within the next second)
        shoot_times = df[df['pbp2'].isin([3, 4])].timestamps
        for t in shoot_times:
            df.loc[df['timestamps'] - t <= 1, 'shoot_label'] = 1
        df['shoot_label'] = df['shoot_label'].fillna(value=0)

        # Additional filters
        # At least one def has position x in [-6, 6) and y in [-2, 10)
        df = df[keep]

        # Dist(ball, bh) < 4 ft
        df = df.loc[df['bh_dist_from_ball'] <= 4.]

        df = df.reset_index(drop=True)

        return msg, df

    def _normalize_coords(self):
        normalized = self.optical.copy()
        if self.meta[4] != self.meta[3]:
            # Team A
            normalized[:,
                       range(8, 18, 2
                             )] = 93.996 - self.optical[:, range(8, 18, 2)]
            normalized[:, range(9, 18, 2)] = 50 - self.optical[:,
                                                               range(9, 18, 2)]

            # Team B
            normalized[:,
                       range(18, 28, 2
                             )] = 93.996 - self.optical[:, range(18, 28, 2)]
            normalized[:,
                       range(19, 28, 2)] = 50 - self.optical[:,
                                                             range(19, 28, 2)]

            # Ball
            normalized[:, 0] = 93.996 - self.optical[:, 0]
            normalized[:, 1] = 50 - self.optical[:, 1]

        self.optical = normalized
