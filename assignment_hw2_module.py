import numpy as np
import pandas as pd

from scipy import stats


def create_player_question_table(dct):
    lst = []
    qnt = 0
    for ids, teams in dct.items():
        for team_id, team in teams.items():
            mask = np.array([np.int32(answer) for answer in team['mask']])
            players = team['players']
            questions = np.tile(np.arange(qnt, qnt + len(mask)), len(players))
            answers = np.array(np.meshgrid(players, mask)).T.reshape(-1, 2)
            answers = np.hstack([
                np.repeat(ids, len(questions)).reshape(-1, 1),
                np.repeat(team_id, len(questions)).reshape(-1, 1),
                answers, 
                questions.reshape(-1, 1)]
            )
            lst.append(answers)
        qnt += len(mask)

    lst_stuck = np.vstack(lst).astype(np.int32)
    df = pd.DataFrame(lst_stuck, columns = ['tournament_id', 'team_id', 'player_id', 'answer', 'question_id'])
    return df
    
def calc_correlation(rating, typ='spearman'):
    if typ == 'spearman':
        return rating.groupby('tournament_id').apply(lambda x: stats.spearmanr(x['real_rank'], x['pred_rank']).correlation).mean()
    return rating.groupby('tournament_id').apply(lambda x: stats.kendalltau(x['real_rank'], x['pred_rank']).correlation).mean()

def create_player_question_table_for_validation(dct, lst_id):
    lst = []
    for ids, teams in dct.items():
        for team_id, team in teams.items():
            mask = np.array([np.int32(answer) for answer in team['mask']])
            for player_id in team['players']:
                if player_id not in lst_id: 
                    continue
                lst.append((ids, team_id, player_id, -99, sum(mask), len(mask))) 
    lst_stuck = np.vstack(lst).astype(np.int32)
    df = pd.DataFrame(lst_stuck, columns = ['tournament_id', 'team_id', 'player_id', 'question_id', 'points', 'n_total'])
    return df