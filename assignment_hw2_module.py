import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def get_scores(data, preds):
    data['pred'] = preds
    data['score'] = data.groupby(['tournament_id', 'team_id'])['pred'].transform(lambda x: 1 - np.prod(1 - x))
    rating = data[['tournament_id', 'team_id', 'points', 'score']].drop_duplicates().reset_index(drop=True)
    rating = rating.sort_values(by=['tournament_id', 'points'], ascending=False)
    rating['real_rank'] = rating.groupby('tournament_id')['points'].transform(lambda x: np.arange(1, len(x) + 1))
    rating = rating.sort_values(by=['tournament_id', 'score'], ascending=False)
    rating['pred_rank'] = rating.groupby('tournament_id')['score'].transform(lambda x: np.arange(1, len(x) + 1))
    rating = rating.astype(np.int32)
    cor1 = rating.groupby('tournament_id').apply(lambda x: stats.spearmanr(x['real_rank'], x['pred_rank']).correlation).mean()
    cor2 = rating.groupby('tournament_id').apply(lambda x: stats.kendalltau(x['real_rank'], x['pred_rank']).correlation).mean()
    return cor1, cor2

def plot_scores(scores):
    spearman = [elem[0] for elem in scores]
    kendalltau = [elem[1] for elem in scores]
    iters = [i+1 for i in range(len(scores))]
    plt.plot(iters, spearman, color='r', label='spearman')
    plt.plot(iters, kendalltau, color='g', label='kendalltau')

    plt.xlabel("iterations")
    plt.ylabel("scores")
    plt.title("")
    plt.legend()
    plt.show()