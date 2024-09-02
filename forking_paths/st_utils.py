import numpy as np

import streamlit as st
import Rbeast

from forking_paths.analysis_utils import get_dists


alpha2_d = {'d_kl': lambda range_y: 10.0 + (1000 ** (1.0 - range_y)),
            'd_l1': lambda range_y: 2.0  + (1000 ** (1.0 - range_y)),
            'd_l2': lambda range_y: 5.0  + (1000 ** (1.0 - range_y))}

# alpha2 = alpha2_d[dist_fn]


def semantic_drift(idx_df):
    dist_df = get_dists(idx_df)
    dist_df = dist_df.sort_values(by='idx')
    return dist_df

@st.cache_data
def run_beast(idx_df, dist_fn = 'd_l1', 
              alpha2_str='lambda range_y: 2.0  + (1000 ** (1.0 - range_y))'):
    ### TODO: turn all beast args into kwarg dict + default dict   
    #                   -- then use default dict to generate that many text entry boxes in streamlit CPD page
    alpha2_fn = eval(alpha2_str)

    dist_df = get_dists(idx_df)
    dist_df = dist_df.sort_values(by='idx')

    y = dist_df[dist_fn].to_numpy()
    time  = dist_df['idx'].to_numpy()

    range_y = y.max() - y.min()
    alpha2 = alpha2_fn(range_y)

    o = Rbeast.beast(
        y, time=time, season='none',
        tcp_minmax=[0, 6],  torder_minmax=[1, 1], tseg_minlength=10,
        mcmc_seed=0, mcmc_chains=10,
        mcmc_burnin=1000, mcmc_samples=20000, mcmc_thin=5,
        # mcmc_burnin=200, mcmc_samples=8000, mcmc_thin=5,
        print_progress=False, print_options=False, quiet=True, 

        precPriorType='constant', precValue=10,      # manually set \nu = 10   (TODO: should this be 1.5?)
        alpha1=.01,                 # https://github.com/zhaokg/Rbeast/blob/master/R/src/beastv2_io_in_args.c#L898
        alpha2=alpha2,              # min alpha2: MIN_ALPHA2_VALUE=.0001
        #### Default values for alpha1/2, delta1/2: 1.0   (or 1e-8??)
        ####   Source: https://github.com/zhaokg/Rbeast/blob/master/Source/beastv2_io_in_args.c#L758
        ####   Alt??:  https://github.com/zhaokg/Rbeast/blob/master/R/src/beastv2_io_in_args.c#L898
    )
    return o.trend.__dict__



def get_logits(idx_tok_df, base_tokens):
    t_min, t_max = min(idx_tok_df['idx']), max(idx_tok_df['idx'])
    return [idx_tok_df[(idx_tok_df['idx'] == t) & (idx_tok_df['tok'] == w)]['tok_p'].iloc[0] 
            for t, w in zip(range(t_min, t_max), base_tokens)]


def get_logits(base_res):
    return np.exp(base_res[0]['choice']['logprobs']['token_logprobs'])


ans_key_map = {
    'CoinFlip': lambda row: row['targets'],
    'LastLetter': lambda row: row['answer'],
    'MMLU': lambda row: row['choices'][row['answer']],
    'HotpotQA': lambda row: row['answer'],
    'AQuA': lambda row: row['correct'],
    'GSM8k': lambda row: row['answer'].split('#### ')[1],
}