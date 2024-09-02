import os
import sys
import json
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

sys.path.append('..')
from forking_paths import st_plot, st_utils, text_viz


st.set_page_config(
    layout='wide',       # 'centered' or 'wide'
    page_title='Individual FPA',
    page_icon='📈'
)


BASE_DIR = './data'
# qa_list = [f'{task}/{qa_id}'   #  f'{task}/{qa_id.split(".")[0]}'
#               for task in os.listdir(BASE_DIR)
#               for qa_id in os.listdir(f'{BASE_DIR}/{task}')]

# # 0. drop-down selector for which csv (task/qa id) to load
# selected_qa = st.selectbox(
#     'Select a file (`task/question_id`):',
#     qa_list,
# )

col1, col2 = st.columns(2)

task_list = sorted(os.listdir(BASE_DIR))

# 0. drop-down selector for which csv (task/qa id) to load
task = col1.selectbox(
    'Select a task:',
    task_list,
)

qa_list = sorted(os.listdir(f'{BASE_DIR}/{task}'))

# 0. drop-down selector for which csv (task/qa id) to load
qa = col2.selectbox(
    'Select a question ID:',
    qa_list,
)

selected_qa = f'{task}/{qa}'

@st.cache_data
def load_data(selected_qa):
    ans_df     = pd.read_csv(f'{BASE_DIR}/{selected_qa}/ans_df.csv')
    idx_df     = pd.read_csv(f'{BASE_DIR}/{selected_qa}/idx_df.csv')
    idx_tok_df = pd.read_csv(f'{BASE_DIR}/{selected_qa}/idx_tok_df.csv')

    base_res = pickle.load(open(f'{BASE_DIR}/{selected_qa}/base_res.pk', 'rb'))
    args_json = json.load(open(f'{BASE_DIR}/{selected_qa}/args.json'))
    return ans_df, idx_df, idx_tok_df, base_res, args_json


ans_df, idx_df, idx_tok_df, base_res, args_json = load_data(selected_qa)
base_tokens = base_res[0]['choice']['logprobs']['tokens']



outcomes_set = set(idx_df['ans'])
idx_df['ans'] = pd.Categorical(idx_df['ans'], categories=outcomes_set)
idx_tok_df['ans'] = pd.Categorical(idx_tok_df['ans'], categories=outcomes_set)

outcome_colors = sns.color_palette(palette='viridis_r', n_colors=len(outcomes_set) - 1) + [(.9, .3, .1)]



o_trend = st_utils.run_beast(idx_df)
logits = st_utils.get_logits(base_res)

answer_fn = st_utils.ans_key_map.get(task, lambda x: 'N/A')
answer = answer_fn(args_json['row'])


# Highlighted text
#    - https://docs.streamlit.io/develop/api-reference/custom-components/st.components.v1.html
col1, col2 = st.columns(2)
if col1.checkbox('Highlighted Text', value=1):
    cpd_probs = 1 - o_trend['cpOccPr']

    print_newln = col2.checkbox('Print Newln', value=0)

    cols = st.columns(3)
    cols[0].html( text_viz.text_base_html(base_res[0]['prompt_raw'], print_newln=True, width='100%', bg_color='#cfffd3'))
    cols[0].markdown('Prompt')
    cols[0].html('<span style="color: black; background-color: #00ff26; font-size: 22px">&nbsp; Answer: &nbsp;' + answer + '&nbsp; &nbsp;</span>')
    cols[1].html( text_viz.text_colors_html(base_tokens, logits, print_newln=print_newln, width='100%'))
    cols[1].markdown('Token Logit Probs')
    cols[2].html( text_viz.text_colors_html(base_tokens, cpd_probs, print_newln=print_newln, width='100%'))
    cols[2].markdown('Change Point Detection Probs   $\quad  p(\\tau = t)$')




# Stacked line plot    (+ baseline bars ????)
if st.checkbox('Outcome Time Series  $\quad  o_t$', value=1):
    fig_stack = st_plot.plotly_stacked(idx_df, base_tokens, outcome_colors, outcomes_set)
    st.plotly_chart(fig_stack)


# Forking tokens sankey
# 	- update when clicking on time series (any of them)
if st.checkbox('Token Outcome Distributions  $\quad  o_{t, w}$', value=1):
    t_min, t_max = min(idx_tok_df['idx']), max(idx_tok_df['idx'])
    t_select = st.slider('Token Index $t$', min_value=t_min, max_value=t_max - 4)

    fig_parcat = st_plot.plotly_parallel_categories(idx_tok_df, base_tokens, outcome_colors,
                                                    plot_idx=t_select, plot_n_idxs=4)   # pd.DataFrame(st.session_state.ts_click)['idx'].iloc[0]
    st.plotly_chart(fig_parcat)



# Semantic drift
if st.checkbox('Semantic Drift  $\quad  y_t$', value=0):
    dist = st.selectbox(
        'Distance Function',
        ['l1', 'l2', 'cos', 'kl'],
        key=0,
    )
    fig_drift = st_plot.plotly_semantic_drift(idx_df, base_tokens, dist_fn=f'd_{dist}')
    st.plotly_chart(fig_drift)

# Change point detection      p(tau = t)
if st.checkbox('Change Point Detection  $\quad  p(\\tau = t)$', value=1):
    fig_cpd = st_plot.plotly_cpd_line(o_trend, base_tokens)
    st.plotly_chart(fig_cpd)

# Forking survival time series
if st.checkbox('Forking Token Survival Analysis', value=1):
    dist = st.selectbox(
        'Distance Function',
        ['l1', 'l2', 'cos', 'kl'],
        key=1,
    )
    fig_h, fig_s = st_plot.survival_lines(idx_tok_df, base_tokens,
                                          dist_fn=f'd_{dist}', d_thresholds=[0, .1, .2, .3, .5, .7])
    st.plotly_chart(fig_h)
    st.plotly_chart(fig_s)


# Baseline -- token log probs
if st.checkbox('Baseline: Token Logit Probs', value=1):
    fig_lp = st_plot.plotly_logprob_line(logits, base_tokens)
    st.plotly_chart(fig_lp)




