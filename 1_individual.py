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



# if not st_utils.check_password():
#     st.stop()

st.set_page_config(
    layout='wide',       # 'centered' or 'wide'
    page_title='Individual FPA',
    page_icon='📈'
)


#BASE_DIR = '../out/gpt3.5-instruct-test_gemini_X_POST'
BASE_DIR = './data'


st.markdown('''\
## Forking Paths Analysis

*Anonymized Dashboard*
''')

if st.checkbox('Instructions', value=1):
    st.markdown('''\
- Select a task and question id from the following drop-down menus
- Use check boxes to show/hide plots
- Hover over highlighted text in the first section to show token index
- Click `...` in the top-right corner to switch between light/dark mode

---
''')


cols = st.columns([.3, .3, .4], gap='medium')

task_list = sorted(os.listdir(BASE_DIR))

# 0. drop-down selector for which csv (task/qa id) to load
task = cols[0].selectbox(
    'Select a task:',
    task_list,
    index=task_list.index('HotpotQA')
)

qa_list = sorted(os.listdir(f'{BASE_DIR}/{task}'))

# 0. drop-down selector for which csv (task/qa id) to load
qa = cols[1].selectbox(
    'Select a question ID:',
    qa_list,
    index=qa_list.index('8076') if task == 'HotpotQA' else 0
)

selected_qa = f'{task}/{qa}'

@st.cache_data
def load_data(selected_qa):
    # ans_df     = pd.read_csv(f'{BASE_DIR}/{selected_qa}/ans_df.csv')
    idx_df     = pd.read_csv(f'{BASE_DIR}/{selected_qa}/idx_df.csv')
    idx_tok_df = pd.read_csv(f'{BASE_DIR}/{selected_qa}/idx_tok_df.csv')

    base_res = pickle.load(open(f'{BASE_DIR}/{selected_qa}/base_res.pk', 'rb'))
    args_json = json.load(open(f'{BASE_DIR}/{selected_qa}/args.json'))

    o_trend = pickle.load(open(f'{BASE_DIR}/{selected_qa}/trend.pk', 'rb'))

    return idx_df, idx_tok_df, base_res, args_json, o_trend


idx_df, idx_tok_df, base_res, args_json, o_trend = load_data(selected_qa)
base_tokens = base_res[0]['choice']['logprobs']['tokens']



outcomes_set = sorted(set(idx_df['ans']))
idx_df['ans'] = pd.Categorical(idx_df['ans'], categories=outcomes_set)
idx_tok_df['ans'] = pd.Categorical(idx_tok_df['ans'], categories=outcomes_set)

outcome_colors = sns.color_palette(palette='viridis_r', n_colors=len(outcomes_set))   # - 1) + [(.9, .3, .1)]


# o_trend = st_utils.run_beast(idx_df)
logits = st_utils.get_logits(base_res)

answer_fn = st_utils.ans_key_map.get(task, lambda x: 'N/A')
answer = answer_fn(args_json['row'])


# ========================================================================

# Highlighted text
#    - https://docs.streamlit.io/develop/api-reference/custom-components/st.components.v1.html
col1, col2 = st.columns(2)
if col1.checkbox('Highlighted Text', value=1):
    cpd_probs = 1 - o_trend['cpOccPr']

    print_newln = col2.checkbox('Print Newline Character  `\\n`', value=0)

    cols = st.columns(3)
    cols[0].html( text_viz.text_base_html(base_res[0]['prompt_raw'], print_newln=True, width='100%', bg_color='#cfffd3'))
    cols[0].markdown('Prompt')
    cols[0].html('<span style="color: black; background-color: #6ff283; font-size: 22px">&nbsp; &nbsp; &nbsp; Ground Truth Answer: &nbsp;' + answer + '&nbsp; &nbsp; &nbsp;</span>')
    cols[1].html( text_viz.text_colors_html(base_tokens, logits, print_newln=print_newln, width='100%', add_start_tok=None))
    cols[1].markdown('Token Logit Probs')
    cols[2].html( text_viz.text_colors_html(base_tokens, cpd_probs, print_newln=print_newln, width='100%'))
    cols[2].markdown('Change Point Detection Probs   $\quad  p(\\tau = t)$')


# Stacked line plot    (+ baseline bars ????)
if st.checkbox('Outcome Time Series  $\quad  o_t$', value=1):
    fig_stack = st_plot.plotly_stacked(idx_df, ['<START>'] + base_tokens, outcome_colors, outcomes_set)
    st.plotly_chart(fig_stack)


# Forking tokens sankey
# 	- update when clicking on time series (any of them)
if st.checkbox('Outcome Parallel Sets Plots  $\quad  o_{t, w}$', value=1):
    cols = st.columns([.2, .8], gap='large')

    t_min, t_max = min(idx_tok_df['idx']), max(idx_tok_df['idx'])
    t_select = cols[1].slider('Token index $t$', min_value=t_min, max_value=t_max - 4,
                              value=32 if (task =='HotpotQA' and qa == '8076') else 0)   # set initial value to match figure in paper

    plot_n_idxs = cols[0].number_input('Number of tokens to show', min_value=1, value=4)

    fig_parcat = st_plot.plotly_parallel_categories(idx_tok_df, ['<START>'] + base_tokens, outcome_colors,
                                                    plot_idx=t_select, plot_n_idxs=plot_n_idxs)

    st.plotly_chart(fig_parcat)



# Semantic drift
if st.checkbox('Semantic Drift  $\quad  y_t = d(o_0, o_t)$', value=1):
    cols = st.columns(3)
    dist = cols[0].selectbox(
        'Distance Function',
        ['l1', 'l2', 'cos', 'kl'],
        index=1,
    )
    fig_drift = st_plot.plotly_semantic_drift(idx_df, ['<START>'] + base_tokens, dist_fn=f'd_{dist}')
    st.plotly_chart(fig_drift)

# Change point detection      p(tau = t)
if st.checkbox('Change Point Detection  $\quad  p(\\tau = t | y)$', value=1):
    fig_cpd = st_plot.plotly_cpd_line(o_trend, ['<START>'] + base_tokens)
    st.plotly_chart(fig_cpd)

# Forking survival time series
if st.checkbox('Survival Analysis  $\quad  S(t)$', value=1):
    cols = st.columns(3)
    dist_ = cols[0].selectbox(
        'Distance Function ',
        ['l1', 'l2', 'cos', 'kl'],
        index=1,
    )
    fig_h, fig_s = st_plot.survival_lines(idx_tok_df, base_tokens,
                                          dist_fn=f'd_{dist_}', d_thresholds=[0, .1, .2, .3, .5, .7])
    st.plotly_chart(fig_h)
    st.plotly_chart(fig_s)


# Baseline -- token log probs
if st.checkbox('Baseline: Token Logit Probs', value=1):
    fig_lp = st_plot.plotly_logprob_line(logits, base_tokens)
    st.plotly_chart(fig_lp)




