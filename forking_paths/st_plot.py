

import numpy as np
import pandas as pd    
import seaborn as sns

import plotly.graph_objs as go
import plotly.offline as offline

from forking_paths.analysis_utils import get_survival_df
from forking_paths.st_utils import semantic_drift


### TODO --- can this be removed?
OTHER_TOK = '$\\it{Other}$'


def plotly_stacked(idx_df, base_tokens, colors, outcomes_set, tick_text=True):
    
    layout =  go.Layout(
        title = None,
        font = {'size': 10},
        
        # xaxis_range=(0,1), 
        margin=dict(l=20, r=20, t=20, b=80),
        yaxis_range=(0, 1),
        # plot_bgcolor='white',
        showlegend=True,
        
        hovermode='x',  ###  'x',

        yaxis=dict(
            title='Outcome %',
            showgrid=False
        ),

        xaxis = dict(
            tickmode = 'array',
            tickvals = list(range(min(idx_df['idx']), max(idx_df['idx']))),
            ticktext = base_tokens if tick_text else None,
            tickangle = -90 if tick_text else None,

            # showline=True,
            showgrid=True,

            gridwidth=.5, 
            gridcolor='rgb(.8, .8, .8)',
        ),

        legend=dict(xanchor="left", itemwidth=30)     # orientation="h",yanchor="bottom",y=1.02,  x=0,font=dict(size=10),
    )

    fig = go.Figure(layout=layout) 

    fig.add_traces([
        go.Scatter( 
            name = '*Other' if outcome == '$\\it{Other}$' else outcome,   #OTHER_TOK else outcome, 
            x = idx_df[idx_df['ans'] == outcome]['idx'], 
            y = idx_df[idx_df['ans'] == outcome]['weighted'], 
            stackgroup='one',
            fillcolor=f'rgba{colors[oi] + (.7,)}',
            line={'color': f'rgb{colors[oi]}'}
        )
        for oi, outcome in enumerate(outcomes_set)
    ])
    
    return fig



def plotly_parallel_categories(idx_tok_df, base_tokens, colors, plot_idx=0, plot_n_idxs=4):
    # base_tokens = base_res[0]['choice']['logprobs']['tokens']

    idx_tok_df_ = idx_tok_df.copy()

    trace_kws = []
    
    # Draw n sankey plots in a sequence
    for i in range(plot_idx, plot_idx + plot_n_idxs):
        dfi = idx_tok_df_[idx_tok_df_['idx'] == i].copy()
        
        dfi['tok_codes'] = dfi['ans'].cat.codes.tolist()
        dfi = dfi.sort_values(by=['tok', 'tok_codes'], ascending=False).reset_index(drop=True)    # sort tokens so highest tok_p token is first
    
        outcomes    = dfi['ans'].tolist()        # 'outcome'
        next_tokens = dfi['tok'].tolist()        # 'next_token'
        next_probs  = dfi['weighted'].tolist()   # 'weighted'
        
        prev_token = base_tokens[i-1] if i > 0 else '<START>'
        # for i, (last_token, next_tokens, next_probs) in enumerate(zip(last_tokens, next_tokens_s, next_probs_s)):
        
        out_idxs = dfi['ans'].cat.codes.tolist()
        
        j = i - plot_idx
        pw = 1 / plot_n_idxs
        
        # One trace per sankey plot
        trace_kws.append(
            dict(
                # https://plotly.com/python/builtin-colorscales/
                line={'colorscale': [f'rgb{c}' for c in colors],        # sunset   'color': outcome_colors * n_toks,
                      'color': out_idxs,
                      'shape': 'hspline',
                     },
                
                domain={'x': [pw*j, pw*(j+1)]},
                dimensions=[
                    {'label': 'Current Token',                      ###prev_token,
                     'values': [prev_token] * len(next_tokens)},
                     # 'values': outcomes if j == 0 else   [prev_token] * len(outcomes)},
                    {'label': 'Next Token',
                     'values': next_tokens}
                     # 'values': repeat_ls(next_tokens, n_out)}
                ],
                counts=next_probs
            )
        )

    layout = go.Layout(
        title = None,
        font = {'size': 10},
        
        xaxis = go.layout.XAxis(
            title = None,
            showticklabels=False),
        yaxis = go.layout.YAxis(
            title=None,
            showticklabels=False
        ),

        xaxis_range=(0,1), 
        yaxis_range=(0,1),
        margin=dict(l=20, r=20, t=20, b=20),
        # plot_bgcolor='white',
        showlegend=False,
    )
    
    fig = go.Figure(data=[go.Parcats(**trace_kw) for trace_kw in trace_kws],
                    layout=layout)
    return fig




######## TODO: compress these with a generic "plot_single_line" function with y and color as args


def plotly_logprob_line(logit_probs, base_tokens):

    layout =  go.Layout(
        title = None,
        font = {'size': 10},
        
        # xaxis_range=(0,1), 
        margin=dict(l=20, r=20, t=60, b=40),
        yaxis_range=(0, 1),
        # plot_bgcolor='white',
        showlegend=True,
        hovermode='x',

        yaxis=dict(
            title='Token Logit Prob.',
            showgrid=False,
            # gridwidth=.5, 
            # gridcolor='rgb(.8, .8, .8)',
        ),

        xaxis = dict(
            tickmode = 'array',
            ticktext = base_tokens,

            showgrid=True,
            gridwidth=.5, 
            gridcolor='rgba(.8, .8, .8, .3)',
        )
    )

    fig = go.Figure(layout=layout) 
    fig.add_trace(
        go.Scatter( 
            y = logit_probs, 
            # stackgroup='one',

            line={'color': 'orange'}
        )
    )

    return fig


def plotly_cpd_line(o_trend, base_tokens):
    cpd_probs = o_trend['cpOccPr']

    layout =  go.Layout(
        title = None,
        font = {'size': 10},
        
        # xaxis_range=(0,1), 
        margin=dict(l=20, r=20, t=60, b=40),
        yaxis_range=(0, 1),
        # plot_bgcolor='white',
        showlegend=True,
        hovermode='x',

        yaxis=dict(
            title='Change Point Prob.',
            showgrid=False,
            # gridwidth=.5, 
            # gridcolor='rgb(.8, .8, .8)',
        ),

        xaxis = dict(
            tickmode = 'array',
            ticktext = base_tokens,

            showgrid=True,
            gridwidth=.5, 
            gridcolor='rgba(.8, .8, .8, .3)',
        )
    )

    fig = go.Figure(layout=layout) 
    fig.add_trace(
        go.Scatter( 
            x = list(range(len(base_tokens))), 
            y = cpd_probs, 
            # stackgroup='one',

            line={'color': 'blue'}
        )
    )

    return fig



def plotly_semantic_drift(idx_df, base_tokens, dist_fn='d_l1'):

    dist_df = semantic_drift(idx_df)
    dists = dist_df[dist_fn]
    i_min, i_max = min(idx_df['idx']), max(idx_df['idx'])

    layout =  go.Layout(
        title = None,
        font = {'size': 10},
        
        margin=dict(l=20, r=20, t=60, b=40),
        showlegend=True,
        hovermode='x',

        yaxis=dict(
            title=f'Semantic Drift ({dist_fn[2:]})',
            showgrid=False,
            # gridwidth=.5, 
            # gridcolor='rgb(.8, .8, .8)',
        ),

        xaxis = dict(
            tickmode = 'array',
            tickvals = list(range(i_min, i_max)),
            ticktext = base_tokens,
            tickangle = -90,

            showgrid=True,
            gridwidth=.5, 
            gridcolor='rgba(.8, .8, .8, .3)',
        )
    )

    fig = go.Figure(layout=layout) 
    fig.add_trace(
        go.Scatter( 
            x = list(range(i_min, i_max)), 
            y = dists, 
            line={'color': 'purple'}
        )
    )

    return fig




def plotly_generic_line(ys, base_tokens, y_title, color='purple', ynorm=False):
    xs = list(range(len(base_tokens)))
    layout =  go.Layout(
        title = None,
        font = {'size': 10},
        
        margin=dict(l=20, r=20, t=60, b=40),
        showlegend=True,
        hovermode='x',

        yaxis_range=(0, 1) if ynorm else None,
        yaxis=dict(
            title=y_title,
            showgrid=False,
        ),

        xaxis = dict(
            tickmode='array',
            tickvals=xs,
            ticktext=base_tokens,

            showgrid=True,
            gridwidth=.5, 
            gridcolor='rgba(.8, .8, .8, .3)',
        )
    )

    fig = go.Figure(layout=layout) 
    fig.add_trace(
        go.Scatter( 
            x=xs, y=ys, 
            line={'color': color}
        )
    )
    return fig









def get_generic_line_layout(y_title=None, yrange=(0, 1)):
    # xs = list(range(len(base_tokens)))
    layout =  go.Layout(
        title = None,
        font = {'size': 10},
        
        margin=dict(l=20, r=20, t=60, b=40),
        showlegend=True,
        hovermode='x',

        yaxis_range=yrange,
        yaxis=dict(
            title=y_title,
            showgrid=False,
        ),

        xaxis = dict(
            title='Token Index',

            tickmode='array',
            # tickvals=xs,
            # ticktext=base_tokens,

            showgrid=True,
            gridwidth=.5, 
            gridcolor='rgba(.8, .8, .8, .3)',
        ),

        legend=dict(
            title='Dist Thresh.'
        )
    )
    return layout


def rgb_1_to_255(rgb):
    return ','.join([str(int(np.round(c * 255))) for c in rgb])




def survival_lines(idx_tok_df, base_tokens, dist_fn='d_l1', d_thresholds=[0, .1, .2, .3, .5, .7]):

    survive_df = get_survival_df(idx_tok_df, base_tokens, dist_fn=dist_fn, d_thresholds=d_thresholds)
    palette = sns.color_palette('viridis', n_colors=len(d_thresholds))

    layout_h =  get_generic_line_layout(y_title='Hazard Prob.', yrange=(0, 1))
    fig_h = go.Figure(layout=layout_h) 

    layout_s =  get_generic_line_layout(y_title='Survival Prob.', yrange=(0, 1))
    fig_s = go.Figure(layout=layout_s) 
    

    for d_idx, d_thresh in enumerate(d_thresholds):
        df_ = survive_df[survive_df['threshold'] == d_thresh]
        color = palette[d_idx]
        color_rgb = f'rgb({rgb_1_to_255(color)})'

        # print(df_['idx'])
        # import ipdb; ipdb.set_trace()

        fig_h.add_trace(
            go.Scatter( 
                x=df_['idx'], y=df_['hazard'], 
                line={'color': color_rgb},
                name=d_thresh,
            )
        )
        fig_s.add_trace(
            go.Scatter( 
                x=df_['idx'], y=df_['survival'], 
                line={'color': color_rgb},
                name=d_thresh,
                # [f'rgb({rgb_1_to_255(palette[i])})' for i in pd.Categorical(survive_df['threshold']).codes]
            )
        )

    return fig_h, fig_s

