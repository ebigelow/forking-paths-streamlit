
### Data format

Path format: `data/{task}/{question_id}`

Files in each sub-directory:
- `idx_df.csv` : Outcome distribution by token index  $o_t$
- `idx_tok_df.csv` :  Outcome distribution by token index and token value  $o_{t, w}$
- `ans_df.csv` : Parsed outcome data for all completion samples, along with probabilities for the completion $p(x_{>t}^{(s)} | x_{<t}^*, x_t = w)$ and each alternate token $p(x_t = w | x_{<t}^*)$. This isn't used in the dashboard currently, but was aggregated to compute `idx_df` and `idx_tok_df`.

- `base_res.pk` : Base path completion object, including text output $x^*$ and top-k token logits $p(x_t = w | x_{<t}^*)$
- `trend.pk` :  Bayesian change point detection model results; format is [RBeast](https://github.com/zhaokg/Rbeast) results dict

- `args.json` : High-level parameters used when collecting data from GPT



