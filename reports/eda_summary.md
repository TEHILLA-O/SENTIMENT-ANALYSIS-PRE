# EDA summary

## 2. What does the dataset contain?

- 50,000 labelled reviews, one plain-text file each
- Two usable columns: `text` (input X) and `label` (target y: 0=neg, 1=pos)
- Exact duplicate review texts: 418

## 3. Is the dataset balanced?

**train** (25,000 reviews)

- pos: 12,500 (50.0%) `####################`
- neg: 12,500 (50.0%) `####################`

**test** (25,000 reviews)

- pos: 12,500 (50.0%) `####################`
- neg: 12,500 (50.0%) `####################`

## 4. How long are the reviews?

- Mean: 228.0 words
- Median: 171 words
- Min / Max: 4 / 2459 words
- 90th percentile: 445 words
- Truncated at SEQUENCE_LENGTH=250: 29.0% of reviews
- Mean length (pos): 229.8 words
- Mean length (neg): 226.2 words

## 5. Which tokens appear frequently, and which discriminate?

Most frequent non-stopword tokens (train):

`like (19,682)`, `good (14,710)`, `time (12,130)`, `really (11,693)`, `story (11,675)`, `well (9,586)`, `bad (9,058)`, `people (9,040)`, `great (8,953)`, `first (8,889)`, `dont (8,482)`, `movies (7,924)`, `films (7,845)`, `characters (7,404)`, `think (7,258)`

Top 20 positive-associated tokens (min 100 occurrences):

| word | pos | neg | log2(pos/neg) |
| --- | --- | --- | --- |
| edie | 101 | 0 | +7.63 |
| paulie | 100 | 1 | +6.03 |
| felix | 102 | 4 | +4.47 |
| 710 | 197 | 9 | +4.34 |
| 810 | 211 | 11 | +4.16 |
| matthau | 128 | 7 | +4.06 |
| 910 | 146 | 9 | +3.91 |
| victoria | 182 | 12 | +3.83 |
| flawless | 115 | 8 | +3.73 |
| mildred | 100 | 7 | +3.71 |
| 1010 | 246 | 21 | +3.48 |
| gandhi | 95 | 8 | +3.45 |
| astaire | 101 | 9 | +3.38 |
| superbly | 112 | 11 | +3.25 |
| perfection | 124 | 13 | +3.17 |
| wonderfully | 283 | 35 | +2.96 |
| captures | 192 | 24 | +2.94 |
| brosnan | 97 | 12 | +2.93 |
| voight | 89 | 11 | +2.92 |
| peters | 88 | 12 | +2.79 |

Top 20 negative-associated tokens:

| word | pos | neg | log2(pos/neg) |
| --- | --- | --- | --- |
| boll | 1 | 128 | -6.46 |
| uwe | 1 | 100 | -6.10 |
| 410 | 2 | 166 | -6.10 |
| 210 | 2 | 121 | -5.64 |
| seagal | 3 | 133 | -5.29 |
| mst3k | 4 | 133 | -4.93 |
| 310 | 5 | 162 | -4.92 |
| unwatchable | 4 | 103 | -4.56 |
| incoherent | 7 | 129 | -4.15 |
| unfunny | 17 | 252 | -3.89 |
| waste | 96 | 1,355 | -3.85 |
| blah | 13 | 169 | -3.69 |
| pointless | 40 | 457 | -3.54 |
| horrid | 9 | 105 | -3.51 |
| drivel | 10 | 112 | -3.46 |
| atrocious | 16 | 176 | -3.46 |
| redeeming | 28 | 297 | -3.42 |
| lousy | 19 | 194 | -3.36 |
| worst | 249 | 2,450 | -3.33 |
| prom | 10 | 102 | -3.33 |

## Raw examples

**neg** (204 words): This gawd-awful piece of tripe is all over the place. The script is bad, the plot is bad, the acting is bad. There are a couple of decent actors in it (Charles Durning, eg.), but the director got nothing out of them. The plot line has Santa, feeling dejected and thinking no one needs him any more, t...

**neg** (241 words): Not even Bob Hope, escorted by a raft of fine character actors, can save this poorly written attempt at wartime comedy, as his patented timing has little which which to work. The plot involves a Hollywood film star named Don Bolton (Hope), and his attempt to evade military service at the beginning o...
