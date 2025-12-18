### Setup

* A dialogue has utterances $u=1,\dots,U$.
* Each utterance has a **final ground-truth string** $G_u$.
* During stepwise revealing, at **step** $s=0,1,\dots,S_u-1$ (0 = earliest), the model emits a **full prediction** $P_{u,s}$.

### Similarities at each step

1. **Lexical similarity** (normalized Levenshtein):

$$
\text{lex}(P_{u,s},G_u)\;=\;1-\frac{d_\text{lev}(P_{u,s},G_u)}{\max\{|P_{u,s}|,\ |G_u|\}}
$$

where $d_\text{lev}$ is character-level edit distance and $|\cdot|$ = string length in characters.
Range: $[0,1]$.

2. **Semantic similarity** (token Jaccard):

$$
\text{sem}(P_{u,s},G_u)\;=\;\frac{|\,T(P_{u,s})\cap T(G_u)\,|}{|\,T(P_{u,s})\cup T(G_u)\,|}
$$

where $T(\cdot)$ is the set of whitespace-split tokens.
Range: $[0,1]$.

### Earliness weight

Earlier correct predictions are better. For step $s$ (0-indexed):

$$
\text{earliness}(s)\;=\;\frac{1}{s+1}
$$

So the first prediction gets weight $1$, the second $1/2$, the third $1/3$, etc.

### Per-step utility

Blend lexical and semantic similarity with weight $w\in[0,1]$ (default $w=0.5$), then discount by earliness:

$$
U_{u,s}\;=\;\Big(w\cdot \text{lex}(P_{u,s},G_u) + (1-w)\cdot \text{sem}(P_{u,s},G_u)\Big)\cdot \frac{1}{s+1}
$$

### Per-utterance score (best-early)

Take the **best** (highest) step utility within the utterance:

$$
U_u\;=\;\max_{0\le s<S_u} \;U_{u,s}
$$

This captures “the earliest step at which the model is most correct,” rewarding both accuracy and being early.

### Dialogue score

Average the utterance scores:

$$
U_{\text{dialogue}}\;=\;\frac{1}{U}\sum_{u=1}^{U} U_u
$$

---

### Notes & rationale

* Using $\max_s U_{u,s}$ (instead of the last step) emphasizes **earliest strong matches**.
* Normalized Levenshtein captures fine-grained character overlap; Jaccard captures token-level semantic overlap.
* $w$ trades off sensitivity: higher $w$ favors character-exactness; lower $w$ favors token overlap.
* The harmonic-like earliness $1/(s+1)$ gives a principled, monotone decay with each additional revealed token.
