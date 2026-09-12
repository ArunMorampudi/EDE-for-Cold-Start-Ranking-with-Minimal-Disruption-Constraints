"""Apply reviewer-driven edits to the original manuscript, retaining its structure."""
from pathlib import Path
from copy import deepcopy
from difflib import SequenceMatcher
import json
import re
import shutil
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / 'revision1'
OUT = ROOT / 'submitted_peerj_docs'
WORK = REV / 'work'
WORK.mkdir(exist_ok=True)
source = WORK / 'expanded_manuscript.docx'
if not source.exists():
    shutil.copy2(OUT / 'cs-134580-v0.4.docx', source)
expanded = Document(source)
ep = expanded.paragraphs
d = Document(REV / 'original_submitted/cs-134580-v0.4.docx')
op = list(d.paragraphs)
original_text = [p.text for p in op]
edits = []


def patch(i, *pairs, append=''):
    """Change specified phrases only; all surrounding original prose is retained."""
    text = original_text[i]
    for old, new in pairs:
        assert old in text, (i, old)
        text = text.replace(old, new, 1)
    replace(i, text + append)

def replace(i, text):
    p = op[i]
    old = p.text
    if old == text:
        return
    rpr = deepcopy(p.runs[0]._r.rPr) if p.runs and p.runs[0]._r.rPr is not None else None
    for node in list(p._p):
        if node.tag != qn('w:pPr'):
            p._p.remove(node)
    if i in (19,20,21,22):
        lead, rest = text.split('. ',1)
        p.add_run(lead+'. ').bold=True
        p.add_run(rest).bold=False
    elif i>=105 and re.search(r'\b(?:19|20)\d{2}\.',text):
        cut=re.search(r'\b(?:19|20)\d{2}\.',text).start()
        p.add_run(text[:cut]).bold=True
        p.add_run(text[cut:]).bold=False
    else:
        run = p.add_run(text)
        if rpr is not None:
            run._r.insert(0, rpr)
    edits.append({'original_paragraph': i, 'original': old, 'revised': text})

def insert_before(i, text, heading=False):
    node = OxmlElement('w:p')
    op[i]._p.addprevious(node)
    p = Paragraph(node, op[i]._parent)
    p.style = 'Normal'
    if heading:
        if op[92]._p.pPr is not None:
            p._p.insert(0,deepcopy(op[92]._p.pPr))
        run=p.add_run(text)
        if op[92].runs[0]._r.rPr is not None:
            run._r.insert(0,deepcopy(op[92].runs[0]._r.rPr))
    elif i>=105 and re.search(r'\b(?:19|20)\d{2}\.',text):
        cut=re.search(r'\b(?:19|20)\d{2}\.',text).start()
        p.add_run(text[:cut]).bold=True
        p.add_run(text[cut:]).bold=False
    else:
        p.add_run(text)
    return p

def remove(i):
    edits.append({'original_paragraph': i, 'original': op[i].text, 'revised': ''})
    op[i]._p.getparent().remove(op[i]._p)

def equation(i, expanded_index):
    original_node = op[i]._p
    node = deepcopy(ep[expanded_index]._p)
    original_node.addprevious(node)
    original_node.getparent().remove(original_node)

def add_equation_before(i, expanded_index):
    op[i]._p.addprevious(deepcopy(ep[expanded_index]._p))

patch(19)
patch(20,
    ('user-click uncertainty (measured as click entropy)', 'click entropy'),
    ('To preserve user experience,', 'To preserve the original top results,'),
    ('This design makes EDE a minimal-disruption control layer that improves cold-start exposure without materially altering high-confidence top results. ', ''),
    ('and subsequently conduct a conservative offline replay analysis', 'with independent validation and test seeds and subsequently conduct an offline replay analysis'))
patch(21,
    ('significantly accelerates new-document discovery while maintaining ranking quality', 'improves new-document discovery in some settings with small average ranking-quality losses'),
    ('At moderate difficulty,', 'At penalty 0.15,'),
    ('approximately 0.05 to 0.58', '0.087 to 0.186'),
    ('a negligible Normalized Discounted Cumulative Gain (NDCG) drop of ≈0.005, consistently outperforming random tail injection and ε-greedy baselines', 'a paired Normalized Discounted Cumulative Gain (NDCG@10) difference of −0.0026 (95% CI −0.0041 to −0.0011); novelty-only was competitive or stronger'),
    ('the conservative replay analysis supports EDE’s deployment-oriented behavior as a minimal-disruption policy', 'the replay analysis measures displacement on logged candidates'),
    ('Under click-preservation filtering (≈97% session acceptance), EDE functioned as a stable tail-swap, exhibiting a mean position change of 0.661.', 'Across 372,282 pages, exact-list acceptance was 93.611% and mean absolute document shift was 0.014919.'),
    ('Furthermore, these accepted sessions showed small positive engagement deltas (ΔCTR@10 ≈ +0.00285; Δclick-NDCG@10 ≈ +0.00102), indicating that exploration did not negatively affect user experience.', 'Set and click preservation were 100% by construction on the ten logged candidates; this does not establish causal engagement effects or deployment safety.'))
patch(22,
    ('practical constrained', 'constrained'),
    ('meaningfully accelerating', 'improving'),
    ('while largely preserving user experience', 'while preserving the original top results'),
    ('More broadly, the results suggest that minimal-disruption post-ranking control layers can offer a practical and low-risk way to address cold-start exposure in existing ranking systems.', 'The results do not establish a consistent entropy advantage over novelty alone or guarantee overall ranking quality; replay supplies no real cold-start discovery evidence.'))

# Keep the original motivation and prior-work paragraphs; remove only repeated or unsupported framing.
patch(28,
    ('In contrast,', 'Here,'),
    ('This design prioritizes deployment viability: EDE makes only small modifications to the provided list, primarily through conservative tail reranking based on real-time click signals. ', ''),
    ('two complementary signals', 'two signals'),
    ('which captures user-side uncertainty', 'which captures variability in recent, online-available click outcomes'),
    ('limited recent exposure', 'limited exposure'),
    ('uncertain but potentially relevant new items', 'low-exposure items'),
    ('a short grace period', 'smooth exposure-dependent entropy weighting'),
    ('explicit safety constraints', 'explicit structural constraints'),
    ('Together, these mechanisms ensure that exploration occurs under explicit minimal-disruption constraints.', 'These constraints preserve the prefix but do not guarantee an NDCG bound.'))
patch(30,
    ('The key novelty of EDE is therefore not merely its entropy-based exploration score, but its formulation as a low-risk reranking control layer that can be attached to an existing ranking system to improve cold-start exposure without materially disrupting user-facing ranking quality.', 'EDE evaluates an empirical combination of click entropy, novelty, prefix preservation and candidate gating.'),
    ('Unlike prior online learning-to-rank methods that primarily optimize learning efficiency through continued exploration, EDE', 'EDE'),
    ('carefully bounded intervention', 'structurally constrained intervention'),
    ('safer cold-start exploration', 'constrained cold-start exploration'),
    append=' Safe exploration of previously unranked items (Shiino et al., 2023) and exposure-based risk minimization (Gupta et al., 2023) overlap with this goal; EDE does not claim their guarantees.')
patch(33, ('a deployment-oriented reranking layer', 'a reranking layer'))
patch(34, ('an uncertainty- and novelty-aware', 'an entropy- and novelty-aware'))
patch(35,
    ('a carefully calibrated synthetic benchmark', 'a synthetic benchmark'),
    ('show that EDE substantially improves new-document discovery while remaining within a small NDCG degradation budget', 'evaluate new-document discovery and NDCG with independent validation and test seeds'))
patch(36,
    ('a conservative offline replay analysis', 'an offline replay analysis'),
    ('showing that under strict click-preservation filtering, EDE behaves as a stable tail-reranking policy that rarely disrupts clicked results and exhibits only small ranking changes', 'measuring displacement on logged candidates, without ground-truth cold-start labels or causal engagement estimates'))
remove(38)
replace(43, op[43].text.replace('These “click-model-based” bandits enhance rankings through online exploration; however, they frequently lack safety constraints and can degrade user experience by prioritizing less-relevant items during exploration.', 'These “click-model-based” bandits enhance rankings through online exploration, but guarantees depend on their feedback and click-model assumptions.').replace('thereby preserving the rank near the initial base list', 'thereby constraining local exploration around the initial base list').replace('click entropy and novelty, rather than pairwise upper Confidence Bound (UCB)', 'click entropy and novelty, rather than pairwise upper confidence bounds (UCB)').replace('they are not replicas', 'they are not replicas'))
insert_before(45, 'Shiino et al. (2023) connect cold-start and safe exploration by selecting previously unranked items with KL-UCB; BubbleRank learns pairwise preferences, whereas EDE uses aggregate document counters and guarantees only prefix preservation. Exposure constraints target allocation (Singh & Joachims, 2018), and diversification targets redundancy or aspects; EDE has neither a provider-group target nor a diversity objective. Item-centric exploration selects audiences for new items (Wang et al., 2025), whereas EDE reranks for a selected query. Stateful cascading reinforcement learning (Du et al., 2024) differs from our prescribed cascade click generator.')
insert_before(45, 'Counterfactual learning-to-rank corrects biased feedback using logging propensities (Swaminathan & Joachims, 2015; Agarwal et al., 2019); intervention harvesting estimates examination effects from ranking variation (Fang et al., 2019). Gupta et al. (2023) regularize exposure mismatch to reduce risk. Our unweighted replay lacks propensities and a reward model, so cannot provide these guarantees or doubly robust estimates.')
patch(45,
    ('Our work differs by focusing on', 'Our work focuses on'),
    ('user-side uncertainty (measured by click entropy)', 'click entropy'),
    ('explicit safety constraints', 'explicit structural constraints'),
    ('Therefore, the main contribution is not only the entropy-based exploration signal, but also a practical and low-risk way to improve cold-start exposure under a controlled ranking-quality budget. This framing supports clear claims about improved cold-start discovery with limited disruption to the original ranking.', 'We test discovery in simulation and displacement in replay. Click entropy can reflect mixed intent, position bias or noise; it is not an estimate of epistemic relevance uncertainty.'))
insert_before(47, 'We describe the simulator, policy and evaluation below. Simulation supplies the direct cold-start evidence; replay measures displacement on logged candidates.')
patch(48,
    ('true continuous relevance score', 'continuous relevance score'),
    ('scaled to [0,1].', 'scaled to [0,1] as r=(1+cos(q,d))/2.'),
    ('a relevance threshold', 'a relevance threshold of 0.5'),
    ('exactly  relevant items', 'exactly n_rel=10 relevant items'),
    append=' Each run uses 100 queries and 1,000 documents sampled independently in 16 dimensions from a standard normal distribution and normalized by their Euclidean norm plus 10⁻⁸. Removing the lowest relevant items or adding the highest nonrelevant items makes the adjusted labels effectively top-ten relevance. A fixed-threshold condition without this adjustment tests naturally varying relevance counts (Section 3.4).')
patch(50,
    ('from  ', 'from t=0'),
    ('at time  ', 'at time T₀=1,000'),
    ('last  docs', 'last n_new=200 docs'),
    ('at the start', 'at arrival'),
    ('a noisy relevance score:', 'a noisy relevance score and a penalty p for new documents (Equation (1)).'),
    append=' The horizon is T=2,000 query timesteps, with one query and ten displayed results per step. All available documents are retrieved; the legacy k=100 argument does not subsample them.')
# Preserve the original equation object; add only the missing penalty and number.
for math_text in op[51]._p.iter(qn('m:t')):
    if math_text.text == ', ':
        math_text.text = '−p 1[d is new], '
        break
else:
    raise AssertionError('Original base-score equation delimiter not found')
op[51].add_run('   (1)')
patch(52,
    ('Where noise   is fixed per query to control variance.', 'Noise ε is fixed per query to control variance (standard deviation σ=0.15; seed+query_index). Here true_relevance(d) is r(q,d) for the current query.'),
    ('This ranker utilizes an off-the-shelf Learning to Rank (LTR) model, such as gradient boosted decision trees (Chen & Guestrin, 2016), that under-ranks new items and has no knowledge of clicks.', 'This is a stationary noisy relevance model, not a trained LTR model. The same penalized base score is used for the baseline and EDE tail.'))
patch(54,
    ('standard click models', 'click models'),
    ('position k', 'one-based position k'),
    ('true relevance r is', 'relevance r is given by Equation (2), with a=3 and b=0.35:'))
op[55].add_run('   (2)')
patch(56,
    ('where a controls steepness and b captures position bias.', 'Here a controls steepness and b captures position bias; σ(x)=1/(1+exp(−x)) is the logistic function.'),
    ('a cascade model (Craswell et al., 2008)', 'a cascade model'),
    ('click at most one item', 'stop after a click with probability 0.8, allowing multiple clicks'),
    ('(random clicks with probability noise)', '(with probability 0.1, replacing the click vector with one uniformly chosen clicked position)'),
    ('These variants help verify that EDE’s gains are not driven by a single click-model assumption.', 'Each displayed position receives a Bernoulli trial; unexamined cascade positions are recorded as displayed nonclicks. This logistic model is not a factorized examination estimator, and its entropy is not unbiased relevance uncertainty.'))
insert_before(57, 'Query, click, cascade-stop and policy choices use independent seed-derived streams, sharing query and click trial schedules across policies. Counters update after serving; delayed feedback postpones both counters by 50 timesteps but records first clicks at their event time.')
patch(58,
    ('impressions  and clicks ,', 'impressions I(d) and clicks C(d), aggregated across queries,'),
    ('This design makes the method suitable for settings where cold-start exposure must be improved without materially altering high-confidence top results. ', ''),
    ('Laplace smoothing', 'symmetric pseudocount smoothing'),
    ('parameters  = 0.5', 'parameters α = 0.5'),
    ('and  = 0', 'and α = 0'),
    ('“unsmoothed”):', '“unsmoothed”). For α=0 and I=0, we define the probability in Equation (3) as 0.5:'))
op[60].add_run('   (3)')
patch(62, ('From this we compute', 'From Equation (3) we compute'),
    ('(Shannon, 1948)  , normalized by  to lie in .', '(Shannon, 1948), normalized with base-two logarithms to lie in [0,1] (Equation (4)).'))
add_equation_before(63,35)
patch(64,
    ('constant , novelty is  ,', 'constant τ=50 impressions, novelty is N(d)=exp(−I(d)/τ),'),
    ('when  (', 'when I(d)≈0 ('),
    ('Refined entropy is  which', 'Refined entropy R(d) in Equation (5)'),
    ('(so we don’t explore purely at random before some exposures)', '(using the complementary weight 1−N(d))'))
add_equation_before(65,37)
patch(65, ('weight :   is an exploration bonus in  where  (e.g.\\ 0.65)', 'weight η: E(d) in Equation (6) is an exploration bonus in [0,1], where η=0.65'))
add_equation_before(66,38)
insert_before(67, 'There is no discrete grace period: the refinement weight 1−exp(−I/50) is zero at zero impressions, approximately 0.095 at five and 0.632 at 50. Novelty applies to every low-exposure document, not only new arrivals.')
patch(67,
    ('To rank results, we first form a candidate set of size top_L by taking the highest-scoring documents under an adjusted score (which includes a small discount for new docs so they can pass the gate). Then we fix the top-m items from the base ranker as a safe prefix (to protect the user experience).', 'To rank results, we first fix the top-m items from the original base ranker as a safe prefix. We fill a candidate set of size top_L=50 with the remaining highest-scoring documents under gate score g(d)=s₀(d)+ηN(d). Ties follow candidate order.'),
    ('remaining positions  we re-rank by the score :', 'remaining 10−m positions we re-rank by Equations (7) and (8):'))
equation(68,41)
add_equation_before(69,42)
patch(69,
    ('Where  is a min-max normalized base score, ensuring the initial ranking signals operate on a standardized computational scale (Han et al., 2011).', 'Here s_norm is a rank-normalized base score: rank_pool is one-based within the remaining pool of M=top_L−m documents (or fewer if unavailable); for M=1, s_norm=1.'),
    ('The scalar  tunes', 'The scalar λ tunes'),
    ('Critically, if   the base score is zero, EDE does not reorder the ranking (the base ranking is returned, except possibly due to new-doc gating), ensuring the validity of comparisons.', 'If λ=0, EDE bypasses gating and returns the base ranking exactly.'),
    ('and  is set so that  decays appreciably after a few impressions', 'and τ=50 impressions'),
    ('We also implemented a “hard-mode” where new-doc penalties apply even before gating, to simulate extremely adverse conditions.', '“Hard mode” denotes p>0 applied before gating and retained in the tail; p=0 removes this penalty.'))
insert_before(70, 'All results were regenerated after correcting the submitted code’s prefix selection within the gate and its unpenalized tail scores.')
patch(71, ('(Jarvelin and Kekalainen, 2002).', '(Järvelin & Kekäläinen, 2002). The ideal top ten uses binary relevance across the full corpus and may include unavailable items before arrival; post-arrival NDCG is also reported.'))
patch(72,
    ('relevant new documents', 'new documents relevant to at least one query'),
    ('within the session horizon while being eligible for ranking', 'before T'),
    ('the number of simulated timesteps until the first click on a relevant new document', 'the number of query timesteps after arrival until the first click, with unclicked eligible documents assigned T−T₀=1,000'),
    ('across sessions (a measure of exposure diversity)', 'during the run'),
    ('were averaged', 'are averaged'),
    ('20 simulation seeds', '20 test seeds'),
    ('mean ± standard deviation', 'mean ± sample standard deviation'),
    append=' TTF includes all eligible documents. Query-matched coverage additionally requires binary relevance one on the clicked query. CTR is clicks divided by displayed positions; age-based exposure does not measure provider-group fairness.')
insert_before(73, 'Post-arrival disruption metrics compare each served list with its base list: extrapolated rank-biased overlap (RBO; Webber et al., 2010; persistence 0.9), normalized Kendall discordance among shared documents, impression/query fractions with NDCG loss greater than 0.01, and minimum per-query mean ΔNDCG. We also report promoted new-item continuous/binary relevance, top-ten relevant fractions and discounted new-item exposure share.')
insert_before(73, 'Seed-level policy-minus-baseline differences use Student-t 95% intervals and paired two-sided t tests. Holm correction covers all nonbaseline test comparisons for coverage and NDCG across suites/conditions; other metrics are descriptive. Nominal CIs are not simultaneous bands. Per-seed values and all raw/adjusted p-values accompany the code.')
insert_before(73, 'We select λ from {0.1,0.2,0.3,0.4} on validation seeds 100–109 at p∈{0.15,0.30}, maximizing mean coverage gain subject to a paired 95% ΔNDCG lower bound of at least −0.01 at each penalty (fallback λ=0). Other defaults remain fixed. Test seeds 1000–1019 are disjoint from validation and the original seeds 0–19; headline results all use the selected λ and m=2.')
insert_before(73, 'Scoring comparators share the prefix and gate: novelty-only uses N, entropy-only H, unrefined entropy ηN+(1−η)H, and recency the new-item indicator. UCB-style clips p̂+sqrt[2 log(max(2,ΣI))/(I+1)] to [0,1]; Thompson draws Beta(C+1,I−C+1). Random injection swaps two pool items with probability 0.2; ε-greedy samples a remaining candidate with probability 0.2 per tail position. Local-swap exchanges disjoint neighbors when composite scores favor it; exposure-quota reserves one tail slot for an eligible new item if none would appear. These are heuristics, not full published safe-ranking algorithms; global click counters do not correct position bias.')
insert_before(73, 'Penalties are {0,0.10,0.15,0.20,0.25,0.30,0.40,0.60,0.80}; sensitivity uses η∈{0.35,0.50,0.65,0.80,1.00}, m∈{1,2,3}, p∈{0.15,0.30,0.40}. Ablations set m=0 or α=0, remove gating or rank normalization, use unrefined entropy, or hold N=1 (removing decay and the complementary refinement weight). These interventions need not preserve the default constraints.')
insert_before(73, 'Stress tests at p=0.30 use cascade/noisy clicks; b=0.8; query frequencies proportional to rank⁻¹·² with alternating b=0.15/0.8; navigational top-one relevance with click logit 5(y−0.5)−0.35j; 1,000 uniform queries; 50-step delayed feedback; and varying relevance counts from r≥0.78 without adjustment. This threshold targets sparse labels. All conditions retain the frozen λ and the same 20 test seed identifiers.')
patch(73)
patch(74,
    ('here .', 'at https://doi.org/10.5281/zenodo.19242717 (original archive). The revised code and per-seed results require a new version-specific DOI before submission.'),
    ('The study utilized Python 3.11.2 on a Windows workstation with an Intel i7-12700K processor and 32GB of Random Access Memory, and is fully reproducible through the provided baseline suites and artifact generation modules.', 'The revision utilized Python 3.12.14, NumPy 1.26.4, SciPy 1.15.0, pandas 2.3.1 and Matplotlib 3.11.2; exact dependencies and reproduction commands accompany the code.'),
    ('engagement metrics', 'descriptive metrics'),
    ('programmatically retrieved or generated', 'retrieved'))
insert_before(75, 'Replay uses the first 200,000 complete sessions in Yandex train.gz: a contiguous prefix, not a random sample or reconstruction of the historical 61,105-session selection. Its 372,282 pages each contain ten distinct documents; page/document identifiers map clicks, with test records and malformed pages excluded. The manifest records zero invalid pages and unmatched clicks. Events follow logged order: ranking uses prior counters, then logged impressions update counts and later clicks update the clicked document once per page. Logged order supplies rank-based scores; EDE uses binary entropy, m=2 and all ten candidates, without inferring unlogged items or birth dates.')
insert_before(75, 'For logged list B, proposed list A and clicked set C, exact-match means A=B as ordered lists, set-match means set(A)=set(B), and click preservation means C⊆set(A), allowing changed clicked positions. The latter two always pass on ten-item support. We report all-page and accepted/rejected diagnostics. Mean-metric 95% CIs use 2,000 whole-session bootstrap resamples (seed 20260911). Fixed observed labels produce zero ΔCTR; click-NDCG changes are descriptive, not treatment effects or cold-start evidence.')
patch(76,
    ('limited assistance in generating and refactoring minor code components', 'assistance in generating and refactoring simulation and analysis code and in literature discovery and references'),
    ('Generative AI was not employed for dataset synthesis, result generation, or the compilation of the reference list, the latter of which underwent manual verification to ensure accuracy.', 'Codex assisted with this revision’s code audit, experimental scripts, validation, manuscript, tables, figures and reviewer responses. Results were computed by the supplied programs; the authors must review the outputs and reconcile this disclosure with the original supplementary prompt log before submission.'))
insert_before(78, 'We report corrected simulation results first, followed by replay diagnostics; all tables are supplied separately.')
patch(79,
    ('EDE’s intended minimal-disruption behavior', 'average ranking-quality loss'),
    ('approximately 0.1797 and a coverage of approximately 0.0058', '0.1793 ± 0.0158 and a coverage of 0.087 ± 0.025 at p=0.15 on held-out seeds'),
    ('we swept λ while maintaining a minimum ΔNDCG@10 of -0.01.', 'we swept λ on independent validation seeds, requiring a paired 95% ΔNDCG interval lower bound of at least −0.01 at both validation penalties.'),
    ('A representative operating point (config C: λ=0.3, η=0.65, m=1) achieved a Δcoverage of +0.5603, with NDCG increasing slightly by +0.0021.', 'The selected operating point (λ=0.4, η=0.65, m=2) achieved a held-out Δcoverage of +0.099 [95% CI +0.087, +0.110] and ΔNDCG of −0.0026 [−0.0041, −0.0011] at p=0.15.'),
    ('substantially while remaining within a small ranking-quality budget', 'in this setting with a small average ranking-quality loss'),
    ('Figure 2 illustrates', 'The mean-quality rule does not guarantee quality for every query. Figure 2 illustrates'),
    ('discovery-quality Pareto frontier', 'discovery-quality trade-off'))
patch(81,
    ('rapidly ', ''),
    ('(decreasing from approximately 20% coverage at a penalty of 0.10 to approximately 0% at 0.60), whereas EDE maintains substantial discovery gains over a wide range (e.g., approximately 73% coverage at 0.10 and approximately 58% at 0.30)', '(from 0.087 at p=0.15 to 0.0055 at p=0.30), and EDE coverage also declines (0.186, 0.0117 and 0.0004 at p=0.15, 0.30 and 0.40)'),
    ('demonstrating that EDE maintains a consistent quality level across different penalty regimes', 'with paired ΔNDCG at p=0.30 of −0.0026 [95% CI −0.0041, −0.0011]'),
    ('Across easy-to-moderate penalties (0.10–0.30), EDE reduces', 'At p=0.15, EDE reduces'),
    ('by roughly 20–26% (e.g., approximately 19.5% faster at a penalty of 0.30)', 'from 938.6 ± 18.6 to 884.1 ± 21.5 query timesteps after arrival'),
    append=' Unique new docs shown were 50.9 ± 6.8 versus 20.4 ± 3.4 for baseline; query-matched EDE coverage was 0.075 ± 0.019.')
patch(83,
    ('simple exploration baselines', 'exploration baselines'),
    ('settings..', 'settings.'),
    ('Table 2 demonstrates that smoothed EDE (α=0.5) produced significant coverage gains while maintaining ranking quality.', 'Table 2 adds novelty-only, entropy-only, recency, UCB-style, Thompson, local-swap and exposure-quota comparators.'),
    ('For instance, at a penalty of 0.30, EDE-Smoothed achieved a Δcoverage of +0.566 and a ΔNDCG@10 of +0.0016, whereas ε-greedy resulted in a ΔNDCG@10 of approximately-0.0071.', 'At p=0.15, novelty-only coverage was 0.224 ± 0.044 versus 0.186 ± 0.032 for smoothed EDE (α=0.5). This comparison does not establish an additional discovery benefit from entropy.'))
patch(85,
    ('Table 3, Table 4 reports coverage and NDCG@10 sensitivity at a penalty of 0.15, while Table 5, Table 6 reports the same sensitivity at a penalty of 0.30.', 'Table 3 consolidates coverage and NDCG@10 sensitivity at penalties 0.15 and 0.30 and adds 0.40; η extends through 0.80 and 1.00.'),
    append=' These are held-out sensitivity results, not additional tuning. Table 4 reports component ablations and disruption diagnostics.')
insert_before(86, 'Table 4 also reports disruption and promoted-item relevance. At p=0.15, EDE’s RBO was 0.814 ± 0.009 and normalized shared-document Kendall distance was 0.040 ± 0.003. The fraction of post-arrival impressions with NDCG loss greater than 0.01 was 0.273 ± 0.043, and the worst observed query-mean ΔNDCG was −0.157 ± 0.030. Discounted new-document exposure share was 0.040 ± 0.005; promoted items had mean continuous relevance 0.712 ± 0.017 and binary query-relevance fraction 0.201 ± 0.093. Prefixes were preserved in every run, but tail losses remained.')
patch(87,
    ('To ensure EDE’s effectiveness remains consistent across diverse user behaviors,', 'To examine sensitivity to user behavior,'),
    ('As shown in Figure 3, EDE maintains a significant discovery advantage over all baselines, irrespective of the underlying user interaction pattern.', 'Figure 3 and Table 5 also test stronger position bias, heterogeneous query frequency/bias, navigational relevance, sparse queries, delayed feedback and naturally varying relevance counts. Outcomes depend on the condition and do not establish a universal robustness advantage; coverage denominators can differ between conditions.'))
replace(88, '4.6. Real Click-Log Replay (Yandex Dataset)')
patch(89,
    ('a conservative offline replay', 'an offline replay'),
    ('to evaluate whether EDE behaves as a stable, minimal-disruption re-ranking layer under realistic click data', 'to measure displacement on logged candidates'),
    ('Figure 4 reports replay acceptance rates and the frequency with which EDE changes positions within the top-10 among accepted sessions; under click preservation, EDE typically changes position by only approximately one.', 'Figure 4 and Table 6 report replay acceptance rates for 372,282 pages from 200,000 sessions: exact-list acceptance was 93.611% (348,498 pages), and set/click preservation was 100% by construction on the ten logged candidates.'),
    ('This supports the view that EDE functions as a conservative tail-re-ranking policy rather than an aggressive reordering method.', 'Mean changed positions per page were 0.149 [95% CI 0.147, 0.152], and mean absolute document shift was 0.014919 [0.014688, 0.015155].'),
    ('In this sense, the replay serves not only as an offline validation step, but also as evidence that EDE’s safety constraints operate as intended in a deployment-oriented setting.', 'These session-cluster bootstrap estimates measure neither cold-start discovery nor deployment safety.'))
replace(91, 'Ranking-change diagnostics distinguish exact-match accepted and rejected pages (Table 6). Set-match and click-preservation rejected buckets are empty, so their conditional means are undefined. Observed-label ΔCTR was exactly zero and the click-NDCG proxy difference was −0.000018 [95% CI −0.000023, −0.000013]; neither estimates live engagement. The historical ≈97% acceptance statement disagreed with its own CSV (60,781/61,105=99.4698%). Correcting page parsing, entropy and counter updates required a new sample and replacement of the old displacement, engagement and λ-sensitivity estimates.')
patch(93,
    ('significantly improves cold-start discovery while preserving ranking quality within a small and controlled NDCG budget', 'improves cold-start discovery in some synthetic settings with small average NDCG differences'),
    ('systematically increases exposure for uncertain but potentially relevant tail items', 'increases exposure for tail items'),
    ('Unlike blind random exploration, EDE’s informed strategy produces much larger discovery gains - for example, coverage gains that are approximately an order of magnitude higher in several settings.', 'Novelty-only is competitive or stronger, so the corrected results do not establish a distinct entropy benefit.'),
    ('; in most cases, we maintain a ΔNDCG threshold of at least -0.01', '; they do not bound losses elsewhere in the list'),
    ('Importantly, EDE remains effective across a range of difficulty regimes, from “easy” settings (where the baseline still discovers some new items) to “hard” settings (where the baseline finds almost none), although extremely', 'Extremely'),
    ('From a systems perspective, these results suggest that EDE is not only an exploration heuristic, but also a practical constrained re-ranking layer for improving cold-start exposure without requiring aggressive changes to the ranking stack.', 'The query-level diagnostics show losses that an average NDCG difference can conceal.'))
patch(95,
    ('further supports this deployment-oriented interpretation', 'provides a narrower displacement diagnostic'),
    ('In live settings, this means EDE could operate as a low-risk background layer that occasionally swaps in new items without materially disturbing the user-facing top of the list.', 'With only ten logged candidates and no cold-start labels, it cannot test the entry or discovery of new items.'),
    ('The small positive changes in Click-Through Rate (CTR) and click-NDCG under the click-preservation filter suggest that this exploration does not harm, and may slightly improve, user engagement in accepted sessions.', 'Reassigning fixed observed clicks to hypothetical positions does not estimate user engagement.'),
    ('it still provides useful evidence that EDE’s safety constraints function as intended and that the method behaves consistently with a minimal-disruption deployment model', 'it measures displacement on observed support, not live user-experience effects'))
insert_before(97, 'Unlike BubbleRank and Shiino et al. (2023), EDE does not learn pairwise evidence or confidence bounds; unlike Gupta et al. (2023), its prefix does not control counterfactual deployment risk. Direct comparisons under compatible feedback remain future work. Randomized logs could support IPS/doubly robust evaluation, followed by interleaving or A/B tests with CTR, abandonment, dwell time, complaints and discovery guardrails.')
insert_before(97, '6. Limitations', heading=True)
patch(97,
    append=' Fixed top-ten relevance labels, synchronized arrivals and the new-document penalty constrain external validity; the natural-threshold and intent tests remain stylized.')
insert_before(98, 'Only λ is validated; other parameters remain defaults explored through selected sensitivities. Heuristic comparators supply no regret or per-query safety guarantees, and latency is unmeasured. Yandex lacks cold-start labels and propensities; session bootstrapping cannot remove log-prefix selection bias or cross-session dependence. Updated items, new providers, new query-item pairs, seasonality and adversarial/low-quality arrivals are untested, and no provider-group fairness metric is available. Future work should test these cases with quality safeguards and adapt exploration to query frequency, ranker confidence, traffic and category-specific risk.')
replace(98, '7. Conclusion')
patch(99,
    ('cold-start learning-to-rank', 'cold-start ranking'),
    ('Restating our goals, we investigated whether we could improve the exposure of new relevant items without substantially degrading ranking quality. Our results strongly support this objective. ', ''),
    ('EDE produced significant gains in discovery metrics—specifically, coverage, time-to-first-click, and the number of unique new items—while keeping NDCG essentially stable.', 'EDE improved discovery in some synthetic settings with small average NDCG losses.'),
    ('Notably, EDE consistently outperformed random and ε-greedy baselines.', 'The corrected evidence does not establish a consistent entropy advantage over novelty alone.'),
    ('These improvements were also robust across multiple synthetic settings and were supported by a real click-log replay, where EDE behaved as a conservative, minimal-disruption tail re-ranking policy.', 'Real click-log replay measures displacement without cold-start ground truth. Stronger counterfactual and online evaluation is needed before deployment or engagement claims.'))
remove(101)
insert_before(102, 'Data Availability', heading=True)
insert_before(102, 'The original code archive is https://doi.org/10.5281/zenodo.19242717; the revised code, protocol, dependency list and per-seed results accompany this submission and require a new version-specific DOI. Yandex training data are available from https://www.kaggle.com/c/yandex-personalized-web-search-challenge, subject to source terms. Raw logs are not redistributed. The acquisition script and SHA-256 manifest specify the exact prefix used in Section 3.5.')
patch(103, append=' The broader scope of AI assistance is described in Section 3.6.')

# Retain the original references used by the retained text. Correct only needed entries and add cited work.
for i in [108,110,114]:
    remove(i)
replace(116, op[116].text.replace('Jarvelin K, Kekalainen J', 'Järvelin K, Kekäläinen J'))
replace(118, 'Kveton B, Szepesvári C, Wen Z, Ashkan A. 2015. Cascading bandits: learning to rank in the cascade model. Proceedings of Machine Learning Research 37:767–776. https://proceedings.mlr.press/v37/kveton15.html.')
replace(119, 'Lattimore T, Kveton B, Li S, Szepesvári C. 2018. TopRank: a practical algorithm for online stochastic ranking. Advances in Neural Information Processing Systems 31. https://proceedings.neurips.cc/paper/2018/hash/de03beffeed9da5f3639a621bcab5dd4-Abstract.html.')
replace(120, 'Li C, Kveton B, Lattimore T, Markov I, de Rijke M, Szepesvári C, Zoghi M. 2020. BubbleRank: safe online learning to re-rank via implicit click feedback. Proceedings of Machine Learning Research 115:196–206. https://proceedings.mlr.press/v115/li20b.html.')
replace(130, op[130].text + ' https://jmlr.org/papers/v16/swaminathan15a.html.')
newrefs = [p.text for p in ep if p.text.startswith(('Agarwal A,', 'Du Y,', 'Fang Z,', 'Gupta S,', 'Shiino H,', 'Wang D,', 'Webber W,'))]
for ref in newrefs:
    surname = ref.split(',')[0]
    anchor = next((i for i in range(105,133) if op[i]._p.getparent() is not None and op[i].text.split(',')[0] > surname), None)
    if anchor is not None:
        insert_before(anchor,ref)
    else:
        d.add_paragraph(ref)
d.add_page_break()
legend=d.add_paragraph()
if op[92]._p.pPr is not None:
    legend._p.insert(0,deepcopy(op[92]._p.pPr))
legend_run=legend.add_run('Figure Legends')
if op[92].runs[0]._r.rPr is not None:
    legend_run._r.insert(0,deepcopy(op[92].runs[0]._r.rPr))
captions = [p.text.replace('Section 3.5', 'Section 3.4') for p in ep if re.match(r'^Figure [1-4]\.', p.text)]
for caption in captions:
    d.add_paragraph(caption)
(OUT / 'figure_captions.txt').write_text('\n\n'.join(captions),encoding='utf-8')

# Preserve original continuous line numbering and page setup explicitly.
for section in d.sections:
    line = section._sectPr.find(qn('w:lnNumType'))
    if line is None:
        line = OxmlElement('w:lnNumType')
        section._sectPr.append(line)
    for key,value in [('countBy','1'),('start','0'),('restart','continuous'),('distance','240')]:
        line.set(qn('w:'+key),value)
for style in d.styles:
    if style.type == 1:
        style.font.name = 'Times New Roman'
        if style.name == 'Normal':
            style.font.size = Pt(12)
for p in d.paragraphs:
    if p._p.pPr is not None:
        for n in list(p._p.pPr):
            if n.tag == qn('w:suppressLineNumbers'):
                p._p.pPr.remove(n)
d.save(OUT / 'cs-134580-v0.4.docx')
old_words = re.findall(r'\S+', '\n'.join(p.text for p in Document(REV / 'original_submitted/cs-134580-v0.4.docx').paragraphs))
new_words = re.findall(r'\S+', '\n'.join(p.text for p in d.paragraphs))
kept = sum(m.size for m in SequenceMatcher(None,old_words,new_words,autojunk=False).get_matching_blocks())
(WORK / 'minimal_edit_audit.json').write_text(json.dumps({'original_words':len(old_words),'revised_words':len(new_words),'original_words_retained_in_order':kept,'retained_fraction':kept/len(old_words),'edits':edits},ensure_ascii=False,indent=2),encoding='utf-8')
print('Original words retained in order:',kept,'/',len(old_words),f'({100*kept/len(old_words):.1f}%)')
print('Saved original-format manuscript with continuous line numbers')
