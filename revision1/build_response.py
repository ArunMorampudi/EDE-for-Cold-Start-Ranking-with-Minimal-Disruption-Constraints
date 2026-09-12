"""Point-by-point responses retaining the complete supplied reviewer comments."""
from pathlib import Path
import re,json
from build_submission import document,p,h,save

R={}
def responses(reviewer,section,items):
    for number,reply in enumerate(items,1): R[(reviewer,section,number)]=reply

responses(1,'Basic reporting',[
'Thank you. Section 1 retains the feedback-loop motivation and removes repetitive deployment language.',
'Section 2 now compares assumptions, objectives, feedback and guarantees of BubbleRank, Shiino et al. (2023), exposure-risk minimization, exposure-constrained ranking and diversification. EDE is positioned as an empirical heuristic with an exact prefix constraint, not a new general safety guarantee.',
'Section 2 now discusses propensity weighting, counterfactual learning-to-rank, intervention harvesting and exposure-based conservative learning. Section 3.6 explicitly explains why this replay cannot perform off-policy evaluation.',
'The Abstract, Sections 1, 3.6, 4.5, 5, 6 and 7 consistently distinguish synthetic discovery evidence from the restricted logged-support diagnostic. The corrected replay does not measure real cold-start discovery.',
'All four captions now state sample sizes, parameter settings and uncertainty conventions. Figures 1 and 3 show sample SD across 20 test seeds; Figure 2 shows paired 95% validation CIs; Figure 4 shows session-cluster bootstrap CIs for displacement.',
'Unsupported significance language has been removed. Section 3.4 specifies paired tests and Holm correction; complete raw and corrected p-values accompany the results. Descriptive conclusions are used unless supported by the specified analysis.'
])
responses(1,'Experimental design',[
'Sections 1 and 3 introduce the two evidence sources separately. The synthetic experiments are the only direct cold-start evaluation.',
'Section 3.1 specifies 16-dimensional standard-normal embeddings, normalization, 100 queries, 1,000 documents, 200 new documents, ten relevant documents per query, arrival at 1,000 and horizon 2,000. Section 3.2 describes one query and one trial per displayed position per timestep, with independent shared random streams.',
'A naturally varying relevance condition now uses r≥0.78 without top/bottom adjustment. Section 3.5 explains the choice and Table 5/Figure 3 report outcomes. We also acknowledge the artificial fixed-top-ten labels and do not assume they cannot favor a policy.',
'The gradient-boosted LTR implementation claim is removed. Equation (1) and Section 3.1 define the actual stationary noisy relevance ranker, and distinguish it from intended production use.',
'Section 3.5 prespecifies lambda selection on seeds 100–109 and tests only on seeds 1000–1019. Other defaults are fixed and subjected to ablation/sensitivity, not claimed to be jointly optimized. The selection protocol is saved with the results.',
'Sections 1 and 3.3 distinguish click entropy from epistemic relevance uncertainty. Novelty-only, entropy-only, unrefined and full EDE comparisons are added. Results and Discussion explicitly acknowledge that novelty-only is competitive or stronger, limiting the claimed entropy contribution.',
'Section 3.5 adds stronger position bias, heterogeneous query frequency/bias, navigational relevance, sparse queries and delayed counters, in addition to cascade/noisy models. Figure 3 and Table 5 report 20-seed uncertainty. These are stylized stress tests, not validated population models.',
'Section 3.6 formalizes all three filters and fully specifies the new log-prefix sample and page/document mapping. The original parser merged result pages; the corrected parser preserves page IDs and event ordering. Set/click preservation is now honestly identified as structural with ten logged candidates. Table 6 and supplementary replay data distinguish exact-match acceptance/rejection.',
'Main tables/figures show SD or CIs as labeled, and supplementary CSVs provide per-seed means, SD, CIs and paired deltas for every requested simulation metric. Replay intervals now come from actual session-cluster bootstrapping; the old disabled-bootstrap output is withdrawn.'
])
responses(1,'Validity of the findings',[
'The revised results are explicitly simulator-based. The original large gains and claims of real-system effectiveness have been replaced by the smaller corrected estimates.',
'The audit found that the old policy removed the new-document penalty after gating, unlike its baseline. The corrected comparison retains the same penalized score for both policies, and all results are rerun. Table 4 adds continuous/binary query relevance of promoted items and exposure diagnostics; p=0 is included. The original improvements cannot be attributed solely to entropy.',
'Table 4 and supplementary CSVs add RBO, normalized shared-document Kendall distance, worst observed query-mean NDCG delta, top-ten relevance fraction and fractions exceeding a 0.01 degradation threshold. These reveal tail losses despite exact prefix preservation.',
'The replay is now described throughout as a logged-support position-change diagnostic without cold-start labels. Real discovery claims are removed.',
'Table 6 separates exact-match accepted and rejected pages and their observed click/displacement metrics. Set/click preservation has no rejected pages because the corrected candidate set contains exactly the logged ten. This is stated as a limitation, not reported as a successful discovery/safety result. Hypothetical clicked-document shifts are retained in the summary.',
'The expanded comparison includes novelty-only, entropy-only, recency, UCB-style, Thompson, adjacent local-swap and exposure-quota policies, as well as random and epsilon-greedy. All share the gate and prefix. Sections 3.5 and 6 explicitly state that these are operational heuristics, not exact implementations of BubbleRank or published constrained optimizers.',
'Table 4 adds prefix removal, gate removal, no novelty decay, unrefined entropy, alpha=0 and no rank-normalization ablations. The difficulty sweep includes no penalty. Section 3.3 corrects the nonexistent discrete grace-period description to the actual smooth exposure weighting.',
'Unsupported “significantly” statements are removed. Seed-level paired t tests and Holm correction are described in Section 3.4, and raw/corrected p-values are supplied for the entire stated family.',
'The Conclusion now calls EDE a constrained reranking heuristic with conditional simulator gains and a restricted offline diagnostic. It explicitly requires stronger counterfactual and online evaluation before deployment or engagement claims.'
])
responses(1,'Additional comments',[
'Section 6 separately states the missing cold-start labels, absent propensities, contiguous-prefix sampling and acceptance-selection limitations.',
'Section 5 discusses randomized exploration logs, IPS/doubly robust estimators, interleaving and online A/B tests with CTR, abandonment, dwell time, complaints and longer-term discovery guardrails. None is falsely claimed to have been conducted.',
'Section 6 proposes query-frequency, confidence, traffic and category-sensitive parameter adaptation. Current defaults are acknowledged to be only partially selected.',
'Section 6 explicitly identifies updated items, new providers, new query-item pairs, seasonality and adversarial/low-quality arrivals as untested cases requiring independent safeguards.',
'We add new-item discounted exposure share and top-ten new-item fraction, while removing any implication that these establish provider-group fairness. The absence of group labels and group fairness metrics is explicit in Sections 4.3 and 6.'
])
responses(2,'Basic reporting',[
'Sections 1–2 reposition novelty against safe unranked-item exploration (Shiino et al., 2023), exposure-risk minimization (Gupta et al., 2023), item-centric exploration (Wang et al., 2025) and stateful cascading reinforcement learning (Du et al., 2024). The differences concern feedback, objective and guarantee, not merely lightweight implementation.',
'Eight equations are numbered and referenced in Section 3. Actual values a=3, b=0.35, tau=50 and n_rel=10 are stated, together with the full simulator defaults. The equation explanations now match the corrected code.',
'Table 2 uses penalty rows and strategy columns with baseline coverage reported once. Original Tables 3–6 are combined in new Table 3: four panels cover p=0.15/0.30, and two additional panels cover p=0.40. New Tables 4–6 contain ablations, stress tests and corrected replay.',
'All replacement figures are generated at 300 DPI with shared typography, fixed colors and markers, and visible panel letters. Figure 2 is regenerated with the explicitly selected validation grid and offset labels, replacing the overlapping historical lambda annotations.',
'Data Availability and a broader AI disclosure are added. Funding and competing-interest declarations require confirmation from both authors; neither the supplied manuscript nor repository establishes them, and we do not infer “no funding” or “no conflicts” from independent-researcher affiliations. These two declarations remain an author action before upload. The original supplementary AI prompt log was not supplied in this checkout, so the disclosure must also be reconciled with that log.',
'References are standardized and include a DOI where verified or an identifiable publisher/archive URL. The two uncited textbook/NeurIPS entries are removed rather than retained merely to meet a DOI convention. We do not invent DOIs for proceedings with publisher URLs. The journal instruction page was inaccessible during this revision, so no unverified claim of portal/style compliance is made.'
])
responses(2,'Experimental design',[
'Section 3.7 and Data Availability print https://doi.org/10.5281/zenodo.19242717 visibly as the original archive. The revision branch and exact protocol are identified separately. A new version DOI is an explicit author action; the old DOI is not presented as if it contains the revised code.',
'A revision-specific dependency file pins the actual numerical runtime and plotting/test packages. The invalid “-e .” requirement is removed because this repository has no package build metadata. Python-module commands run from the repository root.',
'Section 3.3 specifies the smooth weighting numerically: tau=50 gives refinement weights of approximately 0.095 at five impressions and 0.632 at 50. There is no discrete grace period. Hard mode is defined by a positive pre-gating new-document penalty retained in both policies.',
'Table 3 extends eta to 0.80 and 1.00 at three penalties, including 0.40. This is reported as held-out sensitivity, not a post hoc change to the selected operating point.',
'The old mismatched m=1 headline operating point is replaced. Table 1 confirmation and Figure 1 both use m=2 and the validation-selected lambda, with any alternative m confined to sensitivity/ablation.',
'The real-log section is now explicitly a page-level logged-support diagnostic. It does not evaluate cold-start discovery or a new-candidate insertion policy.'
])
responses(2,'Validity of the findings',[
'Tables now show sample SD and/or paired CIs, and the supplementary results provide all per-seed values, paired t tests and Holm-adjusted p-values. The manuscript no longer calls small NDCG deltas significant without this analysis.',
'The archived diagnostics marked bootstrapping disabled, so the old repeated endpoints were not valid uncertainty estimates. The replay has been rerun with 2,000 session-cluster bootstrap resamples, and Table 6/Figure 4 report the resulting mean-metric intervals.',
'The historical CSV records 60,781/61,105=99.4698%, not approximately 97%. After correcting the parser, entropy and event updates, Section 4.5 reports the replacement sample and results. The new set/click acceptance is 100% by construction and is not interpreted as empirical safety evidence.',
'Table 4 explicitly includes alpha=0 with a finite zero-count convention. Full EDE, unsmoothed and other scoring ablations are reported with seed-level uncertainty.'
])
responses(2,'Additional comments',[
'Thank you. The revised design retains multiple click models and 20 test seeds, while adding independent validation, paired random streams and more stress conditions.',
'The paper has been edited throughout, and all figure legends use consistent punctuation, parameter reporting and uncertainty descriptions.'
])
responses(3,'Basic reporting',[
'The Introduction has been shortened to the motivation, overlap with prior work, mechanism and scope. Repeated deployment and minimal-disruption claims have been removed.',
'“Real-time” is replaced with recent, online-available click signals. The real-data experiment is consistently described as offline and descriptive.',
'Sections 1–2 narrow novelty to an empirical scoring/constraint combination and acknowledge the overlap with safe cold-start exploration. No first-of-its-kind or general low-risk claim remains.',
'Shiino et al. (2023) is directly discussed, together with exposure-based safe deployment (2023), cascading reinforcement learning (2024), and item-centric exploration (2025). References are included for topical relevance rather than recency alone.',
'Section 2 connects the two streams through safe exploration of previously unranked items and distinguishes item-centric audience selection from query-conditioned tail reranking.',
'Eight equations are numbered and referenced; entropy and ranking metrics are cited and all symbols/defaults are defined. Reference entries use a consistent author-year form.',
'Section 3.4 defines TTF relative to arrival, with one query per timestep and right-censoring at 1,000 post-arrival steps. It includes all eligible documents, not only clicked documents, and removes misleading old percentage speedups.',
'Unsupported significance language is removed. Paired seed-level tests, CIs and Holm correction are now provided.',
'The erroneous trained-LTR claim is removed. Section 3 defines the actual noisy ranker, exact gate, normalization, penalty and prefix logic. The audit also corrected code mismatches, requiring all headline results to be regenerated.',
'Each table is now explicitly cited in Results; tables remain separate upload files as required by the author’s submission workflow. Figure legends are included in the manuscript and supplied separately.',
'Results now include paired uncertainty, a validation/test split and stronger diagnostics. Claims are narrowed to what the corrected numerical evidence supports; large old improvements are not retained.',
'The comparison now includes an adjacent local-swap policy and an exposure-quota policy, alongside novelty, entropy, recency, UCB-style and Thompson heuristics. They are transparently defined rather than presented as exact BubbleRank implementations; comparison to full published algorithms remains a limitation.',
'Lambda is selected on seeds 100–109 and evaluated on disjoint seeds 1000–1019. The selection criterion and complete validation outputs are saved; stress conditions do not retune lambda.',
'Methods and Results now begin with orienting paragraphs. Policy, metrics, validation and replay are separated into explicit subsections with connected explanations.',
'Discussion is expanded to interpret the smaller corrected gains, novelty-versus-entropy findings, quality-loss diagnostics and differences from prior safe methods. It also explains what the replay can and cannot establish.',
'Limitations now has its own section covering simulator assumptions, partial tuning, heuristic comparators, missing live evaluation, unavailable propensity/cold-start labels, selection bias, fairness and adversarial/updated content.',
'The absence of real cold-start ground truth is stated in the Abstract, Introduction, replay methods/results, Discussion, Limitations and Conclusion. The replay is not represented as discovery evidence.',
'The Conclusion is shortened and qualified. It reports conditional simulator effects, no established general entropy advantage, and the need for stronger evaluation before deployment claims.'
])

def main():
    # Retain the original manuscript section order in the conservative revision.
    section_map={'3.5':'3.4','3.6':'3.5','3.7':'3.5','4.5':'4.6','4.3':'4.4'}
    for key, reply in list(R.items()):
        R[key]=re.sub(r'\b(?:3\.5|3\.6|3\.7|4\.5|4\.3)\b',lambda m: section_map[m.group()],reply)
    R[(2,'Basic reporting',6)]='Original references used by the retained manuscript are preserved. Inconsistent author-year entries are corrected, recent cited sources are added, and the TopRank proceedings entry now has its official publisher URL. The cited reinforcement-learning textbook is retained; we do not invent a DOI where one is not established.'
    R[(3,'Basic reporting',14)]='Methods and Results now begin with orienting paragraphs. The original section order is retained, with the additional validation and replay details placed in Sections 3.4 and 3.5.'
    R[(3,'Basic reporting',1)]='The Introduction retains its original motivation and contribution structure. Repetitive deployment language and unsupported low-risk claims are shortened or qualified, with focused additions on overlapping prior work.'
    R[(2,'Additional comments',2)]='We preserve the original wording except where reviewer requests or the corrected evidence require a change. Figure legends consistently report parameters and uncertainty.'
    R[(2,'Basic reporting',3)]='Each numbered table is now one editable Word table in its own one-page DOCX, cited in order in Results. Table 2 retains penalty rows and strategy columns in three bands within one grid, with baseline coverage reported once. Table 3 combines the original Tables 3–6 in columns (a)–(d), with columns (e)–(f) adding p=0.40. Reviewer-requested ablations, stress tests and replay diagnostics occupy Tables 4–6; detailed numerical results remain in the archived CSVs.'
    R[(2,'Basic reporting',5)]='Data Availability is added. Funding and competing-interest declarations require confirmation from both authors; neither the supplied manuscript nor repository establishes them. Section 3.6 retains the original author disclosure. The authors must verify that it accurately covers the revision and reconcile it with their supplementary AI prompt log before submission.'
    R[(2,'Basic reporting',3)]+=' Table titles and explanatory notes are provided in a separate reference document for entry in the submission form; each table DOCX contains only its grid.'
    R[(2,'Additional comments',2)]+=' Figure legends are supplied as a separate document.'
    feedback=Path('revision1/revision1_feedback').read_text(encoding='utf-8')
    d=document('Response to reviewers for PeerJ Computer Science manuscript 134580')
    p(d,'We have revised the manuscript while retaining its original structure and wording where the evidence permits, regenerated the simulation and replay analyses, updated the separate tables and figures, and supplied a tracked-changes manuscript with continuous line numbering. The code audit uncovered discrepancies in prefix construction, penalty handling and replay semantics that affected the submitted evidence. The revised paper reports the corrected, smaller effects and withdraws unsupported engagement and deployment-safety claims.')
    p(d,'Locations below use stable section/table/figure identifiers. Complete original reviewer comments are retained beneath each identifier. The final funding/competing-interest declarations and new revision-specific Zenodo DOI require author confirmation before submission.')
    reviewer=1; section='Basic reporting'; current=None; entries=[]; ordinal={}
    for line in feedback.splitlines():
        line=line.strip()
        if line.startswith('Reviewer:') or line=='Reviewer 2' or line=='Reviewer 3':
            if current: entries.append(current); current=None
            reviewer=1 if line.startswith('Reviewer:') else int(line[-1]); continue
        if line in ['Basic reporting','Experimental design','Validity of the findings','Additional comments']:
            if current: entries.append(current); current=None
            section=line; continue
        match=re.match(r'^(\d+)\.\s*(.+)',line)
        if match:
            if current: entries.append(current)
            key=(reviewer,section); ordinal[key]=ordinal.get(key,0)+1
            current=dict(reviewer=reviewer,section=section,ordinal=ordinal[key],original_number=match.group(1),comment=match.group(2)); continue
        if current and line and not line.startswith('_') and not line.startswith('**PeerJ'):
            if line in ['These are independent researchers - not sure if this info is relevant.','16.']: continue
            current['comment']+=' '+line
    if current: entries.append(current)
    last=None; audit=[]
    for entry in entries:
        key=(entry['reviewer'],entry['section'],entry['ordinal'])
        reply=R.get(key)
        if reply is None: raise ValueError(('Missing response',entry))
        group=key[:2]
        if group!=last: h(d,f'Reviewer {group[0]} {group[1]}'); last=group
        h(d,f'Comment {entry["original_number"]}',2)
        p(d,entry['comment']); p(d,'Response. '+reply)
        audit.append(dict(entry,response=reply))
    save(d,'cs-134580-response-to-reviewers.docx')
    Path('revision1/reviewer_response_matrix.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Responses:',len(audit))

if __name__=='__main__': main()
