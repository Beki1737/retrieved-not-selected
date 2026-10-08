# Pre-registration (committed before analysing the 32B outputs; after the 7B development pilot)
Primary match notion: exact five-vote configuration (strict earlier, undefined pairs reported under lower and upper bounds).
Primary LLM metrics: macro-F1, macro-F1 after Batch Calibration, AUROC of P(against), Yes-rate gap; resolution-cluster bootstrap; Holm across H5-H7.
H1 Crowding: for mode-blind retrievers on vetoed/failed queries, the largest failure share lies between R@50 and R@10.
H2 Composition law: mode-blind retrievers' R@k is predicted by top-50 composition (hypergeometric); entropy and stratified rerankers exceed it.
H3 Crossover: entropy reranking raises R@1 on split/vetoed and lowers it on consensus, also under symmetric text (Config A').
H4 Stratification: strat-bge R@3 >= bge in every regime and not below CSR3 on consensus, split and failed.
H5 Evidence use: D-strat4-coal and D-oracle-coal exceed B0-std in AUROC(P against); D-oracle-coal exceeds D-random-coal in macro-F1 (BC).
H6 Prior: Batch Calibration raises macro-F1 in all conditions; the evidence advantage of H5 persists after calibration.
H7 Amplification: CAD (alpha=1, fixed a priori) and AdaCAD raise macro-F1 (BC) for relevant evidence but not for D-random-coal.
Tuning rule: any tuned parameter is fit on 2013-2018 drafts and evaluated on 2019-2024; all conditions are reported.

## Wave 2 (added 2026-10-02, before any wave-2 output exists; analysis = src/coalrag/analysis/e5_analysis.py)
H8 Causal evidence use: P(true) is higher with the oracle precedent's votes shown than with its text only (F-orc-coal > F-orc-text), and flipping the displayed target vote lowers P(true) (F-orc-flip < F-orc-coal).
H9 Evidence-level affirmative prior: uptake of a displayed 'against' when the truth is favour is smaller than uptake of a displayed 'favour' when the truth is against (uptake measured against the text-only arm).
H10 Coalition inference: the other four members' votes alone raise AUROC(P against) over text only (F-orc-others > F-orc-text).
H11 Robustness: the sign of AUROC(N) for D-strat4-coal minus B0-std holds under every discovered Choi paraphrase and under reversed and shuffled precedent order.
H12 No false-veto cost: on adopted consensus drafts, D-strat4-coal does not significantly raise the false-against rate over B0-std.
Descriptive only: memorization split (M-symbol), k = 2/4/6, Qwen3 thinking on vs off, Mistral-24B-2501 and Qwen2.5-7B replications.

## Selection and augmentation (added 2026-10-03, before any E7 or E8 output; analyses: analysis/e7_analysis.py, experiments/e8_augment.py)
H13 Selection: aggregating the four single-precedent predictions with the model's own choice of the closest precedent (S-llmsel; texts only, no votes) raises AUROC(P against) over showing the four stratified precedents together (D-strat4-coal). One-sided, Holm across models. Secondary metric: macro-F1 after Batch Calibration.
H14 Bound: taking the single-precedent prediction of the stratified slot whose five-vote configuration equals the draft's (S-perfect; uses the label, so a bound and not a method) exceeds D-strat4-coal in AUROC(P against). The oracle gap is reported as selection + coverage.
H15a Outcome filter: in Choi's adopted-only pool, adding dissent-bearing documents raises Sup and bge R@10 for non-adopted drafts; adding unanimous documents changes neither.
H15b Topic: at equal K, topically similar dissent-bearing documents raise bge R@10 more than random ones.
H15c Dilution: in the unfiltered pool, adding K topically similar unanimous documents lowers bge R@10 on contested drafts, while stratified R@10 does not fall significantly.

## Follow-up analyses E7 and E8 (amendment; committed 2026-10-03 after their first outputs)
Specified on 2026-10-03 at about 12:30 KST in the analysis plan and in configs/cell49_select_augment.sh (text below, unchanged), before any E7 or E8 output existed. That bundle refused to run because E7 (Mistral) and E8 had already been executed outside it; logs/e7e8_provenance.log records whether the executed code is identical to the specified code. E7 and E8 are therefore reported as follow-up analyses with rules fixed before results, not as part of the original pre-registration. E8b (stratification depth under dilution) was specified after the E8 results and is exploratory.
H13 Selection: aggregating the four single-precedent predictions with the model's own choice of the closest precedent (S-llmsel; texts only, no votes) raises AUROC(P against) over showing the four stratified precedents together (D-strat4-coal). One-sided, Holm across models. Secondary metric: macro-F1 after Batch Calibration.
H14 Bound: taking the single-precedent prediction of the stratified slot whose five-vote configuration equals the draft's (S-perfect; uses the label, so a bound and not a method) exceeds D-strat4-coal in AUROC(P against). The oracle gap is reported as selection + coverage.
H15a Outcome filter: in Choi's adopted-only pool, adding dissent-bearing documents raises Sup and bge R@10 for non-adopted drafts; adding unanimous documents changes neither.
H15b Topic: at equal K, topically similar dissent-bearing documents raise bge R@10 more than random ones.
H15c Dilution: in the unfiltered pool, adding K topically similar unanimous documents lowers bge R@10 on contested drafts, while stratified R@10 does not fall significantly.

## Wave 4: H16 (D91), committed before any E7b output
Selector S-lift: among the k = 4 stratified precedents of a draft (the E7 slots), show alone the one whose five-vote configuration is
most over-represented among the draft's 20 nearest earlier documents (bge-small on UNSC-CKG text, Choi pool, strictly earlier)
relative to its share of the whole eligible pool:
  score(c) = log[(n_c,20 + 0.5) / (20 + 0.5k)] - log[(N_c + 0.5) / (N + 0.5k)]; ties go to the more similar slot.
The rule uses neither the draft's votes nor its outcome. Code: src/coalrag/analysis/e7b_structure.py (this commit).
- H16a: S-lift > D-strat4-coal in AUROC(N) on the 66 x 5 rows; one-sided resolution-cluster paired bootstrap (B = 2000);
  Holm across the models with E7 output (Llama-70B, Mistral-24B, Qwen-32B).
- H16b (retrieval only, every draft of the unfiltered pool with complete votes): selection hit (the chosen slot carries the
  draft's configuration) of S-lift minus the slot-1 rule is > 0 on split, vetoed and failed drafts and < 0 on consensus drafts
  (an expected trade-off); paired bootstrap over drafts (B = 5000), reported per regime.
- Exploratory, no support claim: soft weights exp(score); neighbourhood 10 and 50; S-lift vs S-llmsel and S-top1; false-against rate.

## Wave 5: H17 (D99), committed before any E10 output
E10 guaranteed-match selection experiment (src/coalrag/experiments/e10_guaranteed.py, this commit). Drafts: the non-adopted drafts that
have an earlier precedent with exactly the same five-vote configuration (oracle) and at least three earlier stratified precedents of
other configurations. Evidence set = the oracle precedent + the three best-ranked stratified precedents of other configurations, so the
matching precedent is always present and exactly one of four configurations matches. All conditions are scored in one run per model.
- H17a: AUROC(N) of the matching precedent shown alone > the four shown together (mean over the four positions of the match).
- H17b: AUROC(N) of the matching precedent shown alone > the precedent the model itself selects (texts only), shown alone
  (mean over the four orderings).
- H17c: the model's selection hits the matching precedent more often when it is listed first than when it is listed last.
- H17d: AUROC(N) of the four shown together is higher with the match first than with the match last.
One-sided resolution-cluster bootstrap (B = 2000); Holm within model over H17a-d; models: Llama-70B, Mistral-24B, Qwen-32B, Qwen-7B.
Exploratory, no support claim: four together vs no evidence; first-ranked and random single-precedent strategies; selection accuracy
by position vs chance. Also exploratory (D98): the harmonized re-analysis of E7 (e7c) and the data accounting (accounting.py).

## Wave 6 (H18): augmentation tested at final prediction (E11)

Motivation: design review item asks for the corpus augmentation to be tested at retrieval AND final prediction.
E8 (H15) tested retrieval only. E11 re-uses the E8 augmentation sets unchanged.

Design (fixed before any E11 code is run):
- Queries: the 66 non-adopted drafts x 5 P5 members = 330 rows.
- Base pool: the 515 adopted LT-RAG documents dated strictly before each draft (outcome-filtered pool).
- Text space: UNSC-CKG text for every document (added records have no released summary), as in E8.
- Arms (identical record selection to E8, K = 100 pre-2013 archive records per draft):
  A0 none; A1 +100 most similar dissent-bearing; A2 +100 random dissent-bearing; A3 +100 most similar unanimous.
- Evidence per arm: four stratified precedents (depth 50, bge dense ranking over the augmented pool), all five P5 votes
  shown, same prompt template and option-likelihood scoring as D-strat4-coal (D73 serving standard).
- Additional condition: A1-match-alone = the matching precedent alone (all votes) when the A1 four slots contain one.
- Reference: B0 (no evidence), re-scored in the same run.
- Models: Llama-3.3-70B-FP8, Mistral-Small-24B-2501-FP8, Qwen2.5-32B-GPTQ-Int8, Qwen2.5-7B. All conditions of a model in one run.
- Metrics: AUROC(N) primary; BC-F1 and P(true) secondary; manipulation check = share of drafts whose four slots contain a match.
- Statistics: paired resolution-cluster bootstrap, B = 2000, one-sided, Holm within model over H18a-c.

Hypotheses:
- H18a: AUROC(N), four stratified, A1 > A3 (same quantity; dissent-bearing vs unanimous).
- H18b: AUROC(N), four stratified, A1 > A0 (augmentation vs outcome-filtered pool).
- H18c: on drafts whose A1 four slots contain a match, AUROC(N) of the match alone > the four together (H17a in a realistic pool).
Exploratory: A1 vs A2 (similar vs random), coverage by arm, BC-F1 and P(true) contrasts, A1 four stratified vs B0.
Expectation stated in advance (not a hypothesis): if selection is the bottleneck, H18a/H18b gains are small relative to the coverage gain, and H18c holds.

## Wave 7 (H19): content appropriateness of precedents (E12)

Motivation: design review item (8 Oct). Identical five-member votes do not make a precedent content-appropriate for the
agenda item. E12 judges appropriateness with votes and outcomes masked and tests whether the model could have identified
appropriate precedents from the information it was given. Fixed before any E12 code or output.

Design:
- Pairs: every (draft, candidate) pair in the E10 candidate sets (matching precedent + three distractors; 50 drafts) and in
  the E7 stratified slots (66 non-adopted drafts).
- Judges: J1 Llama-3.3-70B-FP8 (primary), J2 Qwen2.5-32B-GPTQ-Int8 (second family). Option-likelihood scoring, D73 serving.
- Questions: T same situation or agenda item (1-3); C same countries or parties (1-3); A same kind of Council action (1-3);
  O appropriate precedent judging only by content (1 no, 2 yes).
- Information: P = exactly the text the predicting LLM sees (draft text; candidate date and summary) with votes and outcome
  removed; R = P plus UNSC-CKG metadata (agenda item, subjects, countries concerned, action items, keywords). Sentences with
  vote or outcome wording are removed from every field.
- Primary label: appropriate = P(O = yes) >= 0.5 under J1, information P.
- Validation: blind human annotation of 120 pairs (30 E10 drafts x 4 candidates, provided information only); agreement and
  Cohen's kappa with J1-P; a second annotator on a subset if available. Same-agenda metadata rule reported as a reference.
- Statistics: draft-cluster bootstrap, B = 2000, one-sided; Holm across the four models for H19b and H19d.

Hypotheses:
- H19a: in E10 sets, the matching precedent is judged appropriate more often than the distractors.
- H19b: per model, the model's own pick (E10, four orderings) is appropriate more often than a random pick from the same set.
- H19c (decision analysis): on U = E10 drafts whose matching precedent is the only appropriate candidate, the hit rate of the
  model's pick, pooled over models, with a 95% CI. Reading fixed in advance: upper bound < 0.5 -> "selection failure among
  content-appropriate precedents"; lower bound > 0.5 -> the claim is restricted to "the model did not consistently select
  precedents whose votes matched"; otherwise inconclusive. Fewer than 10 drafts in U -> inconclusive by design.
- H19d: per model, AUROC(N) of the candidate J1-P rates most appropriate, shown alone (existing E10 single-precedent rows),
  exceeds that of the model's own pick shown alone.
Exploratory: J1 vs J2, P vs R, metadata rule, E7 sets, rubric means for matches vs non-matches.

## Wave 8 (H20): generation-stage replication on the UN General Assembly (E13)

Motivation: design review item. Auxiliary to the Security Council study; generalization is limited to UN voting simulation.
Fixed before any E13 code or output.

Design:
- Queries: General Assembly roll calls 2013-2019 with all five P5 votes recorded, a matching precedent (identical P5
  configuration) among earlier roll calls after removing those with the same normalized title (E9 rule), and at least three
  earlier precedents of other configurations among the stratified candidates. If more than 150 qualify, a uniform random
  sample of 150 (seed 0). Rows: query x P5 member with a recorded vote.
- Candidate set (E10 construction): the most similar matching precedent + the three best-ranked stratified precedents of
  other configurations; text as in E9.
- Conditions, all scored in one run per model: none; matching precedent alone ("one appropriate precedent"); four together
  with the match at positions 1 to 4 (ordering sensitivity); each distractor alone (random-selection baseline = uniform pick
  over the four singles); the model's own pick from the four texts (selector prompt adapted to the Assembly, four orderings),
  shown alone via the single-precedent rows ("single model-selected precedent").
- Prompt: the Security Council template adapted to the Assembly (no veto sentence); same option-likelihood scoring.
- Models: Llama-3.3-70B-FP8, Mistral-Small-24B-2501-FP8, Qwen2.5-32B-GPTQ-Int8, Qwen2.5-7B.
- Metrics: AUROC(N) primary; BC-F1, P(true); selection hit (pick = match, chance 0.25) and selection appropriateness (J1-P
  judge of E12 applied to the Assembly pairs).
- Statistics: query-cluster bootstrap, B = 2000, one-sided; Holm within model over H20a-d.

Hypotheses:
- H20a: AUROC(N), match alone > four together (mean over positions).
- H20b: AUROC(N), match alone > the model's own pick shown alone (mean over orderings).
- H20c: AUROC(N), four together (mean over positions) > none.
- H20d: AUROC(N), four together with the match first > with the match last.
Exploratory: own pick vs random and first-ranked; selection hit vs 0.25 and by position; selection appropriateness.

## Correction note (8 Oct 2026) on the wave-3 amendment above
The amendment heading "Follow-up analyses E7 and E8 (amendment; committed 2026-10-03 after their first outputs)" is superseded
(decision D85): file creation times show that commit cf135a5 (wave 3, 2026-10-03 03:17 UTC) precedes every E7 and E8 output,
and the executed code equals the committed code, so H13 to H15 count as pre-registered. Only E8b (whole-pool stratification
under dilution) remains exploratory. The amendment text is kept unchanged for the record.
