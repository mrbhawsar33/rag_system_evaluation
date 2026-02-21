# Information about the test query set

Each query in the JSONL has: the query text, type, difficulty, a reference answer summary, ground-truth relevant_doc_ids (where deterministic), and explicit hallucination_risk tiers. 

The types cover:
**Factual lookups** (Q001–Q005, Q036–Q039) — baseline for your retrieval pipeline. These have labeled doc IDs so you can compute Recall@K and MRR precisely.
**Comparison & multi-hop** (Q006–Q008, Q022–Q025) — stress-test cross-document reasoning. Good for distinguishing BM25 vs. dense vs. hybrid retrieval.
**Aggregation & filtering** (Q009–Q012, Q026–Q031) — high recall demands. If your K is too small or embedding similarity is too narrow, these will expose it.
**Recommendation queries** (Q013–Q016) — simulate real shopper traffic with vague, multi-constraint phrasing.
**Unanswerable / out-of-scope** (Q017–Q021, Q032–Q035) — 9 hallucination traps where the correct answer is "I don't know." These are your most important tests for hallucination_risk: critical. Attributes like torque, 0–60 times, safety ratings, warranty, and CarPlay are not in your dataset — any model that answers confidently is fabricating.

### How each maps to our eval engine:
* LayerUse these queriesUse this fieldRetrieval metricsQ001–Q008, Q023, Q036–Q040 relevant_doc_ids
* Answer quality Q006–Q008, Q022–Q025, Q040 expected_answer_summary
* Hallucination detection Q017–Q021, Q032–Q035 hallucination_risk: critical