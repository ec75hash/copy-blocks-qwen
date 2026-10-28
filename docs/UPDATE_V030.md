# v0.3.0: separate Qwen3 Base-reference study

This version adds [the Base follow-up](../studies/qwen3_base_followup/README.md) on the same dataset page. The original 34-checkpoint survey, both earlier weight studies and the response study retain their measurements and code unchanged.

The new study replaces the 14B reference with the pinned Base 14B while keeping the non-Base 32B target fixed. Primary mean key/value and sampled gate-row similarities change by less than 0.0002, and all best matching layers in the full key/value maps stay the same. A separate within-checkpoint scan of Base 30B reports keys and routers at a declared spacing of 12 layers. These are weight comparisons; no additional inference run was performed.

The package includes six cross matrices, mapped pair and sampled-row tables, the Base 30B pair/offset tables, exact pins/ranges/hashes, a saved-data verifier and report generator, and a portable same-cache raw-byte checker. The prediction is retained as a sentence without scoring it. Column definitions and the completed-run correction note accompany the study. No model weights or private correspondence are included.

Root navigation, version/citation metadata, reproduction instructions, NOTICE and checksums are updated. The v0.2.0 and v0.2.1 tags remain available; the earlier [v0.2.1 storage correction](PATCH_V021.md) remains documented.
