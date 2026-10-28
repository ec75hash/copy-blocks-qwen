# Short glossary

The study pages give the specific comparisons and sampling choices. Column guides:
[original survey](SURVEY_COLUMNS.md), [Qwen2.5 weights](../studies/qwen25_weights/COLUMNS.md),
[Qwen3 weights](../studies/qwen3_weights/COLUMNS.md), and
[Qwen3 responses](../studies/qwen3_response/COLUMNS.md).

| Term | Meaning in this release |
|---|---|
| Checkpoint and pinned revision | A checkpoint is a saved set of model parameters. A pinned revision fixes a particular commit of its repository, so later repository changes do not change the comparison. |
| Q8 GGUF and BF16 | The response experiment uses Qwen's published Q8_0, an 8-bit quantized model in GGUF format. The weight scans read BF16 (bfloat16) parameter values from separately pinned safetensors files. These are different artifacts. |
| Layer, block, neuron and MLP width | Layer indices are zero-based original decoder indices. A decoder block includes attention and a multilayer perceptron (MLP). Here an MLP neuron is an indexed intermediate channel; MLP width is the number of these channels. “Block” can also mean a consecutive range of decoder layers; displayed ranges make that use explicit. |
| Cosine, flattened cosine and row-mean cosine | Cosine compares vector directions: +1 is the same direction, 0 is orthogonal, and −1 is opposite. It does not compare magnitudes or establish equal weights. Flattened cosine treats a whole tensor as one vector; row-mean cosine averages separate aligned-neuron cosines. For the down projection, a neuron's weights are a column of the stored tensor. |
| K, offset and block spacing | An offset is the difference between layer indices. K is the declared spacing between corresponding blocks. In the response tables, `top_k` instead means the number of text windows selected for a list. |
| Corresponding, within-block and off-block | Within a checkpoint, corresponding pairs align positions in declared block ranges. Within-block means summarize those pairs; off-block means use the other pairs at the same offset. For Qwen3-32B at offset 24, (8,32) is corresponding and (0,24) is off-block. Cross-checkpoint studies instead use an explicit map between two models. |
| Eligible offset, spike and reporting rule | The original survey ranks offsets with at least six tensor pairs. A neighboring-offset spike is an offset's mean minus the mean of its two eligible neighbors. The reporting rule also specifies fallback cases and thresholds; see [METHODS](../METHODS.md#offsets-regions-and-periods). “Confirmed” means passing the declared-pattern rule; “no block” means no detected pattern under the stated rule. Neither label establishes a training history. |
| Gram matrix | A table of every vector’s dot product with every other vector. For normalized vectors, entries are cosines and the diagonal should be one; the scanner checks that diagonal to detect numerical errors. |
| DeltaNet short convolution | A small convolutional weight tensor inside the models’ linear-attention blocks, compared as another parameter family. |
| Gate and router | In a dense MLP, the gate projection supplies the input to SiLU; the measured final scalar is `SiLU(gate) * value`. An MoE router (`mlp.gate.weight`) selects experts and is a different parameter family. A reporting rule is a test used to label results, not either model component. |
| Twin A/B and neuron segments A/B/C | Qwen2.5 twin A/B name 32B layer ranges 16–31 and 32–47, both compared with 14B 16–31. In Qwen3, A/B/C instead name neuron-index ranges 0–8191, 8192–17407 and 17408–25599. The letters label different things in the two studies; they do not specify construction order. |
| Reference residual | The vector difference between a target tensor and a selected reference tensor. The Qwen2.5 study uses the 14B tensor as reference. This difference is not assumed to be a historical training update; see the [formulas](../studies/qwen25_weights/COLUMNS.md#reference-residuals). |
