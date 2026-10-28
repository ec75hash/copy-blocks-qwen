# WikiText / Wikipedia-derived text

The released windows in `../data/capture/passages.jsonl`, token IDs in `tokens.npy`,
and quoted text in the example files derive from WikiText-103 raw train, distributed
by Salesforce/Hugging Face and compiled from Wikipedia articles.

Source: https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3
File: `wikitext-103-raw-v1/train-00000-of-00002.parquet`.
The complete pinned upstream card is preserved in [WIKITEXT_UPSTREAM_README.md](WIKITEXT_UPSTREAM_README.md).
It attributes WikiText to Stephen Merity, Caiming Xiong, James Bradbury and Richard Socher,
*Pointer Sentinel Mixture Models* (2016), https://arxiv.org/abs/1609.07843.
Wikipedia article text is authored by Wikipedia contributors; article histories are
available through https://en.wikipedia.org/. Source row ranges and original parsed
labels accompany each window, with false-header cases explicitly documented.

The pinned upstream card's frontmatter lists `cc-by-sa-3.0` and `gfdl`; its licensing
paragraph instead links CC BY-SA 4.0. Both declarations are preserved, rather than
silently representing the source as uniformly CC BY 4.0. These upstream text terms
are distinct from this release's original derived measurements and writing.

CC BY-SA 3.0: https://creativecommons.org/licenses/by-sa/3.0/
CC BY-SA 4.0: https://creativecommons.org/licenses/by-sa/4.0/
GFDL: https://www.gnu.org/licenses/fdl-1.3.html

Changes to upstream text: the historical collector grouped source rows, omitted
headings and some equation rows, joined text, and selected fixed 256-token windows.
It sometimes split articles at equation lines. The selected raw source text, exact
token IDs and row provenance are retained; marked example contexts add `[[...]]`
around the measured peak token. No model-generated continuation is included in the corpus.
