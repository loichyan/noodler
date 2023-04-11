# Changelog

## v0.1.0 (2023-04-11)

### Feat

- **ngram**: add `NGram::similarity()`
- **ngram**: `items_shared_ngrams()` yields more useful information
- **ngram**: add `NGram::fill()`
- **ngram**: add `NGram::search_sorted()`
- **ngram**: add getters
- **ngram**: set default type parameter to String
- **keyed**: impl Keyed for common types
- **lib**: implementa basic NGram functions

### Fix

- **msrv**: support rustc v1.41
- **ngram**: allgrams should subtract samegrams
- **ngram**: pad query string before generating n-grams
- **split**: use char indices to avoid out of char boundary
- **ngram**: split() should not yield value if s.len() < arity

### Refactor

- **ngram**: exec_sorted() returns similarites
- **ngram**: use NGramSearcher to search with custom parameters
- **ngram**: make warp and threshold arguments optional
