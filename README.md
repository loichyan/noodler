# 🍜 Noodler

> In computer science, "noodler" is used to describe programs that handle text.
> Because algorithms like n-grams are typically used to extract information from
> text, similar to pulling strands of noodles out of a pile of dough, "noodler"
> can be associated with algorithms that extract information from text because
> they can be seen as "processing" programs for text, just as noodle makers
> "produce" noodles from dough.
>
> _ChatGPT_

A port of the [python-ngram](https://github.com/gpoulter/python-ngram) project
that provides fuzzy search using [N-gram](https://en.wikipedia.org/wiki/N-gram).

## ✍️ Example

```rust
use noodler::NGram;

let ngram = NGram::<&str>::builder()
    .arity(2)
    .warp(3.0)
    .threshold(0.75)
    .build()
    // Feed with known words
    .fill(vec!["pie", "animal", "tomato", "seven", "carbon"]);

// Try an unknown/misspelled word, and find a similar match
let word = "tomacco";
let top = ngram.search_sorted(word).next();
if let Some((text, similarity)) = top {
    if similarity > 0.99 {
        println!("✔ {}", text);
    } else {
        println!(
            "❓{} (did you mean {}? [{:.0}% match])",
            word,
            text,
            similarity * 100.0
        );
    }
} else {
    println!("🗙 {}", word);
}
```

## 💭 Inspired by

Please check out these awesome works that helped a lot in the creation of
noodler:

- [python-ngram](https://github.com/gpoulter/python-ngram): Set that supports
  searching by ngram similarity.
- [ngrammatic](https://github.com/compenguy/ngrammatic): A rust crate providing
  fuzzy search/string matching using N-grams.

## ⚖️ License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or
  <http://opensource.org/licenses/MIT>)

at your option.
