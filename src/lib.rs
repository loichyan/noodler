//! A port of the [python-ngram](https://github.com/gpoulter/python-ngram) project
//! that provides fuzzy search using [N-gram](https://en.wikipedia.org/wiki/N-gram).
//!
//! # ✍️ Example
//!
//! ```
//! use noodler::NGram;
//!
//! let ngram = NGram::<&str>::builder()
//!     .arity(2)
//!     .warp(3.0)
//!     .threshold(0.75)
//!     .build()
//!     // Feed with known words
//!     .fill(vec!["pie", "animal", "tomato", "seven", "carbon"]);
//!
//! // Try an unknown/misspelled word, and find a similar match
//! let word = "tomacco";
//! let top = ngram.search_sorted(word).next();
//! if let Some((text, similarity)) = top {
//!     if similarity > 0.99 {
//!         println!("✔ {}", text);
//!     } else {
//!         println!(
//!             "❓{} (did you mean {}? [{:.0}% match])",
//!             word,
//!             text,
//!             similarity * 100.0
//!         );
//!     }
//! } else {
//!     println!("🗙 {}", word);
//! }
//! ```
//!
//! # 💭 Inspired by
//!
//! Please check out these awesome works that helped a lot in the creation of
//! noodler:
//!
//! - [python-ngram](https://github.com/gpoulter/python-ngram): Set that supports
//!   searching by ngram similarity.
//! - [ngrammatic](https://github.com/compenguy/ngrammatic): A rust crate providing
//!   fuzzy search/string matching using N-grams.
//!
//! # 🚩 Minimal supported Rust version
//!
//! All tests passed with `rustc v1.41`, earlier versions may not compile.

#![allow(clippy::type_complexity)]

use std::{
    collections::{HashMap, HashSet},
    marker::PhantomData,
};

#[derive(Debug, PartialEq)]
pub struct NGram<T = String> {
    arity: usize,
    padding: String,
    warp: f32,
    threshold: f32,
    /// all items
    items: Vec<NGramItem<T>>,
    /// existing keys
    keys: HashSet<String>,
    /// n-grams to their occurrence in items
    records: HashMap<String, Vec<Record>>,
}

#[derive(Debug, Eq, PartialEq)]
struct Record {
    item: usize,
    count: usize,
}

#[derive(Debug, Eq, PartialEq)]
struct NGramItem<T> {
    item: T,
    padded_len: usize,
}

impl<T> Default for NGram<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Keyed> Extend<T> for NGram<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        iter.into_iter().for_each(|item| self.insert(item))
    }
}

impl<T> NGram<T> {
    pub fn new() -> Self {
        Self::builder().build()
    }

    pub fn builder() -> NGramBuilder<T> {
        NGramBuilder::default()
    }

    pub fn arity(&self) -> usize {
        self.arity
    }

    pub fn padding(&self) -> &str {
        &self.padding
    }

    pub fn warp(&self) -> f32 {
        self.warp
    }

    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Pads a string in preparation for splitting into n-grams.
    pub fn pad(&self, s: &str) -> String {
        format!("{}{}{}", self.padding, s, self.padding)
    }

    /// Iterates over the n-grams of a string (no padding).
    pub fn split<'a>(&self, s: &'a str) -> impl 'a + Iterator<Item = &'a str> {
        let char_indices = s
            .char_indices()
            .map(|(i, _)| i)
            .chain(std::iter::once(s.len()))
            .collect::<Vec<_>>();
        let n = self.arity;
        // If s.len() < arity, 0..0 doesn't yield any value but 0..=0 does so.
        (0..char_indices.len().saturating_sub(n))
            .map(move |i| &s[char_indices[i]..char_indices[i + n]])
    }

    /// Generates n-grams with their occurrences (no padding).
    pub fn ngrams<'a>(&self, s: &'a str) -> impl 'a + Iterator<Item = (&'a str, usize)> {
        self.split(s)
            .fold(HashMap::<&str, usize>::default(), |mut ngrams, ngram| {
                // Increment number of times the n-gram appears
                *ngrams.entry(ngram).or_default() += 1;
                ngrams
            })
            .into_iter()
    }

    /// Calculates N-gram similarity.
    pub fn similarity(&self, shared_ngrams: usize, all_ngrams: usize, warp: Option<f32>) -> f32 {
        similarity(
            shared_ngrams,
            all_ngrams - shared_ngrams,
            warp.unwrap_or(self.warp),
        )
    }
}

impl<T: Keyed> NGram<T> {
    pub fn fill<I: IntoIterator<Item = T>>(mut self, iter: I) -> Self {
        self.extend(iter);
        self
    }

    /// Returns `true` if the [`NGram`] contains an item.
    pub fn contains(&self, item: &T) -> bool {
        self.keys.contains(item.key())
    }

    /// Inserts an item if the [`NGram`] doesn't contain it.
    pub fn insert(&mut self, item: T) {
        if self.contains(&item) {
            return;
        }
        // Add key and create a new item
        self.keys.insert(item.key().to_owned());
        // Record item to its n-grams
        let id = self.items.len();
        let padded = self.pad(item.key());
        self.ngrams(&padded).for_each(|(ngram, count)| {
            self.records
                .entry(ngram.to_owned())
                .or_default()
                .push(Record { item: id, count });
        });
        // Add item
        self.items.push(NGramItem {
            item,
            padded_len: length(&padded),
        });
    }

    /// Iterates over items that share n-grams with the query string. Yields item
    /// and the number of shared n-grams, see [`SharedNGrams`] for more details.
    pub fn items_sharing_ngrams<'a>(
        &'a self,
        query: &str,
    ) -> impl 'a + Iterator<Item = SharedNGrams<'a, T>> {
        let query = self.pad(query);
        let query_ngrams = length(&query) + 1 - self.arity;
        self.ngrams(&query)
            // Sum shared n-grams of each item
            .fold(
                HashMap::<usize, usize>::default(),
                |mut shared, (ngram, count)| {
                    if let Some(items) = self.records.get(ngram) {
                        items.iter().for_each(|record| {
                            *shared.entry(record.item).or_default() += count.min(record.count);
                        });
                    }
                    shared
                },
            )
            .into_iter()
            .map(move |(id, shared_ngrams)| {
                let item = &self.items[id];
                SharedNGrams {
                    item: &item.item,
                    item_ngrams: item.padded_len + 1 - self.arity,
                    query_ngrams,
                    shared_ngrams,
                }
            })
    }

    /// Iterates over items sharing n-grams with the query string and their similarities.
    pub fn item_similarities<'a>(
        &'a self,
        query: &str,
        warp: Option<f32>,
    ) -> impl 'a + Iterator<Item = (&'a T, f32)> {
        self.items_sharing_ngrams(query).map(
            move |SharedNGrams {
                      item,
                      item_ngrams,
                      query_ngrams,
                      shared_ngrams,
                  }| {
                (
                    item,
                    self.similarity(shared_ngrams, item_ngrams + query_ngrams, warp),
                )
            },
        )
    }

    pub fn searcher<'i, 'a>(&'i self, query: &'a str) -> NGramSearcher<'i, 'a, T> {
        NGramSearcher {
            ngram: self,
            query,
            threshold: None,
            warp: None,
        }
    }

    /// A shorthand of [`NGramSearcher::exec()`].
    pub fn search<'a>(&'a self, query: &str) -> impl 'a + Iterator<Item = (&'a T, f32)> {
        self.searcher(query).exec()
    }

    /// A shorthand of [`NGramSearcher::exec_sorted()`].
    pub fn search_sorted(&self, query: &str) -> impl '_ + Iterator<Item = (&'_ T, f32)> {
        self.searcher(query).exec_sorted()
    }
}

pub struct SharedNGrams<'a, T> {
    pub item: &'a T,
    /// Number of item's n-grams.
    pub item_ngrams: usize,
    /// Number of query string's n-grams.
    pub query_ngrams: usize,
    /// Number of shared n-grams between item and query string.
    pub shared_ngrams: usize,
}

pub struct NGramSearcher<'i, 'a, T> {
    ngram: &'i NGram<T>,
    query: &'a str,
    threshold: Option<f32>,
    warp: Option<f32>,
}

impl<'i, T: Keyed> NGramSearcher<'i, '_, T> {
    pub fn warp(self, warp: f32) -> Self {
        Self {
            warp: Some(warp),
            ..self
        }
    }

    pub fn threshold(self, threshold: f32) -> Self {
        Self {
            threshold: Some(threshold),
            ..self
        }
    }

    /// Consumes the searcher and returns all items with matched similarities.
    pub fn exec(self) -> impl 'i + Iterator<Item = (&'i T, f32)> {
        let Self {
            ngram,
            query,
            threshold,
            warp,
        } = self;
        let threshold = threshold.unwrap_or(ngram.threshold);
        ngram
            .item_similarities(query, warp)
            .filter(move |&(_, similarity)| similarity > threshold)
    }

    /// Searches items and sorts the results by their similarities.
    pub fn exec_sorted(self) -> impl 'i + Iterator<Item = (&'i T, f32)> {
        let mut matches = self.exec().collect::<Vec<_>>();
        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        matches.into_iter()
    }
}

pub struct NGramBuilder<T> {
    threshold: Option<f32>,
    warp: Option<f32>,
    arity: Option<usize>,
    pad_len: Option<usize>,
    pad_char: Option<char>,
    _marker: PhantomData<T>,
}

impl<T> Default for NGramBuilder<T> {
    fn default() -> Self {
        Self {
            threshold: Default::default(),
            warp: Default::default(),
            arity: Default::default(),
            pad_len: Default::default(),
            pad_char: Default::default(),
            _marker: PhantomData,
        }
    }
}

impl<T> NGramBuilder<T> {
    /// Minimum similarity for a string to be considered a match.
    pub fn threshold(self, threshold: f32) -> Self {
        Self {
            threshold: Some(threshold),
            ..self
        }
    }

    /// Use warp greater than 1.0 to increase the similarity of shorter string pairs.
    pub fn warp(self, warp: f32) -> Self {
        Self {
            warp: Some(warp),
            ..self
        }
    }

    /// Uumber of characters per n-gram.
    ///
    /// # Panics
    ///
    /// Causes [`Self::build()`] to panic when less than _1_.
    pub fn arity(self, arity: usize) -> Self {
        Self {
            arity: Some(arity),
            ..self
        }
    }

    /// How many characters padding to add (defaults to _arity - 1_).
    ///
    /// # Panics
    ///
    /// Causes [`Self::build()`] to panic when greater than _arity_.
    pub fn pad_len(self, pad_len: usize) -> Self {
        Self {
            pad_len: Some(pad_len),
            ..self
        }
    }

    /// Character to use for padding. Default is `'\u{A0}'`.
    pub fn pad_char(self, pad_char: char) -> Self {
        Self {
            pad_char: Some(pad_char),
            ..self
        }
    }

    /// # Panics
    ///
    /// Panics when some parameter is illegal, see the documentation of each parameter
    /// for more details.
    pub fn build(self) -> NGram<T> {
        let arity = self.arity.unwrap_or(3);
        if arity < 1 {
            panic!("arity out of range >= 1");
        }
        let pad_len = self.pad_len.unwrap_or(arity - 1);
        if pad_len >= arity {
            panic!("pad_len out of range < arity = {}", arity);
        }
        let padding = std::iter::repeat(self.pad_char.unwrap_or('\u{A0}'))
            .take(pad_len)
            .collect();
        NGram {
            threshold: self.threshold.unwrap_or(0.8),
            warp: self.warp.unwrap_or(1.0),
            arity,
            padding,
            items: Default::default(),
            keys: Default::default(),
            records: Default::default(),
        }
    }
}

pub trait Keyed {
    /// Returns a string to identify this item and generate n-grams.
    ///
    /// **Note**: This function should be cheap and deterministic.
    fn key(&self) -> &str;
}

macro_rules! impl_for_ptr_type {
    ($ty:ident) => {
        impl<T: Keyed> Keyed for $ty<T> {
            fn key(&self) -> &str {
                T::key(self)
            }
        }
    };
    ($ty:ident<'_>) => {
        impl<T: Keyed> Keyed for $ty<'_, T> {
            fn key(&self) -> &str {
                T::key(self)
            }
        }
    };
}

const _: () = {
    use std::{
        borrow::Cow,
        cell::{Ref, RefMut},
        rc::Rc,
        sync::Arc,
    };

    impl Keyed for String {
        fn key(&self) -> &str {
            self
        }
    }

    impl Keyed for &str {
        fn key(&self) -> &str {
            self
        }
    }

    impl Keyed for Cow<'_, str> {
        fn key(&self) -> &str {
            self
        }
    }

    impl_for_ptr_type!(Box);
    impl_for_ptr_type!(Rc);
    impl_for_ptr_type!(Arc);
    impl_for_ptr_type!(Ref<'_>);
    impl_for_ptr_type!(RefMut<'_>);

    impl<T: Keyed> Keyed for (T,) {
        fn key(&self) -> &str {
            self.0.key()
        }
    }

    impl<T1: Keyed, T2> Keyed for (T1, T2) {
        fn key(&self) -> &str {
            self.0.key()
        }
    }
};

fn length(s: &str) -> usize {
    s.chars().count()
}

fn similarity(shared_ngrams: usize, union_ngrams: usize, warp: f32) -> f32 {
    let samegrams = shared_ngrams as f32;
    let allgrams = union_ngrams as f32;
    if (warp - 1.0).abs() < 1e-9 {
        samegrams / allgrams
    } else {
        let diffgrams = allgrams - samegrams;
        (allgrams.powf(warp) - diffgrams.powf(warp)) / (allgrams.powf(warp))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type NGramStr = NGram<&'static str>;

    fn float_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < std::f32::EPSILON
    }

    #[test]
    fn ascii_length() {
        assert_eq!(length("abc"), 3);
        assert_eq!(length("123"), 3);
    }

    #[test]
    fn unicode_length() {
        assert_eq!(length("一二三"), 3);
        assert_eq!(length("🥳🥳🥳"), 3);
    }

    #[test]
    fn raw_similarity() {
        assert!(float_eq(similarity(5, 10, 1.0), 0.5));
        assert!(float_eq(similarity(5, 10, 2.0), 0.75));
        assert!(float_eq(similarity(5, 10, 3.0), 0.875));
        assert!(float_eq(similarity(2, 4, 2.0), 0.75));
        assert!(float_eq(similarity(3, 4, 1.0), 0.75));
    }

    #[test]
    fn ngram_similarity() {
        let ngram = NGramStr::new();
        assert!(float_eq(ngram.similarity(5, 15, Some(1.0)), 0.5));
        assert!(float_eq(ngram.similarity(5, 15, Some(2.0)), 0.75));
    }

    #[test]
    fn builder() {
        assert_eq!(
            NGramStr::builder()
                .arity(5)
                .pad_len(3)
                .pad_char('$')
                .threshold(0.75)
                .warp(2.0)
                .build(),
            NGramStr {
                threshold: 0.75,
                warp: 2.0,
                arity: 5,
                padding: "$$$".to_owned(),
                ..Default::default()
            }
        );
    }

    #[test]
    fn default_padding() {
        let ngram = NGramStr::default();
        assert_eq!(ngram.pad("abc"), "  abc  ");
    }

    #[test]
    fn set_pad_char() {
        let ngram = NGramStr::builder().pad_char('$').build();
        assert_eq!(ngram.pad("abc"), "$$abc$$");
    }

    #[test]
    fn set_pad_len() {
        let ngram = NGramStr::builder().pad_len(1).build();
        assert_eq!(ngram.pad("abc"), " abc ");
    }

    #[test]
    #[should_panic = "pad_len out of range < arity = 3"]
    fn panic_if_pad_len_ge_arity() {
        NGramStr::builder().pad_len(3).build();
    }

    #[test]
    fn set_arity() {
        let ngram = NGramStr::builder().arity(2).build();
        assert_eq!(
            ngram.split("abcdef").collect::<Vec<_>>(),
            vec!["ab", "bc", "cd", "de", "ef"]
        );
    }

    #[test]
    #[should_panic = "arity out of range >= 1"]
    fn panic_if_arity_lt_1() {
        NGramStr::builder().arity(0).build();
    }

    fn item<T>(item: T, padded_len: usize) -> NGramItem<T> {
        NGramItem { item, padded_len }
    }

    fn record(item: usize, count: usize) -> Record {
        Record { item, count }
    }

    #[test]
    fn ngram_insert() {
        let mut ngram = NGramStr::builder().arity(1).build();
        ngram.insert("abbc");
        assert_eq!(&ngram.items, &vec![item("abbc", 4)]);
        ngram.insert("abbc"); // should be ignored
        assert_eq!(ngram.keys.len(), 1);
        ngram.insert("bccd");
        assert_eq!(ngram.keys.len(), 2);
        assert_eq!(
            &ngram.records,
            &vec![
                ("a", vec![record(0, 1)]),
                ("b", vec![record(0, 2), record(1, 1)]),
                ("c", vec![record(0, 1), record(1, 2)]),
                ("d", vec![record(1, 1)]),
            ]
            .into_iter()
            .map(|(k, v)| (k.to_owned(), v))
            .collect::<HashMap<_, _>>()
        );
    }

    #[test]
    fn ngram_split() {
        let ngram = NGramStr::default();
        assert_eq!(
            ngram.split("abcdef").collect::<Vec<_>>(),
            vec!["abc", "bcd", "cde", "def"]
        );
    }

    #[test]
    fn ngram_split_unicode_chars() {
        let ngram = NGramStr::default();
        assert_eq!(
            ngram.split("一二三四五六").collect::<Vec<_>>(),
            vec!["一二三", "二三四", "三四五", "四五六"]
        );
    }

    #[test]
    fn ngram_split_no_yield_if_len_lt_arity() {
        let ngram = NGramStr::default();
        assert_eq!(ngram.split("a").collect::<Vec<_>>(), Vec::<&str>::default());
    }

    #[test]
    fn ngram_ngrams() {
        let ngram = NGramStr::default();
        assert_eq!(
            ngram.ngrams("abcdabcd").collect::<HashMap<_, _>>(),
            vec![("abc", 2), ("bcd", 2), ("cda", 1), ("dab", 1)]
                .into_iter()
                .collect::<HashMap<_, _>>()
        );
    }

    #[test]
    fn ngram_items_sharing_ngrams() {
        let ngram = NGramStr::default().fill(vec!["abcde", "cde", "bcdef", "fgh"]);
        assert_eq!(
            ngram
                .items_sharing_ngrams("abcdefg")
                .map(|t| (*t.item, t.shared_ngrams))
                .collect::<HashMap<_, _>>(),
            vec![("abcde", 5), ("cde", 1), ("bcdef", 3)]
                .into_iter()
                .collect::<HashMap<_, _>>()
        );
    }

    #[test]
    fn ngram_items_sharing_ngrams_min_count() {
        let ngram = NGramStr::builder()
            .arity(1)
            .build()
            .fill(vec!["aaa", "bbb"]);
        assert_eq!(
            ngram
                .items_sharing_ngrams("aaaaab")
                .map(|t| (*t.item, t.shared_ngrams))
                .collect::<HashMap<_, _>>(),
            vec![("aaa", 3), ("bbb", 1)]
                .into_iter()
                .collect::<HashMap<_, _>>()
        );
    }

    #[test]
    fn ngram_similarities() {
        let ngram = NGramStr::default().fill(vec!["abcde", "cdcd", "cde"]);
        assert_eq!(
            ngram
                .item_similarities("cde", None)
                .map(|(s, i)| (*s, i))
                .collect::<HashMap<_, _>>(),
            vec![
                ("abcde", similarity(3, 9, 1.0)),
                ("cdcd", similarity(2, 9, 1.0)),
                ("cde", similarity(5, 5, 1.0)),
            ]
            .into_iter()
            .collect::<HashMap<_, _>>()
        );
    }

    #[test]
    fn ngram_search() {
        let ngram = NGramStr::builder()
            .threshold(0.5)
            .warp(2.0)
            .build()
            .fill(vec!["abcde", "cdcd", "cde", "cdef"]);
        assert_eq!(
            ngram
                .search_sorted("cde")
                .map(|(item, _)| *item)
                .collect::<Vec<_>>(),
            vec!["cde", "cdef", "abcde"]
        );
    }

    #[test]
    fn ngram_search_with_parameters() {
        let ngram = NGramStr::builder()
            .threshold(1.0)
            .warp(1.0)
            .build()
            .fill(vec!["abcde", "cdcd", "cde", "cdef"]);
        assert_eq!(
            ngram
                .searcher("cde")
                .warp(2.0)
                .threshold(0.5)
                .exec_sorted()
                .map(|(item, _)| *item)
                .collect::<Vec<_>>(),
            vec!["cde", "cdef", "abcde"]
        );
    }
}
