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
        format!("{}{s}{}", self.padding, self.padding)
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
}

impl<T: Keyed> NGram<T> {
    pub fn fill<I: IntoIterator<Item = T>>(mut self, iter: I) -> Self {
        self.extend(iter);
        self
    }

    pub fn contains(&self, item: &T) -> bool {
        self.keys.contains(item.key())
    }

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

    fn raw_items_sharing_ngrams<'a>(
        &'a self,
        query: &str,
    ) -> impl 'a + Iterator<Item = (&'a NGramItem<T>, usize)> {
        self.ngrams(query)
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
            .map(|(id, count)| (&self.items[id], count))
    }

    /// Iterates over items that share n-grams with the query string. Yields item
    /// and the number of shared n-grams.
    pub fn items_sharing_ngrams<'a>(
        &'a self,
        query: &str,
    ) -> impl 'a + Iterator<Item = (&'a T, usize)> {
        self.raw_items_sharing_ngrams(&self.pad(query))
            .map(|(item, count)| (&item.item, count))
    }

    /// Iterates over items sharing n-grams with the query string and their similarities.
    pub fn item_similarities<'a>(
        &'a self,
        query: &str,
        warp: Option<f32>,
    ) -> impl 'a + Iterator<Item = (&'a T, f32)> {
        let warp = warp.unwrap_or(self.warp);
        let query = self.pad(query);
        let query_len = length(&query);
        self.raw_items_sharing_ngrams(&query)
            .map(move |(item, samegrams)| {
                let allgrams = query_len + item.padded_len + 2 - (2 * self.arity) - samegrams;
                let similarity = similarity(samegrams, allgrams, warp);
                (&item.item, similarity)
            })
    }

    pub fn searcher<'i, 'a>(&'i self, query: &'a str) -> NGramSearcher<'i, 'a, T> {
        NGramSearcher {
            ngram: self,
            query,
            threshold: None,
            warp: None,
        }
    }

    pub fn search<'a>(&'a self, query: &str) -> impl 'a + Iterator<Item = (&'a T, f32)> {
        self.searcher(query).exec()
    }

    pub fn search_sorted(&self, query: &str) -> impl '_ + Iterator<Item = &'_ T> {
        self.searcher(query).exec_sorted()
    }
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

    pub fn exec_sorted(self) -> impl 'i + Iterator<Item = &'i T> {
        let mut matches = self.exec().collect::<Vec<_>>();
        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        matches.into_iter().map(|(item, _)| item)
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
            panic!("pad_len out of range < arity = {arity}");
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

fn similarity(samegrams: usize, allgrams: usize, warp: f32) -> f32 {
    let samegrams = samegrams as f32;
    let allgrams = allgrams as f32;
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

    #[test]
    fn unicode_length() {
        assert_eq!(length("哈哈哈"), 3);
    }

    #[test]
    fn ngram_similarity() {
        assert_eq!(similarity(5, 10, 1.0), 0.5);
        assert_eq!(similarity(5, 10, 2.0), 0.75);
        assert_eq!(similarity(5, 10, 3.0), 0.875);
        assert_eq!(similarity(2, 4, 2.0), 0.75);
        assert_eq!(similarity(3, 4, 1.0), 0.75);
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
            &[
                ("a", vec![record(0, 1)]),
                ("b", vec![record(0, 2), record(1, 1)]),
                ("c", vec![record(0, 1), record(1, 2)]),
                ("d", vec![record(1, 1)]),
            ]
            .map(|(k, v)| (k.to_owned(), v))
            .into_iter()
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
            [("abc", 2), ("bcd", 2), ("cda", 1), ("dab", 1)]
                .into_iter()
                .collect::<HashMap<_, _>>()
        );
    }

    #[test]
    fn ngram_items_sharing_ngrams() {
        let ngram = NGramStr::default().fill(["abcde", "cde", "bcdef", "fgh"]);
        assert_eq!(
            ngram
                .items_sharing_ngrams("abcdefg")
                .map(|(s, t)| (*s, t))
                .collect::<HashMap<_, _>>(),
            [("abcde", 5), ("cde", 1), ("bcdef", 3)]
                .into_iter()
                .collect::<HashMap<_, _>>()
        );
    }

    #[test]
    fn ngram_items_sharing_ngrams_min_count() {
        let ngram = NGramStr::builder().arity(1).build().fill(["aaa", "bbb"]);
        assert_eq!(
            ngram
                .items_sharing_ngrams("aaaaab")
                .map(|(s, t)| (*s, t))
                .collect::<HashMap<_, _>>(),
            [("aaa", 3), ("bbb", 1)]
                .into_iter()
                .collect::<HashMap<_, _>>()
        );
    }

    #[test]
    fn ngram_similarities() {
        let ngram = NGramStr::default().fill(["abcde", "cdcd", "cde"]);
        assert_eq!(
            ngram
                .item_similarities("cde", None)
                .map(|(s, i)| (*s, i))
                .collect::<HashMap<_, _>>(),
            [
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
            .fill(["abcde", "cdcd", "cde", "cdef"]);
        assert_eq!(
            ngram.search_sorted("cde").copied().collect::<Vec<_>>(),
            vec!["cde", "cdef", "abcde"]
        );
    }
}
