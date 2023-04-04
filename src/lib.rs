#![allow(clippy::type_complexity)]

use std::{
    collections::{HashMap, HashSet},
    marker::PhantomData,
};

pub struct NGram<T = String> {
    threshold: f32,
    warp: f32,
    arity: usize,
    padding: String,
    /// all items
    items: Vec<NGramItem<T>>,
    /// existing keys
    keys: HashSet<String>,
    /// n-grams to their occurrence in items
    records: HashMap<String, Vec<Record>>,
}

struct Record {
    item: usize,
    count: usize,
}

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

    /// Pads a string in preparation for splitting into n-grams.
    pub fn pad(&self, s: &str) -> String {
        format!("{}{s}{}", self.padding, self.padding)
    }

    /// Iterates over the n-grams of a string (no padding).
    pub fn split<'a>(&self, s: &'a str) -> impl 'a + Iterator<Item = &'a str> {
        let n = self.arity;
        (0..=s.len().saturating_sub(n)).map(move |i| &s[i..i + n])
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
        self.raw_items_sharing_ngrams(query)
            .map(|(item, count)| (&item.item, count))
    }

    /// Iterates over items sharing n-grams with the query string and their similarities.
    pub fn item_similarities<'a>(
        &'a self,
        query: &str,
        warp: f32,
    ) -> impl 'a + Iterator<Item = (&'a T, f32)> {
        let padded_len = length(&self.pad(query));
        self.raw_items_sharing_ngrams(query)
            .map(move |(item, samegrams)| {
                let allgrams = padded_len + item.padded_len + 2 - (2 * self.arity);
                let similarity = similarity(samegrams, allgrams, warp);
                (&item.item, similarity)
            })
    }

    pub fn search_with<'a>(
        &'a self,
        query: &str,
        threshold: f32,
        warp: f32,
    ) -> impl 'a + Iterator<Item = (&'a T, f32)> {
        self.item_similarities(query, warp)
            .filter(move |&(_, similarity)| similarity > threshold)
    }

    pub fn search<'a>(&'a self, query: &str) -> impl 'a + Iterator<Item = (&'a T, f32)> {
        self.search_with(query, self.threshold, self.warp)
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

#[test]
fn test_length() {
    assert_eq!(length("哈哈哈"), 3);
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

#[test]
fn test_similarity() {
    assert_eq!(similarity(5, 10, 1.0), 0.5);
    assert_eq!(similarity(5, 10, 2.0), 0.75);
    assert_eq!(similarity(5, 10, 3.0), 0.875);
    assert_eq!(similarity(2, 4, 2.0), 0.75);
    assert_eq!(similarity(3, 4, 1.0), 0.75);
}
