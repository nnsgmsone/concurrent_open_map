//! Golang like sync.Map

use parking_lot::RwLock;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::mem;
use std::sync::Arc;

/// A concurrent map, Map is like sync.Map in Golang.
///
/// - K must implement `Eq + PartialEq + Hash`.
/// - V must implement `Eq + PartialEq + Clone`.
pub struct Map<K, V> {
    shards: Vec<Arc<RwLock<Shard<K, V>>>>,
}

struct Shard<K, V> {
    count: u32,
    shift: u32,
    size: u32,
    max_dist: u32,
    buckets: Vec<Option<Bucket<K, V>>>,
}

struct Bucket<K, V> {
    h: u64,
    k: K,
    v: V,
    dist: u32,
}

const BITS8_LEN: [u8; 256] = [
    0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
];

fn bits32_len(x: u32) -> u32 {
    let mut r = 0;
    let mut x = x;

    if x >= 0x10000 {
        x >>= 16;
        r = 16;
    }

    if x >= 0x100 {
        x >>= 8;
        r += 8;
    }

    r + BITS8_LEN[x as usize] as u32
}

fn max_dist_for_size(size: u32) -> u32 {
    bits32_len(size).max(4)
}

fn calculate_hash<T: Hash>(t: &T) -> u64 {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    s.finish()
}

fn shr(x: u64, s: u32) -> u32 {
    if s >= 64 {
        0
    } else {
        (x >> s) as u32
    }
}

impl<K, V> PartialEq for Bucket<K, V>
where
    K: Eq + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.h == other.h && self.k == other.k
    }
}

impl<K, V> Eq for Bucket<K, V> where K: Eq + PartialEq {}

impl<K, V> Shard<K, V>
where
    K: Eq + PartialEq + Hash,
    V: Clone,
{
    fn new() -> Self {
        Shard {
            count: 0,
            shift: 0,
            size: 0,
            max_dist: 0,
            buckets: Vec::new(),
        }
    }
    fn rehash(&mut self, size: u32) {
        let old_buckets = mem::take(&mut self.buckets);
        self.count = 0;
        self.size = size;
        self.shift = 64 - bits32_len(size - 1);
        self.max_dist = max_dist_for_size(size);
        let bucket_num = size + self.max_dist;
        self.buckets = (0..bucket_num).map(|_| None).collect();
        for b in old_buckets.into_iter().flatten() {
            self.set(b.h, b.k, b.v);
        }
    }
    fn find(&self, h: u64, k: &K) -> Option<u32> {
        let mut dist = 0;
        let mut i = shr(h, self.shift);
        loop {
            let b = &self.buckets[i as usize];
            match b {
                Some(b) if b.h == h && b.k == *k => return Some(i),
                Some(b) if b.dist < dist => return None,
                None => return None,
                _ => {}
            }
            dist += 1;
            i += 1;
        }
    }
    fn del(&mut self, i: u32) -> Option<V> {
        let mut j = i + 1;
        let mut b_idx = i as usize;
        let r = self.buckets[b_idx].take().map(|v| v.v);
        loop {
            let t_idx = j as usize;
            if let Some(t) = &mut self.buckets[t_idx] {
                if t.dist == 0 {
                    self.count -= 1;
                    self.buckets[b_idx] = None;
                    return r;
                }
                t.dist -= 1;
                self.buckets[b_idx] = self.buckets[t_idx].take();
                self.buckets[t_idx] = None;
            } else {
                self.count -= 1;
                self.buckets[b_idx] = None;
                return r;
            }
            b_idx = t_idx;
            j += 1;
        }
    }
    fn range<F>(&self, mut f: F) -> bool
    where
        F: FnMut(&K, &V) -> bool,
    {
        self.buckets
            .iter()
            .filter_map(|b| b.as_ref())
            .all(|b| f(&b.k, &b.v))
    }
    // set sets the value for the given key.
    fn set(&mut self, h: u64, k: K, v: V) {
        let mut maybe_existing = true;
        let mut n = Bucket { h, k, v, dist: 0 };
        let mut i = shr(h, self.shift);
        loop {
            let b = &mut self.buckets[i as usize];
            if let Some(b) = b {
                if maybe_existing && b.h == h && b.k == n.k {
                    // update existing
                    b.h = n.h;
                    b.v = n.v;
                    return;
                }
                if b.dist < n.dist {
                    std::mem::swap(b, &mut n);
                    maybe_existing = false;
                }
                // far away bucket, swap and move on
                n.dist += 1;
                // rehash if the bucket is too far away
                if n.dist == self.max_dist {
                    self.rehash(self.size * 2);
                    i = shr(n.h, self.shift).wrapping_sub(1);
                    n.dist = 0;
                    maybe_existing = false;
                }
                i = i.wrapping_add(1);
            } else {
                self.count += 1;
                *b = Some(n);
                return;
            }
        }
    }
}

impl<K, V> Map<K, V>
where
    K: Eq + PartialEq + Hash,
    V: Eq + PartialEq + Clone,
{
    /// Create a new Map.
    pub fn new() -> Self {
        // 4 * num_cpus is a good number for most cases.
        let num_cpus = num_cpus::get() * 4;
        let shards = (0..num_cpus)
            .map(|_| Arc::new(RwLock::new(Shard::new())))
            .collect::<Vec<_>>();
        for shard in &shards {
            shard.write().rehash(1);
        }
        Map { shards }
    }
    ///  compare_and_delete deletes the key-value pair for the given key if the value is equal to the given value.
    ///  returns true if the kv was already in. returns false if the kv was not in.
    pub fn compare_and_delete(&self, k: K, v: V) -> bool {
        let h = calculate_hash(&k);
        let shard = &self.shards[h as usize % self.shards.len()];
        let mut guard = shard.write();
        let i = match guard.find(h, &k) {
            Some(i) => i,
            None => return false,
        };
        let b = &guard.buckets[i as usize];
        match b {
            Some(b) if b.v == v => {
                guard.del(i);
                true
            }
            _ => false,
        }
    }
    /// compare_and_swap sets the value for the given key if the value is equal to the given old value.
    /// returns true if the kv was already in. returns false if the kv was not in.
    pub fn compare_and_swap(&self, k: K, old_v: V, new_v: V) -> bool {
        let h = calculate_hash(&k);
        let shard = &self.shards[h as usize % self.shards.len()];
        let mut guard = shard.write();
        let i = match guard.find(h, &k) {
            Some(i) => i,
            None => return false,
        };
        let b = &mut guard.buckets[i as usize];
        if let Some(b) = b {
            if b.v == old_v {
                b.v = new_v;
                return true;
            }
        }
        false
    }
    /// store sets the value for the given key.
    pub fn store(&self, k: K, v: V) {
        let h = calculate_hash(&k);
        let shard = &self.shards[h as usize % self.shards.len()];
        shard.write().set(h, k, v)
    }
    /// load returns the value stored in the map for a key, or none if no value is present.
    pub fn load(&self, k: K) -> Option<V> {
        let h = calculate_hash(&k);
        let shard = &self.shards[h as usize % self.shards.len()];
        let guard = shard.read();
        let i = match guard.find(h, &k) {
            Some(i) => i,
            None => return None,
        };
        let b = &guard.buckets[i as usize];
        b.as_ref().map(|b| b.v.clone())
    }
    /// without clone: a little faster
    /// return true if the key is in the map, false otherwise.
    pub fn get<F>(&self, k: K, mut f: F) -> bool
    where
        F: FnMut(&V),
    {
        let h = calculate_hash(&k);
        let shard = &self.shards[h as usize % self.shards.len()];
        let guard = shard.read();
        let i = match guard.find(h, &k) {
            Some(i) => i,
            None => return false,
        };
        let b = &guard.buckets[i as usize];
        match b {
            Some(b) => {
                f(&b.v);
                true
            }
            None => false,
        }
    }
    /// delete deletes the key-value pair for the given key.
    pub fn delete(&self, k: K) {
        let h = calculate_hash(&k);
        let shard = &self.shards[h as usize % self.shards.len()];
        let mut guard = shard.write();
        let i = match guard.find(h, &k) {
            Some(i) => i,
            None => return,
        };
        guard.del(i);
    }
    /// load_and_delete deletes the key-value pair for the given key and returns the value
    pub fn load_and_delete(&self, k: K) -> Option<V> {
        let h = calculate_hash(&k);
        let shard = &self.shards[h as usize % self.shards.len()];
        let mut guard = shard.write();
        let i = match guard.find(h, &k) {
            Some(i) => i,
            None => return None,
        };
        {
            println!("load_and_delete: i = {}", i);
        }
        guard.del(i)
    }
    /// load_or_store returns the existing value for the key if present.
    /// otherwise, it stores and returns the given value.
    pub fn load_or_store(&self, k: K, v: V) -> V {
        let h = calculate_hash(&k);
        let shard = &self.shards[h as usize % self.shards.len()];
        let mut guard = shard.write();
        let i = match guard.find(h, &k) {
            Some(i) => i,
            None => {
                guard.set(h, k, v.clone());
                return v;
            }
        };
        let b = &guard.buckets[i as usize];
        if let Some(b) = b {
            return b.v.clone();
        }
        // should not reach here
        panic!("load_or_store: bucket is none");
    }
    /// swap swaps the value for a key and returns the old value
    pub fn swap(&self, k: K, v: V) -> Option<V> {
        let h = calculate_hash(&k);
        let shard = &self.shards[h as usize % self.shards.len()];
        let mut guard = shard.write();
        let i = match guard.find(h, &k) {
            Some(i) => i,
            None => {
                guard.set(h, k, v);
                return None;
            }
        };
        let b = &mut guard.buckets[i as usize];
        match b {
            Some(b) => {
                let old_v = mem::replace(&mut b.v, v);
                Some(old_v)
            }
            None => None,
        }
    }
    /// range calls f sequentially for each key and value present in the map.
    /// if f returns false, range stops the iteration.
    pub fn range<F>(&self, mut f: F)
    where
        F: FnMut(&K, &V) -> bool,
    {
        for shard in &self.shards {
            let guard = shard.read();
            if !guard.range(&mut f) {
                return;
            }
        }
    }
}

impl<K, V> Default for Map<K, V>
where
    K: Eq + PartialEq + Hash,
    V: Eq + PartialEq + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_and_load() {
        let mp: Map<u64, u64> = Map::new();
        mp.store(1, 1);
        assert_eq!(mp.load(1), Some(1));
    }

    #[test]
    fn test_store_and_get() {
        let mp: Map<u64, u64> = Map::new();
        mp.store(1, 1);
        let mut result = 0;
        mp.get(1, |v| {
            result = *v;
        });
        assert_eq!(result, 1);
    }

    #[test]
    fn test_delete() {
        let mp: Map<u64, u64> = Map::new();
        mp.store(1, 1);
        mp.delete(1);
        assert_eq!(mp.load(1), None);
    }

    #[test]
    fn test_compare_and_swap() {
        let mp: Map<u64, u64> = Map::new();
        mp.store(1, 1);
        assert!(mp.compare_and_swap(1, 1, 2));
        assert_eq!(mp.load(1), Some(2));
        assert!(!mp.compare_and_swap(1, 1, 3));
    }

    #[test]
    fn test_compare_and_delete() {
        let mp: Map<u64, u64> = Map::new();
        mp.store(1, 1);
        assert!(mp.compare_and_delete(1, 1));
        assert_eq!(mp.load(1), None);
        assert!(!mp.compare_and_delete(1, 1));
    }

    #[test]
    fn test_load_or_store() {
        let mp: Map<u64, u64> = Map::new();
        assert_eq!(mp.load_or_store(1, 1), 1);
        assert_eq!(mp.load_or_store(1, 2), 1);
    }

    #[test]
    fn test_load_and_delete() {
        let mp: Map<u64, u64> = Map::new();
        mp.store(1, 1);
        assert_eq!(mp.load_and_delete(1), Some(1));
        assert_eq!(mp.load_and_delete(1), None);
    }

    #[test]
    fn test_swap() {
        let mp: Map<u64, u64> = Map::new();
        assert_eq!(mp.swap(1, 1), None);
        assert_eq!(mp.swap(1, 2), Some(1));
        assert_eq!(mp.load(1), Some(2));
    }

    #[test]
    fn test_range() {
        let mp: Map<u64, u64> = Map::new();
        for i in 0..100 {
            mp.store(i, i);
        }
        let mut sum = 0;
        mp.range(|_, v| {
            sum += v;
            true
        });
        assert_eq!(sum, 4950);
    }
}
