# Concurrent HashMap for Rust

This library provides a concurrent hash map implementation similar to Go's `sync.Map`, with additional performance optimizations for the `get` function.

## Features

- **Concurrency**: Safe concurrent access using fine-grained locking.
- **Performance**: Optimized `get` function for faster reads.
- **API Compatibility**: Supports all functions provided by Go's `sync.Map`.

## Usage


## Getting Started

```rust
// create a new hashmap
let mp: Map<u64, u64> = Map::new();

// set the value for key
mp.store(1, 1);
mp.store(1, 2);
mp.store(2, 1);

// get value by key
let v = mp.load(1)

// delete the value for a key.
m.delete(1)
```

## Benchmark

## License

`hashmap` source code is available under the GPL [License](/LICENSE).
