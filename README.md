# txn_lock
A futures-aware read-write lock for Rust which supports transaction-specific versioning

Example:
```rust
use futures::executor::block_on;
use txn_lock::*;

let lock = TxnLock::new(0, "zero");
{
    let mut guard = lock.try_write(&1).expect("write lock");
    *guard = "one";
}

block_on(lock.commit(&1));

assert_eq!(*lock.try_read(&0).expect("old value"), "zero");
assert_eq!(*lock.try_read(&1).expect("current value"), "one");
assert_eq!(*lock.try_read_exclusive(&2).expect("new value"), "one");

lock.rollback(&2);

{
    let mut guard = lock.try_write(&3).expect("write lock");
    *guard = "three";
}

lock.finalize(&2);

assert_eq!(lock.try_read(&0).unwrap_err(), Error::Outdated);
assert_eq!(*lock.try_read(&2).expect("old value"), "one");
assert_eq!(*lock.try_read(&3).expect("current value"), "three");
assert_eq!(lock.try_read(&4).unwrap_err(), Error::WouldBlock);
```
