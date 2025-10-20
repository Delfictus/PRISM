//! Circular buffer with thread-safe operations

use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Thread-safe circular buffer for data points
#[derive(Debug, Clone)]
pub struct CircularBuffer<T> {
    buffer: Arc<RwLock<VecDeque<T>>>,
    capacity: usize,
}

impl<T: Clone> CircularBuffer<T> {
    /// Create a new circular buffer with given capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Arc::new(RwLock::new(VecDeque::with_capacity(capacity))),
            capacity,
        }
    }

    /// Push an item, removing oldest if at capacity
    pub async fn push(&self, item: T) {
        let mut buffer = self.buffer.write().await;

        if buffer.len() >= self.capacity {
            buffer.pop_front();
        }

        buffer.push_back(item);
    }

    /// Push multiple items
    pub async fn push_batch(&self, items: Vec<T>) {
        let mut buffer = self.buffer.write().await;

        for item in items {
            if buffer.len() >= self.capacity {
                buffer.pop_front();
            }
            buffer.push_back(item);
        }
    }

    /// Get the most recent N items
    pub async fn get_recent(&self, n: usize) -> Vec<T> {
        let buffer = self.buffer.read().await;
        let len = buffer.len();
        let start = if n > len { 0 } else { len - n };

        buffer.iter().skip(start).cloned().collect()
    }

    /// Get all items in the buffer
    pub async fn get_all(&self) -> Vec<T> {
        let buffer = self.buffer.read().await;
        buffer.iter().cloned().collect()
    }

    /// Get current buffer size
    pub async fn len(&self) -> usize {
        let buffer = self.buffer.read().await;
        buffer.len()
    }

    /// Check if buffer is empty
    pub async fn is_empty(&self) -> bool {
        let buffer = self.buffer.read().await;
        buffer.is_empty()
    }

    /// Clear the buffer
    pub async fn clear(&self) {
        let mut buffer = self.buffer.write().await;
        buffer.clear();
    }

    /// Get buffer capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_circular_buffer_basic() {
        let buffer = CircularBuffer::new(3);

        buffer.push(1).await;
        buffer.push(2).await;
        buffer.push(3).await;

        assert_eq!(buffer.len().await, 3);
        assert_eq!(buffer.get_all().await, vec![1, 2, 3]);
    }

    #[tokio::test]
    async fn test_circular_buffer_overflow() {
        let buffer = CircularBuffer::new(3);

        buffer.push(1).await;
        buffer.push(2).await;
        buffer.push(3).await;
        buffer.push(4).await; // Should remove 1

        assert_eq!(buffer.len().await, 3);
        assert_eq!(buffer.get_all().await, vec![2, 3, 4]);
    }

    #[tokio::test]
    async fn test_get_recent() {
        let buffer = CircularBuffer::new(5);

        for i in 1..=5 {
            buffer.push(i).await;
        }

        assert_eq!(buffer.get_recent(3).await, vec![3, 4, 5]);
        assert_eq!(buffer.get_recent(10).await, vec![1, 2, 3, 4, 5]);
    }
}
