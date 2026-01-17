//! TimeSeries container for indicator output.

/// A time-indexed series of values, typically used for indicator output.
#[derive(Debug, Clone)]
pub struct TimeSeries<T> {
    /// The values in the series, aligned with candle indices.
    values: Vec<Option<T>>,
    /// Starting index (offset from the first candle).
    start_index: usize,
}

impl<T> TimeSeries<T> {
    /// Creates a new empty TimeSeries.
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            start_index: 0,
        }
    }

    /// Creates a TimeSeries with the given values starting at the specified index.
    pub fn with_offset(values: Vec<Option<T>>, start_index: usize) -> Self {
        Self {
            values,
            start_index,
        }
    }

    /// Returns the starting index of this series.
    pub fn start_index(&self) -> usize {
        self.start_index
    }

    /// Returns the number of values in this series.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns true if this series is empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Gets the value at the given candle index, if available.
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.start_index {
            return None;
        }
        let local_idx = index - self.start_index;
        self.values.get(local_idx).and_then(|v| v.as_ref())
    }

    /// Returns an iterator over (index, value) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (usize, &T)> {
        self.values
            .iter()
            .enumerate()
            .filter_map(|(i, v)| v.as_ref().map(|val| (self.start_index + i, val)))
    }

    /// Returns the underlying values slice.
    pub fn values(&self) -> &[Option<T>] {
        &self.values
    }
}

impl<T> Default for TimeSeries<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> TimeSeries<T> {
    /// Creates a TimeSeries from a slice of values, all present.
    pub fn from_values(values: &[T], start_index: usize) -> Self {
        Self {
            values: values.iter().cloned().map(Some).collect(),
            start_index,
        }
    }
}
