use ndarray::Array;
use std::time::Instant;

#[test]
fn cuda_dot_finder() {
    let shape = 10;
    let a = Array::from_shape_simple_fn((shape, shape), || 1f32);
    let b = Array::from_shape_simple_fn((shape, shape), || 2f32);
    let start = Instant::now();
    let c = a.dot(&b);
    println!("{:?}", c);
    println!("Elapsed Time: {:?}", start.elapsed());
}