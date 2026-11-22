import numpy as np
import copy
import os
import platform
from neural_network import NeuralNetwork, DenseLayer
import time

def benchmark(model,inputs,warmups=20,repeats = 200):
    measurements = []
    for i in range(warmups):
        model.predict(inputs)


    for i in range(repeats):
        t0 = time.perf_counter_ns()
        model.predict(inputs)
        t1 = time.perf_counter_ns()
        measurements.append((t1-t0)/1e6)


    return measurements


def main():
    seed, warmups, repeats = 42, 20, 200
    np.random.seed(seed)
    print(f"Seed: {seed}; model: 64 -> 128 ReLU -> 64 ReLU -> 10 linear")
    print(f"Warmups: {warmups}; repeats: {repeats}")
    print(f"Python: {platform.python_version()}; NumPy: {np.__version__}")
    print(f"Platform: {platform.platform()}; CPU: {platform.processor() or platform.machine()}")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        print(f"{name}={os.environ.get(name, 'unset')} (requested limit)")
    np.show_config()
    print("Model prediction time, including forward caches and allocations")

    X = np.random.rand(128,64)

    model = NeuralNetwork([
        DenseLayer(64,128,'relu'),
        DenseLayer(128,64,'relu'),
        DenseLayer(64,10)
    ])


    records = {}
    out_records = {}
    for dtype in [np.float32, np.float64]:
        copy_model = copy.deepcopy(model)
        for layer in copy_model.layers:
            layer.weight_matrix = layer.weight_matrix.astype(dtype)
            layer.bias = layer.bias.astype(dtype)

        X_cast = X.astype(dtype)
        parameters = [
            value for layer in copy_model.layers
            for value in (layer.weight_matrix, layer.bias)
        ]
        assert all(value.dtype == dtype for value in parameters)
        original_parameters = [value.copy() for value in parameters]

        for batch_size in [1,8,32,128]:
            inputs = X_cast[:batch_size]
            assert inputs.dtype == dtype
            original_inputs = inputs.copy()

            out = copy_model.predict(inputs)
            assert out.dtype == dtype and np.isfinite(out).all()
    
            out2 = copy_model.predict(inputs)
            assert np.array_equal(out, out2)

            measurements = benchmark(copy_model, inputs, warmups, repeats)
            assert np.isfinite(measurements).all() and np.all(np.array(measurements) > 0)
            np.testing.assert_array_equal(inputs, original_inputs)
            for value, original in zip(parameters, original_parameters):
                np.testing.assert_array_equal(value, original)

            median = np.median(measurements)
            p95 = np.percentile(measurements,95)
            throughput = batch_size * len(measurements) / (sum(measurements) / 1e3)
            records[(dtype,batch_size)] = {"median":median,"p95":p95, "throughput":throughput}
            out_records[(dtype,batch_size)] = out


    print("| Dtype | Batch | Median (ms) | P95 (ms) | Throughput (samples/s) |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for (dtype, batch_size), value in records.items():
        assert np.isfinite(list(value.values())).all() and all(v > 0 for v in value.values())
        print(f"| {np.dtype(dtype).name} | {batch_size} | {value['median']:.6f} | {value['p95']:.6f} | {value['throughput']:.0f} |")

    for batch_size in [1,8,32,128]:
        np.testing.assert_allclose(
            out_records[(np.float32, batch_size)], out_records[(np.float64, batch_size)],
            rtol=1e-4, atol=1e-5,
        )
    print("Float32 and float64 outputs agree within tolerance for every batch size.")

if __name__ == "__main__":
    main()
