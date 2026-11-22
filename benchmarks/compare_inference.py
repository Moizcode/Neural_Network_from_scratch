import numpy as np
import argparse
import copy
from neural_network import DenseLayer, NeuralNetwork
from neural_network.inference import InferenceModel
from benchmarks.inference import benchmark
import platform
import os


def run_case(model, inputs, rounds=5, warmups=20, repeats=1000):
    batch_size = len(inputs)
    variants = {
        "original": model,
        "no_cache": InferenceModel(model, batch_size),
        "buffers": InferenceModel(model, batch_size, reuse_buffers=True),
    }
    expected = model.predict(inputs).copy()
    for variant in variants.values():
        output = variant.predict(inputs)
        assert output.dtype == inputs.dtype and np.isfinite(output).all()
        np.testing.assert_allclose(output, expected, rtol=1e-4, atol=1e-5)

    names = list(variants)
    for round_index in range(rounds):
        offset = round_index % len(names)
        measurements = {}
        for name in names[offset:] + names[:offset]:
            times = np.array(benchmark(variants[name], inputs, warmups, repeats))
            assert np.isfinite(times).all() and np.all(times > 0)
            measurements[name] = (
                np.median(times), np.percentile(times, 95),
                batch_size * repeats / (times.sum() / 1000),
            )
        baseline = measurements["original"][0]
        for name in names:
            median, p95, throughput = measurements[name]
            print(
                f"| {inputs.dtype} | {batch_size} | {round_index + 1} | {name} | "
                f"{median:.6f} | {p95:.6f} | {throughput:.0f} | {baseline / median:.3f} |"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--large", action="store_true", help="Use a 1024 -> 2048 -> 1024 -> 100 network")
    parser.add_argument("--repeats", type=int, default=1000)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    seed = 42
    rounds, warmups, repeats = 5, 20, args.repeats
    sizes = (1024, 2048, 1024, 100) if args.large else (64, 128, 64, 10)
    np.random.seed(seed)
    print(f"Seed: {seed}; model: {sizes[0]} -> {sizes[1]} ReLU -> {sizes[2]} ReLU -> {sizes[3]} linear")
    print(f"Rounds: {rounds}; warmups per measurement: {warmups}; repeats: {repeats}")
    print(f"Python: {platform.python_version()}; NumPy: {np.__version__}")
    print(f"Platform: {platform.platform()}; CPU: {platform.processor() or platform.machine()}")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        print(f"{name}={os.environ.get(name, 'unset')} (requested limit)")
    np.show_config()
    print("Prediction time includes validation, allocations, and the buffered mode's output copy.")
    print("Ratio = original median / variant median within each round; below 1 means slower.")

    X = np.random.rand(128, sizes[0])

    model = NeuralNetwork([
        DenseLayer(sizes[0], sizes[1], 'relu'),
        DenseLayer(sizes[1], sizes[2], 'relu'),
        DenseLayer(sizes[2], sizes[3])
    ])
    parameters = sum(layer.weight_matrix.size + layer.bias.size for layer in model.layers)
    print(f"Parameters: {parameters:,}; float32 storage: {parameters * 4 / 2**20:.2f} MiB; float64: {parameters * 8 / 2**20:.2f} MiB")


    print("| Dtype | Batch | Round | Variant | Median (ms) | P95 (ms) | Samples/s | Ratio |")
    print("| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |")
    for dtype in (np.float32, np.float64):
        typed_model = copy.deepcopy(model)
        for layer in typed_model.layers:
            layer.weight_matrix = layer.weight_matrix.astype(dtype)
            layer.bias = layer.bias.astype(dtype)
        typed_inputs = X.astype(dtype)
        np.testing.assert_allclose(
            typed_model.predict(typed_inputs), model.predict(X), rtol=1e-4, atol=1e-5,
        )
        for batch_size in (1, 8, 32, 128):
            run_case(typed_model, typed_inputs[:batch_size], rounds, warmups, repeats)


if __name__ == "__main__":
    main()
