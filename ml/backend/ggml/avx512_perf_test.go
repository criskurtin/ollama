package ggml

import (
	"fmt"
	"runtime"
	"testing"
	"time"

	"github.com/ollama/ollama/ml"
)

// BenchmarkMatmulComparison compares old vs new implementation across different matrix sizes
// Small matrices (< 32x32) use the old implementation (fallback)
// Large matrices (>= 32x32) use the AVX-512 optimized implementation if available
func BenchmarkMatmulComparison(b *testing.B) {
	if runtime.GOARCH != "amd64" {
		b.Skip("Benchmarks only relevant for amd64 architecture")
	}

	sizes := []struct {
		name     string
		size     int
		usesOpt  bool // whether this size uses AVX-512 optimization
	}{
		{"16x16_old_path", 16, false},        // Too small, uses old implementation
		{"32x32_new_path", 32, true},         // Exact block size, uses new if available
		{"64x64_new_path", 64, true},         // Medium matrix, uses new if available
		{"128x128_new_path", 128, true},      // Large matrix, uses new if available
		{"256x256_new_path", 256, true},      // Very large matrix, uses new if available
		{"512x512_new_path", 512, true},      // Extra large matrix, uses new if available
		{"1024x1024_new_path", 1024, true},   // Huge matrix, uses new if available
	}

	for _, tc := range sizes {
		b.Run(tc.name, func(b *testing.B) {
			// Create test matrices with realistic data (once, outside the benchmark loop)
			ctx := setupBenchmark(b)
			A := ctx.Arange(0, float32(tc.size*tc.size), 1, ml.DTypeF32).Reshape(ctx, tc.size, tc.size)
			B := ctx.Arange(0, float32(tc.size*tc.size), 1, ml.DTypeF32).Reshape(ctx, tc.size, tc.size)
			C := A.Mulmat(ctx, B)

			// Warm up
			for range 3 {
				ctx.Forward(C).Compute(C)
			}

			// Benchmark
			b.ResetTimer()
			for range b.N {
				ctx.Forward(C).Compute(C)
			}
			b.StopTimer()

			// Report metrics
			totalOps := int64(b.N) * int64(tc.size) * int64(tc.size) * int64(tc.size) * 2 // multiply + add
			opsPerSec := float64(totalOps) / b.Elapsed().Seconds()
			b.ReportMetric(opsPerSec/1e9, "GFLOPS")
			b.ReportMetric(float64(tc.size*tc.size*4*3)/1024/1024, "MB_per_op") // 3 matrices, 4 bytes each
		})
	}
}

// BenchmarkMatmulScaling tests how well the optimization scales with matrix size
func BenchmarkMatmulScaling(b *testing.B) {
	if runtime.GOARCH != "amd64" {
		b.Skip("Benchmarks only relevant for amd64 architecture")
	}

	// Test rectangular matrices too
	cases := []struct {
		name string
		M, N, K int
	}{
		{"square_32", 32, 32, 32},
		{"square_64", 64, 64, 64},
		{"square_128", 128, 128, 128},
		{"square_256", 256, 256, 256},
		{"tall_128x64", 128, 64, 64},
		{"wide_64x128", 64, 128, 64},
		{"deep_64x64x128", 64, 64, 128},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			ctx := setupBenchmark(b)

			// A is M×K, B is K×N, result is M×N
			A := ctx.Arange(0, float32(tc.M*tc.K), 1, ml.DTypeF32).Reshape(ctx, tc.K, tc.M)
			B := ctx.Arange(0, float32(tc.K*tc.N), 1, ml.DTypeF32).Reshape(ctx, tc.K, tc.N)
			C := A.Mulmat(ctx, B)

			// Warm up
			for range 3 {
				ctx.Forward(C).Compute(C)
			}

			b.ResetTimer()
			for range b.N {
				ctx.Forward(C).Compute(C)
			}
			b.StopTimer()

			totalOps := int64(b.N) * int64(tc.M) * int64(tc.N) * int64(tc.K) * 2
			opsPerSec := float64(totalOps) / b.Elapsed().Seconds()
			b.ReportMetric(opsPerSec/1e9, "GFLOPS")
		})
	}
}

// BenchmarkMatmulMemoryBandwidth tests memory bandwidth utilization
func BenchmarkMatmulMemoryBandwidth(b *testing.B) {
	if runtime.GOARCH != "amd64" {
		b.Skip("Benchmarks only relevant for amd64 architecture")
	}

	size := 512
	ctx := setupBenchmark(b)

	A := ctx.Arange(0, float32(size*size), 1, ml.DTypeF32).Reshape(ctx, size, size)
	B := ctx.Arange(0, float32(size*size), 1, ml.DTypeF32).Reshape(ctx, size, size)
	C := A.Mulmat(ctx, B)

	// Warm up
	for range 5 {
		ctx.Forward(C).Compute(C)
	}

	b.ResetTimer()
	for range b.N {
		ctx.Forward(C).Compute(C)
	}
	b.StopTimer()

	// Bytes read: 2 matrices * size * size * 4 bytes
	// Bytes written: 1 matrix * size * size * 4 bytes
	totalBytes := int64(b.N) * int64(size*size*4*3)
	bytesPerSec := float64(totalBytes) / b.Elapsed().Seconds()
	b.ReportMetric(bytesPerSec/1e9, "GB/s")
}

// BenchmarkMatmulThreadScaling tests how well the implementation scales with threads
func BenchmarkMatmulThreadScaling(b *testing.B) {
	if runtime.GOARCH != "amd64" {
		b.Skip("Benchmarks only relevant for amd64 architecture")
	}

	size := 256
	threadCounts := []int{1, 2, 4, 8, 16}

	for _, numThreads := range threadCounts {
		b.Run(fmt.Sprintf("threads_%d", numThreads), func(b *testing.B) {
			oldMaxProcs := runtime.GOMAXPROCS(numThreads)
			defer runtime.GOMAXPROCS(oldMaxProcs)

			ctx := setupBenchmark(b)

			A := ctx.Arange(0, float32(size*size), 1, ml.DTypeF32).Reshape(ctx, size, size)
			B := ctx.Arange(0, float32(size*size), 1, ml.DTypeF32).Reshape(ctx, size, size)
			C := A.Mulmat(ctx, B)

			// Warm up
			for range 3 {
				ctx.Forward(C).Compute(C)
			}

			b.ResetTimer()
			for range b.N {
				ctx.Forward(C).Compute(C)
			}
		})
	}
}

// TestMatmulPerformanceReport generates a detailed performance report
func TestMatmulPerformanceReport(t *testing.T) {
	if runtime.GOARCH != "amd64" {
		t.Skip("Performance report only relevant for amd64 architecture")
	}

	if testing.Short() {
		t.Skip("Skipping performance report in short mode")
	}

	fmt.Println("\n=== Matrix Multiplication Performance Report ===")

	sizes := []int{16, 32, 64, 128, 256, 512, 1024}
	iterations := 100

	fmt.Printf("%-10s %-15s %-15s %-15s %-15s\n", "Size", "Time (ms)", "GFLOPS", "Expected Path", "MB/op")
	fmt.Println("--------------------------------------------------------------------------------")

	for _, size := range sizes {
		ctx := setup(t)

		A := ctx.Arange(0, float32(size*size), 1, ml.DTypeF32).Reshape(ctx, size, size)
		B := ctx.Arange(0, float32(size*size), 1, ml.DTypeF32).Reshape(ctx, size, size)
		C := A.Mulmat(ctx, B)

		// Warm up
		for range 5 {
			ctx.Forward(C).Compute(C)
		}

		// Measure
		start := time.Now()
		for range iterations {
			ctx.Forward(C).Compute(C)
		}
		elapsed := time.Since(start)

		avgTime := elapsed.Seconds() / float64(iterations) * 1000 // in ms
		ops := int64(size) * int64(size) * int64(size) * 2       // multiply + add
		gflops := float64(ops) / (elapsed.Seconds() / float64(iterations)) / 1e9
		mbPerOp := float64(size*size*4*3) / 1024 / 1024

		expectedPath := "old (fallback)"
		if size >= 32 {
			expectedPath = "AVX-512 (if avail)"
		}

		fmt.Printf("%-10s %-15.3f %-15.2f %-15s %-15.2f\n",
			fmt.Sprintf("%dx%d", size, size),
			avgTime,
			gflops,
			expectedPath,
			mbPerOp,
		)
	}

	fmt.Println("\nNotes:")
	fmt.Println("- Matrices < 32x32 use the original implementation (too small for optimization)")
	fmt.Println("- Matrices >= 32x32 use AVX-512 optimized implementation if CPU supports it")
	fmt.Println("- GFLOPS = Giga Floating Point Operations Per Second")
	fmt.Println("- MB/op = Megabytes of data per operation (3 matrices × size × size × 4 bytes)")
	fmt.Println()
}

func setupBenchmark(b *testing.B) ml.Context {
	b.Helper()
	return setup(b)
}
