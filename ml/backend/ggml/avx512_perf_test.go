package ggml

import (
	"fmt"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/ml"
)

// BenchmarkMatmulLLMRealistic tests realistic LLM matrix multiplication sizes
// Based on actual dimensions used in popular LLM architectures
func BenchmarkMatmulLLMRealistic(b *testing.B) {
	if runtime.GOARCH != "amd64" {
		b.Skip("Benchmarks only relevant for amd64 architecture")
	}
 
	cases := []struct {
		name   string
		M, N, K int
		model  string
	}{
		// Small models (7-8B parameters like LLaMA-2 7B, Mistral 7B)
		{"8B_attn_single_token", 1, 4096, 4096, "Single token attention projection"},
		{"8B_attn_batch_32", 32, 4096, 4096, "Batch-32 attention projection"},
		{"8B_ffn_up_single", 1, 11008, 4096, "Single token FFN up-projection"},
		{"8B_ffn_up_batch_32", 32, 11008, 4096, "Batch-32 FFN up-projection"},
		{"8B_ffn_down_single", 1, 4096, 11008, "Single token FFN down-projection"},
		{"8B_ffn_down_batch_32", 32, 4096, 11008, "Batch-32 FFN down-projection"},
		{"8B_output_logits_single", 1, 32000, 4096, "Single token output logits"},
		{"8B_output_logits_batch_32", 32, 32000, 4096, "Batch-32 output logits"},

		// Medium models (13-34B parameters like LLaMA-2 13B/34B)
		{"13B_attn_single_token", 1, 5120, 5120, "13B single token attention"},
		{"13B_attn_batch_64", 64, 5120, 5120, "13B batch-64 attention"},
		{"13B_ffn_up_single", 1, 13824, 5120, "13B single token FFN up"},
		{"13B_ffn_up_batch_64", 64, 13824, 5120, "13B batch-64 FFN up"},
		{"34B_attn_single_token", 1, 8192, 8192, "34B single token attention"},
		{"34B_ffn_up_batch_32", 32, 28672, 8192, "34B batch-32 FFN up"},
 
		// Large models (70B+ parameters like LLaMA-2 70B)
		{"70B_attn_single_token", 1, 8192, 8192, "70B single token attention"},
		{"70B_attn_batch_128", 128, 8192, 8192, "70B batch-128 attention"},
		{"70B_ffn_up_single", 1, 28672, 8192, "70B single token FFN up"},
		{"70B_ffn_up_batch_128", 128, 28672, 8192, "70B batch-128 FFN up"},
		{"70B_ffn_down_single", 1, 8192, 28672, "70B single token FFN down"},
		{"70B_ffn_down_batch_128", 128, 8192, 28672, "70B batch-128 FFN down"},
 
		// Very large models (GPT-3/4 class, 175B+)
		{"175B_attn_single", 1, 12288, 12288, "GPT-3 class single token attention"},
		{"175B_attn_batch_256", 256, 12288, 12288, "GPT-3 class batch-256 attention"},
		{"175B_ffn_up_single", 1, 49152, 12288, "GPT-3 class single FFN up"},
		{"175B_ffn_up_batch_256", 256, 49152, 12288, "GPT-3 class batch-256 FFN up"},
		{"175B_output_logits_single", 1, 50257, 12288, "GPT-3 output logits single"},
		{"175B_output_logits_batch_128", 128, 50257, 12288, "GPT-3 output logits batch-128"},
 
		// Prompt processing scenarios (large batch sizes)
		{"8B_prompt_512_tokens", 512, 4096, 4096, "Processing 512-token prompt"},
		{"8B_prompt_2048_tokens", 2048, 4096, 4096, "Processing 2048-token prompt"},
		{"70B_prompt_1024_tokens", 1024, 8192, 8192, "70B processing 1024-token prompt"},
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
			memoryMB := float64(tc.M*tc.K+tc.K*tc.N+tc.M*tc.N) * 4 / 1024 / 1024
 
			b.ReportMetric(opsPerSec/1e9, "GFLOPS")
			b.ReportMetric(memoryMB, "MB_data")
			b.ReportMetric(float64(tc.M), "M")
			b.ReportMetric(float64(tc.N), "N")
			b.ReportMetric(float64(tc.K), "K")
		})
	}
}
 
// TestLLMMatmulPerformanceReport generates a detailed performance report for LLM-realistic sizes
func TestLLMMatmulPerformanceReport(t *testing.T) {
	if runtime.GOARCH != "amd64" {
		t.Skip("Performance report only relevant for amd64 architecture")
	}
 
	if testing.Short() {
		t.Skip("Skipping performance report in short mode")
	}
 
	fmt.Println("\n=== LLM-Realistic Matrix Multiplication Performance Report ===")
 
	type TestCase struct {
		name   string
		M, N, K int
		desc   string
	}
 
	cases := []TestCase{
		// Single token generation (most common during inference)
		{"7B Single Token Attn", 1, 4096, 4096, "LLaMA-2 7B attention projection"},
		{"7B Single Token FFN-Up", 1, 11008, 4096, "LLaMA-2 7B FFN up-projection"},
		{"7B Single Token FFN-Down", 1, 4096, 11008, "LLaMA-2 7B FFN down-projection"},
		{"7B Single Token Logits", 1, 32000, 4096, "LLaMA-2 7B output logits"},
 
		// Batch processing (batch size 32 - typical for generation)
		{"7B Batch-32 Attn", 32, 4096, 4096, "LLaMA-2 7B attention batch-32"},
		{"7B Batch-32 FFN-Up", 32, 11008, 4096, "LLaMA-2 7B FFN up batch-32"},
		{"7B Batch-32 Logits", 32, 32000, 4096, "LLaMA-2 7B logits batch-32"},
 
		// Prompt processing (large M - initial prompt encoding)
		{"7B Prompt-512 Attn", 512, 4096, 4096, "LLaMA-2 7B encoding 512-token prompt"},
		{"7B Prompt-1024 Attn", 1024, 4096, 4096, "LLaMA-2 7B encoding 1024-token prompt"},
		{"7B Prompt-2048 Attn", 2048, 4096, 4096, "LLaMA-2 7B encoding 2048-token prompt"},
 
		// Larger models (70B class)
		{"70B Single Token Attn", 1, 8192, 8192, "LLaMA-2 70B attention single token"},
		{"70B Single Token FFN-Up", 1, 28672, 8192, "LLaMA-2 70B FFN up single token"},
		{"70B Batch-64 Attn", 64, 8192, 8192, "LLaMA-2 70B attention batch-64"},
		{"70B Prompt-512 Attn", 512, 8192, 8192, "LLaMA-2 70B encoding 512-token prompt"},
 
		// Very large models (GPT-3 class)
		{"GPT-3 Single Token Attn", 1, 12288, 12288, "GPT-3 175B attention single token"},
		{"GPT-3 Single Token FFN-Up", 1, 49152, 12288, "GPT-3 175B FFN up single token"},
		{"GPT-3 Single Token Logits", 1, 50257, 12288, "GPT-3 175B output logits"},
		{"GPT-3 Batch-128 Attn", 128, 12288, 12288, "GPT-3 175B attention batch-128"},
	}

	iterations := 20 // Fewer iterations for large matrices
 
	fmt.Printf("\n%-30s %8s %8s %8s | %10s %10s | %s\n",
		"Operation", "M", "N", "K", "Time(ms)", "GFLOPS", "Description")
	fmt.Println(strings.Repeat("-", 130))
 
	for _, tc := range cases {
		ctx := setup(t)
 
		A := ctx.Arange(0, float32(tc.M*tc.K), 1, ml.DTypeF32).Reshape(ctx, tc.K, tc.M)
		B := ctx.Arange(0, float32(tc.K*tc.N), 1, ml.DTypeF32).Reshape(ctx, tc.K, tc.N)
		C := A.Mulmat(ctx, B)
 
		// Warm up
		for range 3 {
			ctx.Forward(C).Compute(C)
		}
 
		// Measure
		start := time.Now()
		for range iterations {
			ctx.Forward(C).Compute(C)
		}
		elapsed := time.Since(start)
 
		avgTime := elapsed.Seconds() / float64(iterations) * 1000 // in ms
		ops := int64(tc.M) * int64(tc.N) * int64(tc.K) * 2       // multiply + add
		gflops := float64(ops) / (elapsed.Seconds() / float64(iterations)) / 1e9
 
		fmt.Printf("%-30s %8d %8d %8d | %10.3f %10.2f | %s\n",
			tc.name, tc.M, tc.N, tc.K, avgTime, gflops, tc.desc)
	}
 
	fmt.Println("\nNotes:")
	fmt.Println("- M = batch size (number of tokens being processed)")
	fmt.Println("- N = output dimension (embedding size or vocabulary size)")
	fmt.Println("- K = input dimension (embedding size)")
	fmt.Println("- Single token: M=1 (typical during autoregressive generation)")
	fmt.Println("- Batch processing: M=32-128 (parallel generation or beam search)")
	fmt.Println("- Prompt processing: M=512-2048 (encoding initial user prompt)")
	fmt.Println("- All tests use AVX-512 optimization if available and beneficial")
	fmt.Println()
}

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
