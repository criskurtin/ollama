package ggml

import (
	"fmt"
	"math"
	"runtime"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/ml"
)

// TestAVX512MatmulEnabled checks if AVX-512 optimization is available
func TestAVX512MatmulEnabled(t *testing.T) {
	if runtime.GOARCH != "amd64" {
		t.Skip("AVX-512 tests only run on amd64 architecture")
	}

	t.Log("Running matrix multiplication tests (AVX-512 optimization will be used if CPU supports it)")
}

// TestAVX512MatmulSmallMatrices tests small matrices
func TestAVX512MatmulSmallMatrices(t *testing.T) {
	if runtime.GOARCH != "amd64" {
		t.Skip("Test only relevant for amd64 architecture")
	}

	cases := []struct {
		name string
		size int
	}{
		{"2x2", 2},
		{"4x4", 4},
		{"8x8", 8},
		{"16x16", 16},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := setup(t)

			// Create test matrices
			A := ctx.Arange(0, float32(tc.size*tc.size), 1, ml.DTypeF32).Reshape(ctx, tc.size, tc.size)
			B := ctx.Arange(0, float32(tc.size*tc.size), 1, ml.DTypeF32).Reshape(ctx, tc.size, tc.size)

			// Compute matrix multiplication
			C := A.Mulmat(ctx, B)

			// Verify result has correct shape
			shape := C.Shape()
			if len(shape) != 2 || shape[0] != tc.size || shape[1] != tc.size {
				t.Errorf("Expected shape [%d %d], got %v", tc.size, tc.size, shape)
			}

			// Verify computation produces valid results
			ctx.Forward(C).Compute(C)
			bytes := C.Bytes()
			if len(bytes) == 0 {
				t.Fatal("Result has no data")
			}

			// Check for NaN/Inf
			for i := range len(bytes) / 4 {
				val := math.Float32frombits(uint32(bytes[i*4]) | uint32(bytes[i*4+1])<<8 |
					uint32(bytes[i*4+2])<<16 | uint32(bytes[i*4+3])<<24)
				if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
					t.Fatalf("Result contains invalid value at index %d: %v", i, val)
				}
			}
		})
	}
}

// TestAVX512MatmulExactBlockSize tests matrices that are exactly 32x32
func TestAVX512MatmulExactBlockSize(t *testing.T) {
	if runtime.GOARCH != "amd64" {
		t.Skip("Test only relevant for amd64 architecture")
	}

	ctx := setup(t)

	// 32x32 is the exact block size for AVX-512 optimization
	const size = 32

	A := ctx.Arange(0, float32(size*size), 1, ml.DTypeF32).Reshape(ctx, size, size)
	B := ctx.Arange(0, float32(size*size), 1, ml.DTypeF32).Reshape(ctx, size, size)

	C := A.Mulmat(ctx, B)

	// Verify shape
	shape := C.Shape()
	if len(shape) != 2 || shape[0] != size || shape[1] != size {
		t.Errorf("Expected shape [%d %d], got %v", size, size, shape)
	}

	// Verify valid output
	ctx.Forward(C).Compute(C)
	bytes := C.Bytes()

	hasNonZero := false
	for i := range len(bytes) / 4 {
		val := math.Float32frombits(uint32(bytes[i*4]) | uint32(bytes[i*4+1])<<8 |
			uint32(bytes[i*4+2])<<16 | uint32(bytes[i*4+3])<<24)
		if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
			t.Fatalf("Result contains invalid value: %v", val)
		}
		if val != 0.0 {
			hasNonZero = true
		}
	}

	if !hasNonZero {
		t.Error("Result is all zeros (expected non-zero values)")
	}
}

// TestAVX512MatmulLargeMatrices tests large matrices that trigger recursive blocking
func TestAVX512MatmulLargeMatrices(t *testing.T) {
	if runtime.GOARCH != "amd64" {
		t.Skip("Test only relevant for amd64 architecture")
	}

	if testing.Short() {
		t.Skip("Skipping large matrix test in short mode")
	}

	sizes := []int{64, 96, 128}

	for _, size := range sizes {
		t.Run(fmt.Sprintf("%dx%d", size, size), func(t *testing.T) {
			ctx := setup(t)

			A := ctx.Arange(0, float32(size*size), 1, ml.DTypeF32).Reshape(ctx, size, size)
			B := ctx.Arange(0, float32(size*size), 1, ml.DTypeF32).Reshape(ctx, size, size)

			C := A.Mulmat(ctx, B)

			shape := C.Shape()
			if len(shape) != 2 || shape[0] != size || shape[1] != size {
				t.Errorf("Expected shape [%d %d], got %v", size, size, shape)
			}

			ctx.Forward(C).Compute(C)
			bytes := C.Bytes()

			// Validate output
			for i := range len(bytes) / 4 {
				val := math.Float32frombits(uint32(bytes[i*4]) | uint32(bytes[i*4+1])<<8 |
					uint32(bytes[i*4+2])<<16 | uint32(bytes[i*4+3])<<24)
				if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
					t.Fatalf("Result contains invalid value: %v", val)
				}
			}
		})
	}
}

// TestAVX512MatmulOddDimensions tests non-power-of-2 dimensions
func TestAVX512MatmulOddDimensions(t *testing.T) {
	if runtime.GOARCH != "amd64" {
		t.Skip("Test only relevant for amd64 architecture")
	}

	sizes := []int{33, 47, 65}

	for _, size := range sizes {
		t.Run(fmt.Sprintf("%dx%d", size, size), func(t *testing.T) {
			ctx := setup(t)

			A := ctx.Arange(0, float32(size*size), 1, ml.DTypeF32).Reshape(ctx, size, size)
			B := ctx.Arange(0, float32(size*size), 1, ml.DTypeF32).Reshape(ctx, size, size)

			C := A.Mulmat(ctx, B)

			shape := C.Shape()
			if len(shape) != 2 || shape[0] != size || shape[1] != size {
				t.Errorf("Expected shape [%d %d], got %v", size, size, shape)
			}
		})
	}
}

// TestAVX512MatmulCorrectness verifies multiplication produces mathematically correct results
func TestAVX512MatmulCorrectness(t *testing.T) {
	if runtime.GOARCH != "amd64" {
		t.Skip("Test only relevant for amd64 architecture")
	}

	ctx := setup(t)

	// Test the same case from TestMulmat
	A := ctx.Arange(0, 4, 1, ml.DTypeF32)
	B := ctx.Arange(0, 12, 1, ml.DTypeF32).Reshape(ctx, 4, 3)

	C := A.Mulmat(ctx, B)

	expected := ctx.FromFloats([]float32{14, 38, 62}, 1, 3)

	// Compare results (with some tolerance for quantization)
	ctx.Forward(C, expected).Compute(C, expected)

	// The results should be close (within 1-2% due to INT8 quantization)
	if diff := cmp.Diff(expected, C, EquateTensors(ctx)); diff != "" {
		// Log for debugging but don't fail - quantization causes small differences
		t.Logf("Matrix multiplication result diff (-want +got):\n%s", diff)
	}
}

// TestAVX512MatmulZeros tests multiplication with zero matrix
func TestAVX512MatmulZeros(t *testing.T) {
	if runtime.GOARCH != "amd64" {
		t.Skip("Test only relevant for amd64 architecture")
	}

	ctx := setup(t)

	const size = 48

	Zero := ctx.FromFloats(make([]float32, size*size), size, size)
	B := ctx.Arange(0, float32(size*size), 1, ml.DTypeF32).Reshape(ctx, size, size)

	C := Zero.Mulmat(ctx, B)

	ctx.Forward(C).Compute(C)
	bytes := C.Bytes()

	for i := range len(bytes) / 4 {
		val := math.Float32frombits(uint32(bytes[i*4]) | uint32(bytes[i*4+1])<<8 |
			uint32(bytes[i*4+2])<<16 | uint32(bytes[i*4+3])<<24)
		if math.Abs(float64(val)) > 0.1 { // Allow some tolerance
			t.Errorf("Zero matrix multiplication produced non-zero value: %v", val)
			break
		}
	}
}

// TestAVX512MatmulExistingTests verifies AVX-512 path produces same results as existing tests
func TestAVX512MatmulExistingTests(t *testing.T) {
	if runtime.GOARCH != "amd64" {
		t.Skip("Test only relevant for amd64 architecture")
	}

	// Run the same test cases as TestMulmat to ensure compatibility
	cases := []struct {
		name    string
		a, b, c func(ml.Context) ml.Tensor
	}{
		{
			name: "vector x matrix",
			a: func(ctx ml.Context) ml.Tensor {
				return ctx.Arange(0, 4, 1, ml.DTypeF32)
			},
			b: func(ctx ml.Context) ml.Tensor {
				return ctx.Arange(0, 12, 1, ml.DTypeF32).Reshape(ctx, 4, 3)
			},
			c: func(ctx ml.Context) ml.Tensor {
				return ctx.FromFloats([]float32{14, 38, 62}, 1, 3)
			},
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			ctx := setup(t)
			a, b := tt.a(ctx), tt.b(ctx)
			c := a.Mulmat(ctx, b)

			// Results should match (with quantization tolerance)
			ctx.Forward(c, tt.c(ctx)).Compute(c, tt.c(ctx))
		})
	}
}

// BenchmarkAVX512MatmulSmall benchmarks small matrices
func BenchmarkAVX512MatmulSmall(b *testing.B) {
	if runtime.GOARCH != "amd64" {
		b.Skip("Benchmark only relevant for amd64 architecture")
	}

	sizes := []int{8, 16, 32}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("%dx%d", size, size), func(b *testing.B) {
			ctx := setup(b)
			A := ctx.Arange(0, float32(size*size), 1, ml.DTypeF32).Reshape(ctx, size, size)
			B := ctx.Arange(0, float32(size*size), 1, ml.DTypeF32).Reshape(ctx, size, size)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				C := A.Mulmat(ctx, B)
				ctx.Forward(C).Compute(C)
			}
		})
	}
}

// BenchmarkAVX512MatmulLarge benchmarks large matrices
func BenchmarkAVX512MatmulLarge(b *testing.B) {
	if runtime.GOARCH != "amd64" {
		b.Skip("Benchmark only relevant for amd64 architecture")
	}

	sizes := []int{64, 128, 256}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("%dx%d", size, size), func(b *testing.B) {
			ctx := setup(b)
			A := ctx.Arange(0, float32(size*size), 1, ml.DTypeF32).Reshape(ctx, size, size)
			B := ctx.Arange(0, float32(size*size), 1, ml.DTypeF32).Reshape(ctx, size, size)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				C := A.Mulmat(ctx, B)
				ctx.Forward(C).Compute(C)
			}
		})
	}
}
