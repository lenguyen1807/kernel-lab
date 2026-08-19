## 1D Block Tiling

In this kernel, "1D" means each thread computes multiple results along one
dimension of `C`: a vertical `TM x 1` strip.

With the current constants:

```text
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 8
TM      = 8
```

One thread block computes one `64 x 64` tile of `C`. Since each thread computes
`TM = 8` output elements, the block needs:

```text
nThreadsTile = BLOCK_M * BLOCK_N / TM
             = 64 * 64 / 8
             = 512 threads
```

There are three coordinate systems in the kernel:

```text
1. tRow/tCol
   -> which output strip this thread computes

2. tileRowA/tileColA
   -> which A shared-memory element this thread loads

3. tileRowB/tileColB
   -> which B shared-memory element this thread loads
```

### Output Coordinates

Threads are laid out as if they formed a logical grid with:

```text
rows = BLOCK_M / TM = 8
cols = BLOCK_N      = 64
```

So the mapping from a linear thread id is:

```text
tCol = threadIdx.x % BLOCK_N
tRow = threadIdx.x / BLOCK_N
```

For `threadIdx.x = 37`:

```text
tCol = 37 % 64 = 37
tRow = 37 / 64 = 0
```

This thread computes:

```text
C rows: 0 * 8 + 0..7 = 0..7
C col:  37
```

In code:

```cpp
innerRow = tRow * TM + i;
innerCol = tCol;
```

That is the whole idea of 1D coarsening: one thread owns one output column
inside the block tile, but several output rows.

### Shared Memory Loading

For each `K` phase, the block needs:

```text
As: BLOCK_M x BLOCK_K = 64 x 8 = 512 floats
Bs: BLOCK_K x BLOCK_N = 8 x 64 = 512 floats
```

The block has exactly 512 threads, so each thread loads one element of `A` and
one element of `B`.

For `A`:

```text
tileRowA = threadIdx.x / BLOCK_K
tileColA = threadIdx.x % BLOCK_K
```

For `threadIdx.x = 37`:

```text
tileRowA = 37 / 8 = 4
tileColA = 37 % 8 = 5
```

So this thread loads:

```text
As[4, 5]
```

For `B`:

```text
tileRowB = threadIdx.x / BLOCK_N
tileColB = threadIdx.x % BLOCK_N
```

For `threadIdx.x = 37`:

```text
tileRowB = 37 / 64 = 0
tileColB = 37 % 64 = 37
```

So this thread loads:

```text
Bs[0, 37]
```

During calculation, the thread walks across the `BLOCK_K` values:

```text
sum[i] += As[tRow * TM + i, k] * Bs[k, tCol]
```

So for `threadIdx.x = 37`, it accumulates 8 dot products:

```text
C[0, 37], C[1, 37], ..., C[7, 37]
```

## 2D Block Tiling

In this kernel, "2D" means each thread computes multiple results along both
dimensions of `C`: a `TM x TN` patch.

With the current constants:

```text
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 8
TM      = 8
TN      = 8
```

One thread block computes one `128 x 128` tile of `C`. Since each thread
computes `TM * TN = 64` output elements, the block needs:

```text
nThreadsTile = BLOCK_M * BLOCK_N / (TM * TN)
             = 128 * 128 / 64
             = 256 threads
```

### Output Coordinates

Threads are laid out as if they formed a logical grid with:

```text
rows = BLOCK_M / TM = 16
cols = BLOCK_N / TN = 16
```

So the mapping from a linear thread id is:

```text
tCol = threadIdx.x % (BLOCK_N / TN)
tRow = threadIdx.x / (BLOCK_N / TN)
```

For `threadIdx.x = 37`:

```text
tCol = 37 % 16 = 5
tRow = 37 / 16 = 2
```

This thread computes:

```text
C rows: 2 * 8 + 0..7 = 16..23
C cols: 5 * 8 + 0..7 = 40..47
```

In code:

```cpp
innerRow = tRow * TM + i;
innerCol = tCol * TN + j;
```

That is the whole idea of 2D coarsening: one thread owns a small rectangle of
the output tile.

### Shared Memory Loading

For each `K` phase, the block needs:

```text
As: BLOCK_M x BLOCK_K = 128 x 8   = 1024 floats
Bs: BLOCK_K x BLOCK_N = 8   x 128 = 1024 floats
```

The block only has 256 threads. If each thread loaded one `A` element and one
`B` element, it would only load 256 elements from each tile. That is why the
2D kernel uses `strideA` and `strideB`: each thread loads several elements.

For `A`:

```text
tileRowA = threadIdx.x / BLOCK_K
tileColA = threadIdx.x % BLOCK_K
strideA  = nThreadsTile / BLOCK_K
```

For `threadIdx.x = 37`:

```text
tileRowA = 37 / 8 = 4
tileColA = 37 % 8 = 5
strideA  = 256 / 8 = 32
```

So this thread loads:

```text
As[4,   5]
As[36,  5]
As[68,  5]
As[100, 5]
```

For `B`:

```text
tileRowB = threadIdx.x / BLOCK_N
tileColB = threadIdx.x % BLOCK_N
strideB  = nThreadsTile / BLOCK_N
```

For `threadIdx.x = 37`:

```text
tileRowB = 37 / 128 = 0
tileColB = 37 % 128 = 37
strideB  = 256 / 128 = 2
```

So this thread loads:

```text
Bs[0, 37]
Bs[2, 37]
Bs[4, 37]
Bs[6, 37]
```

### Calculation

The output patch has `TM * TN = 8 * 8 = 64` accumulators:

```cpp
float sum[TM * TN];
```

For each shared-memory `k`, the thread first reads:

```text
regA[i] = As[tRow * TM + i, k]      for i = 0..7
regB[j] = Bs[k, tCol * TN + j]      for j = 0..7
```

Then it computes the outer product:

```text
sum[i, j] += regA[i] * regB[j]
```

For `threadIdx.x = 37`, that means:

```text
regA reads rows 16..23 at column k
regB reads row k at columns 40..47
sum accumulates C[16..23, 40..47]
```

The key mental model:

```text
1D coarsening:
  one thread = one vertical strip of C

2D coarsening:
  one thread = one rectangular patch of C
```
