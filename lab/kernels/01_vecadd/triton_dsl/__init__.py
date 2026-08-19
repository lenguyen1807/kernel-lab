import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()
BLOCK_SIZE = 256


@triton.jit
def triton_add_kernel(
    x_ptr,  # *Pointer* to first input vector.
    y_ptr,  # *Pointer* to second input vector.
    output_ptr,  # *Pointer* to output vector.
    n_elements,  # Size of the vector.
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y

    tl.store(output_ptr + offsets, output, mask=mask)


# https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html
def vecadd_triton(a: torch.Tensor, b: torch.Tensor):
    c = torch.empty_like(a)
    assert a.device == DEVICE and b.device == DEVICE and c.device == DEVICE
    n_elements = c.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    triton_add_kernel[grid](a, b, c, n_elements, BLOCK_SIZE=1024)

    # No torch.cuda.synchronize() here: the launch is async like every other
    # variant, and syncing per call would serialise the benchmark loop.
    return c
