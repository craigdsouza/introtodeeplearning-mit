import ctypes

libs = ['libcuda.so.1', 'libcudart.so.12', 'libcublas.so.12', 'libcudnn.so.9', 'libcufft.so.11']
for lib in libs:
    try:
        ctypes.CDLL(lib)
        print(f'OK: {lib}')
    except OSError as e:
        print(f'MISSING: {lib}')
        print(f'  {e}')
