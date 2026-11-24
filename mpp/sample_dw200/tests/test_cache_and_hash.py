#!/usr/bin/env python3
"""Run production buffer handoffs/hash checks with separate CPU and DMA views."""
from pathlib import Path
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
RTSMART = HERE.parents[3]
MPP = RTSMART / 'mpp'
SOURCE = HERE.parent / 'src/sample_dw200.c'
HARNESS = r'''
#define main sample_main
#include "sample_dw200.c"
#undef main
#include <assert.h>
static unsigned char dma[4096];
static void *cpu;
static int synced;
k_s32 kd_mpi_sys_mmz_alloc_cached(k_u64 *phys,void **virt,const k_char *mmb,const k_char *zone,k_u32 size) {
    assert(size<=sizeof(dma)); *phys=0x20000000; cpu=*virt=malloc(size);
    memset(cpu,0xa5,size); memset(dma,0x5a,size); synced=0; return 0;
}
k_s32 kd_mpi_sys_mmz_alloc(k_u64 *phys,void **virt,const k_char *mmb,const k_char *zone,k_u32 size) {
    return kd_mpi_sys_mmz_alloc_cached(phys,virt,mmb,zone,size);
}
k_s32 kd_mpi_sys_mmz_flush_cache(k_u64 phys,void *virt,k_u32 size) {
    assert(virt==cpu); memcpy(dma,virt,size); synced=1; return 0;
}
k_s32 kd_mpi_sys_mmz_invalidate_cache(k_u64 phys,void *virt,k_u32 size) {
    assert(virt==cpu); memcpy(virt,dma,size); return 0;
}
k_s32 kd_mpi_sys_mmz_free(k_u64 phys,void *virt) { free(virt); cpu=NULL; return 0; }
int main(int argc,char **argv) {
    struct dw200_parameters params={0};
    params.output_res[0]=(struct dw200_resolution){.width=32,.height=16,.format=MEDIA_PIX_FMT_YUV420SP,.enable=1};
    struct video_buffer buffers[DW200_OUTPUT_COUNT]={0};
    assert(allocate_video_buffer(&buffers[0],resolution_size(&params.output_res[0]))==0);
    // Old cached aliases must be retired before allowing DMA to overwrite RAM.
    assert(synced);
    compare_outputs=true;
    for(unsigned api=0;api<3;api++) for(unsigned frame=0;frame<3;frame++) {
        memset(cpu,0xa5,buffers[0].size); // stale CPU cache left by the previous user
        memset(dma,0x42,buffers[0].size); // completed device output
        assert(save_low_level_outputs(&params,buffers,argv[1])==0);
        assert(((unsigned char*)cpu)[0]==0x42);
    }
    assert(compared_frames==9);
    dma[100]^=1; // must fail, including a first-frame-only difference
    assert(save_low_level_outputs(&params,buffers,argv[1])!=0);
    assert(compared_frames==9);
    free_video_buffer(&buffers[0]);
    puts("PASS production DW200 cache handoff and sequential hash mismatch detection");
}
'''
with tempfile.TemporaryDirectory(prefix='dw200-cache-') as tmp:
    tmp=Path(tmp)
    (tmp/'test.c').write_text(HARNESS)
    cmd=['cc','-std=gnu99','-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer',
         '-ffunction-sections','-fdata-sections']
    for path in (SOURCE.parent,HERE.parent/'include',MPP/'include',MPP/'include/comm',MPP/'userapps/api',RTSMART/'libs/3rd-party/cjson/cJSON'):
        cmd+=['-I',str(path)]
    cmd+=[str(tmp/'test.c'),'-Wl,--gc-sections','-o',str(tmp/'test')]
    subprocess.run(cmd,check=True)
    result = subprocess.run([str(tmp/'test'),str(tmp)], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, timeout=10)
    print(result.stdout, end='')
    result.check_returncode()
    assert result.stdout.count('DW200_HASH_PASS') == 9
    assert result.stdout.count('DW200_HASH_FAIL') == 1
    assert (tmp/'channel0_32x16_yuv420sp.bin').read_bytes() == b'\x42' * (32 * 16 * 3 // 2)
