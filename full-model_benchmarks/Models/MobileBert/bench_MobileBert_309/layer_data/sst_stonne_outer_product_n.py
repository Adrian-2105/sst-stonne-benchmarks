import sst

# Define SST core options
sst.setProgramOption("timebase", "1 ps")
sst.setProgramOption("stopAtCycle", "10000s")

statLevel = 16
max_addr_gb = 16
tile_clk_mhz = 1

# Define the simulation components
comp_stonne = sst.Component("stonne1", "sstStonne.MAERI")
comp_stonne.addParams({
    "hardware_configuration" : "sparseflex_op_64mses_64_bw.cfg",
    "kernelOperation" : "outerProductGEMM",
    "GEMM_K" : 128,
    "GEMM_N" : 512,
    "GEMM_M" : 8,
    "GEMM_T_K" :4,
    "GEMM_T_N" : 1,
    "mem_init" : "mem_B-row_A-col.ini",
    "mem_matrix_c_file_name" : "result.out",
    "matrix_a_dram_address" : 0,
    "matrix_b_dram_address" : 4096,
    "matrix_c_dram_address" : 135168,
    "rowpointer_matrix_a_init" : "B_csr_rowpointer.in",
    "colpointer_matrix_a_init" : "B_csr_colpointer.in",
    "rowpointer_matrix_b_init" : "A_csc_rowpointer.in",
    "colpointer_matrix_b_init" : "A_csc_colpointer.in",


})

df_l1cache = sst.Component("l1cache", "memHierarchy.Cache")
df_l1cache.addParams({
    "access_latency_cycles" : "1",
    "cache_frequency" : str(tile_clk_mhz) + "GHz",
    "replacement_policy" : "lru",
    "coherence_protocol" : "MESI",
    "associativity" : "16",
    "cache_line_size" : "128",
    "banks" : "32",
    "max_requests_per_cycle" : "64",
    "tag_access_latency_cycles" : "0",
    "verbose" : 10,
    "debug" : 1,
    "debug_level" : 100,
    "L1" : "1",
    "cache_size" : "1024KiB"
})

df_memory = sst.Component("memory", "memHierarchy.MemController")
df_memory.addParams({
    "backing" : "mmap",
    "verbose" : 10,
    "debug" : 1,
    "debug_level" : 100,
    "clock" : str(tile_clk_mhz) + "GHz",
})

backend = df_memory.setSubComponent("backend", "memHierarchy.simpleMem")
backend.addParams({
    "access_time" : "80 ns",
    "mem_size" : str(max_addr_gb) + "GiB"
    #"mem_size" : str(1048576) + "B", # 1 MB
})

# Enable SST Statistics Outputs for this simulation
sst.setStatisticLoadLevel(statLevel)
sst.enableAllStatisticsForAllComponents({"type":"sst.AccumulatorStatistic"})
sst.setStatisticOutput("sst.statOutputTXT", { "filepath" : "output.csv" })

# Define the simulation links
link_df_cache_link = sst.Link("link_cpu_cache_link")
link_df_cache_link.connect( (comp_stonne, "cache_link", "1ps"), (df_l1cache, "high_network_0", "1ps") )
link_df_cache_link.setNoCut()

link_mem_bus_link = sst.Link("link_mem_bus_link")
link_mem_bus_link.connect( (df_l1cache, "low_network_0", "5ps"), (df_memory, "direct_link", "5ps") )



#sst.setStatisticLoadLevel(4)


# Enable statistics outputs


