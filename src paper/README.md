## Sequential and parallel codes on moUKP discussed in the paper

#### • Sequential version

Run in a terminal: `julia hv_ukp_paper_CI_sequentiel.jl`

#### • Parallel version (50 chunks)

Run in a terminal: `julia --threads 8 hv_ukp_paper_CI_50chunks_VF.jl`

A version designed with a distribution on 8 threads (not presented in the paper) is also available:

Run in a terminal: `julia --threads 8 hv_ukp_paper_CI_8threads_VF.jl`
